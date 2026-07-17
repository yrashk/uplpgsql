/*-------------------------------------------------------------------------
 *
 * upl_compile_stmts.cpp
 *		AST-to-LLVM-IR compilation pipeline — statement half of
 *		uplpgsql::function_compiler (see upl_compiler.hpp).
 *
 *		This file is the core of the JIT compiler.  It contains:
 *
 *		1. JIT Heuristic (uplpgsql_should_jit / jit_score_*)
 *		   Walks the AST and scores each statement.  Loops score positive
 *		   (amplified by depth), SPI-dominated statements score negative.
 *		   Returns true if score > 0.
 *
 *		2. Main Compilation Entry (uplpgsql_compile_function)
 *		   Constructs a function_compiler and runs the core pipeline
 *		   (upl_compile_function), which creates the LLVM context/module/
 *		   builder, registers types and runtime functions, compiles the
 *		   function body, verifies, optimizes (O3), and hands the module
 *		   to OrcJIT.
 *
 *		3. Statement Compilation (function_compiler::compile_*)
 *		   One method per PL/pgSQL statement type.  Control flow (IF, WHILE,
 *		   LOOP, FOR, CASE, EXIT/CONTINUE) is compiled to native LLVM basic
 *		   blocks and branches.  SPI-dependent operations delegate to runtime
 *		   helpers or directly to forked executor functions via embedded
 *		   function pointers.
 *
 *		4. Runtime Function Registration (register_runtime_funcs)
 *		   Declares all uplpgsql_rt_* functions in the LLVM module so they
 *		   can be called from generated IR.  OrcJIT resolves them at link
 *		   time via the process symbol search generator.
 *
 *		Helper utilities:
 *		  - llvm_const_int32/llvm_const_ptr: create LLVM constant values
 *		  - append_block: append a basic block to the current function
 *		  - call_fn: call a registered runtime function
 *		  - call_exec: call an exec_* function directly via embedded
 *		    pointer (bypasses the uplpgsql_rt_* wrapper layer)
 *
 *
 * Copyright (c) 2003-2014, Jonah H. Harris <jonah.harris@gmail.com>
 * Copyright (c) 2014-2026, NEXTGRES, LLC. <oss@nextgres.com>
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License in LICENSE or at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */
#include "upl_compiler.hpp"

#include "cppgres.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "catalog/pg_type_d.h"
#include "executor/spi_priv.h"
#include "utils/expandedrecord.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/plancache.h"
#ifdef __cplusplus
}
#endif

#include <algorithm>

namespace uplpgsql
{

/* Helper to create LLVM constant values */
static inline llvm::Value *
llvm_const_int32(UPL_compile_ctx *ctx, int32 val)
{
	return llvm::ConstantInt::get(ctx->types[UPL_INT32], val, false);
}

static inline llvm::Value *
llvm_const_ptr(UPL_compile_ctx *ctx, void *ptr)
{
	return llvm::ConstantExpr::getIntToPtr(
		llvm::ConstantInt::get(ctx->types[UPL_INT64], (uintptr_t) ptr, false),
		ctx->types[UPL_PTR]);
}

static inline llvm::BasicBlock *
append_block(UPL_compile_ctx *ctx, const char *name)
{
	return llvm::BasicBlock::Create(*ctx->context, name, ctx->function);
}

/* Call a registered runtime function */
llvm::Value *
function_compiler::call_fn(UPLpgSQL_rt_func which,
						   llvm::ArrayRef<llvm::Value *> args)
{
	return ctx->builder->CreateCall(ctx->rt_funcs[which], args, "");
}

/*
 * Call an exec_* function directly via embedded function pointer.
 *
 * This is a key optimization for "thin" runtime wrappers.  Many
 * uplpgsql_rt_* functions do nothing but extract plstate from estate
 * and forward to exec_stmt_*.  Instead of calling through the wrapper,
 * we embed the exec_stmt_* function's C address as an LLVM constant
 * and call it directly with (plstate, stmt) arguments.
 *
 * This saves one function call overhead per statement execution and
 * allows LLVM's optimizer to reason about the call more effectively.
 *
 * The function pointer is resolved at JIT compile time via C's normal
 * symbol resolution (the exec_* functions are linked into our .so).
 * The resulting native code contains a direct call instruction to the
 * known address.
 *
 * For i32-returning functions (exec_stmt_*): returns i32.
 * For void-returning functions: returns NULL llvm::Value *.
 *
 * Native arrays are synced to their PG Datums first.  Every exec_* callee
 * reads variables as Datums, and a native array's Datum is stale between
 * escapes, so anything reaching the interpreter must see marshalled data —
 * RAISE, FOREACH, EXECUTE, PERFORM, cursors, and the rest.  Syncing at this
 * one point covers them all; the alternative is remembering to do it in each
 * of the two dozen callers, which is how RAISE/FOREACH/EXECUTE were missed.
 * So this stays the default: correct for any callee, including ones whose
 * variable reads are not visible here.
 *
 * The wasted marshal it costs a callee that reads no arrays is not always
 * acceptable, though.  It is proportional to the size of every native array
 * in the function, and it is paid every time the statement runs, so a single
 * uncompilable expression in a hot loop -- "p := greatest(a,b,c)", which is
 * a MinMaxExpr and reaches neither tier -- marshals megabytes per iteration
 * for arrays it never touches.  Where the statement's expression is known,
 * call_exec_nosync() plus sync_native_arrays_for_expr()
 * syncs only what that expression actually reads.
 */
llvm::Value *
function_compiler::call_exec_nosync(void *fn_addr,
									llvm::Type *ret_type,
									llvm::ArrayRef<llvm::Value *> args)
{
	llvm::Type	   *param_types[4];
	llvm::FunctionType *fn_type;
	llvm::Value	   *fn_ptr;
	size_t			i;
	const char	   *call_name;

	Assert(args.size() <= 4);

	for (i = 0; i < args.size(); i++)
		param_types[i] = args[i]->getType();

	fn_type = llvm::FunctionType::get(ret_type,
									  llvm::ArrayRef<llvm::Type *>(param_types,
																   args.size()),
									  false);
	fn_ptr = llvm::ConstantExpr::getIntToPtr(
		llvm::ConstantInt::get(ctx->types[UPL_INT64],
							   (uintptr_t) fn_addr, false),
		llvm::PointerType::get(*ctx->context, 0));

	call_name = (ret_type == ctx->types[UPL_VOID]) ? "" : "exec.rc";

	return ctx->builder->CreateCall(llvm::FunctionCallee(fn_type, fn_ptr),
									args, call_name);
}

/*
 * Call an exec_* function, marshalling every native array first.
 *
 * The safe default, and what every caller without a known expression uses.
 */
llvm::Value *
function_compiler::call_exec(void *fn_addr,
							 llvm::Type *ret_type,
							 llvm::ArrayRef<llvm::Value *> args)
{
	sync_native_arrays();

	return call_exec_nosync(fn_addr, ret_type, args);
}

/* ----------------------------------------------------------------
 * Core compilation callbacks
 *
 * These are the UPL_callbacks implementations that the core engine
 * calls when compiling control flow (IF/WHILE/LOOP/FOR/CASE/BLOCK).
 * They bridge from the core's opaque void* parameters to the
 * PL/pgSQL-specific types.
 * ----------------------------------------------------------------
 */

/* Callback: compile a statement list */
void
function_compiler::cb_compile_stmts(UPL_compile_ctx *ctx, void *stmts)
{
	self(ctx)->compile_stmts((List *) stmts);
}

/* Callback: try to compile a boolean expression natively */
bool
function_compiler::cb_try_compile_bool(UPL_compile_ctx *ctx,
									   void *expr,
									   llvm::Value **result_out)
{
	llvm::Value *cond = self(ctx)->try_compile_bool((UPLpgSQL_expr *) expr);

	if (cond == NULL)
		return false;

	*result_out = cond;
	return true;
}

/* Callback: assign expression to variable (for CASE test expr) */
void
function_compiler::cb_assign_expr(UPL_compile_ctx *ctx, int varno, void *expr)
{
	self(ctx)->assign_expr(varno, expr);
}

/* Assign expression to variable (for CASE test expr) */
void
function_compiler::assign_expr(int varno, void *expr)
{
	/*
	 * Go through uplpgsql_rt_case_assign_test rather than calling
	 * exec_assign_expr directly: a simple CASE's temporary is an INT4
	 * placeholder until the runtime retypes it to whatever the test
	 * expression actually is, exactly as exec_stmt_case does.  Without that,
	 * CASE over anything but an integer fails to coerce.
	 */
	llvm::Value *args[3];

	args[0] = ctx->function->getArg(0);
	args[1] = upl_const_int32(ctx, varno);
	args[2] = upl_const_ptr(ctx, expr);

	call_exec((void *) uplpgsql_rt_case_assign_test,
					   ctx->types[UPL_VOID], args);
}

/* ----------------------------------------------------------------
 * Compilation pipeline hooks
 *
 * These implement UPL_compile_hooks for the PL/pgSQL driver.
 * Called by upl_compile_function() during the compilation pipeline.
 * ----------------------------------------------------------------
 */

/* Hook trampolines: recover the compiler from ctx and delegate */
void
function_compiler::cb_register_rt_funcs(UPL_compile_ctx *ctx)
{
	self(ctx)->register_runtime_funcs();
}

void
function_compiler::cb_setup_entry(UPL_compile_ctx *ctx)
{
	self(ctx)->setup_entry();
}

void
function_compiler::cb_compile_body(UPL_compile_ctx *ctx)
{
	self(ctx)->compile_body();
}

void
function_compiler::cb_compile_block_exceptions(UPL_compile_ctx *ctx,
											   void *exception_data)
{
	self(ctx)->compile_block_exceptions(exception_data);
}

/* Hook: driver-specific entry setup */
void
function_compiler::setup_entry()
{
	UPLpgSQL_function *func = func_;
	llvm::Value *off, *gep;

	/* Load plstate = estate->uplpgsql_estate */
	off = llvm::ConstantInt::get(ctx->types[UPL_INT64],
					   offsetof(UPLpgSQL_exec_state, uplpgsql_estate), false);
	gep = ctx->builder->CreateGEP(ctx->types[UPL_INT8],
							 ctx->estate_ref, off, "plstate.ptr");
	plstate_ref_ = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
									  gep, "plstate");

	/* Native array analysis + allocas */
	analyze_native_arrays(func);
	{
		for (UPLpgSQL_native_array &na_ref : native_arrays_)
		{
			UPLpgSQL_native_array *na = &na_ref;
			char name[64];

			if (na->elemtype == INT4OID)
				na->llvm_elemtype = ctx->types[UPL_INT32];
			else if (na->elemtype == INT8OID)
				na->llvm_elemtype = ctx->types[UPL_INT64];
			else
				na->llvm_elemtype = ctx->types[UPL_DOUBLE];

			snprintf(name, sizeof(name), "na%d.data", na->dno);
			na->data_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_PTR], nullptr, name);
			ctx->builder->CreateStore(llvm::Constant::getNullValue(ctx->types[UPL_PTR]),
						   na->data_ptr);

			snprintf(name, sizeof(name), "na%d.len", na->dno);
			na->len_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT32], nullptr, name);
			ctx->builder->CreateStore(llvm_const_int32(ctx, 0),
						   na->len_ptr);

			/*
			 * Lower bound.  1 until something says otherwise; from_datum
			 * reports the real one for an array that has a different base.
			 */
			snprintf(name, sizeof(name), "na%d.lb", na->dno);
			na->lb_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT32], nullptr, name);
			ctx->builder->CreateStore(llvm_const_int32(ctx, 1),
						   na->lb_ptr);

			/* Per-element NULL flags; NULL pointer means "no element is" */
			snprintf(name, sizeof(name), "na%d.nulls", na->dno);
			na->nulls_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_PTR], nullptr, name);
			ctx->builder->CreateStore(llvm::Constant::getNullValue(ctx->types[UPL_PTR]),
						   na->nulls_ptr);

			/*
			 * Allocated capacity of data, in elements.  An append bumps len
			 * up to this; past it the buffers grow through
			 * uplpgsql_rt_native_array_reserve.
			 */
			snprintf(name, sizeof(name), "na%d.cap", na->dno);
			na->cap_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT32], nullptr, name);
			ctx->builder->CreateStore(llvm_const_int32(ctx, 0),
						   na->cap_ptr);

			/* 1 when data was palloc'd; 0 for the array_fill stack buffer */
			snprintf(name, sizeof(name), "na%d.onheap", na->dno);
			na->is_heap_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT8], nullptr, name);
			ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT8], 0, false),
						   na->is_heap_ptr);
		}
	}
}

/* Hook: compile the function body */
void
function_compiler::compile_body()
{
	UPLpgSQL_function *func = func_;

	compile_block(func->action);
}

/*
 * Emit RT_INIT_VAR for each of a block's declared variables.
 *
 * uplpgsql_rt_init_var() evaluates the DECLARE default with
 * exec_assign_expr(), writing the variable's PG Datum.  For a native array
 * that leaves flat memory untouched — data_ptr NULL, len_ptr 0 — so a later
 * native subscript would fail its bounds check against a length of zero
 * ("array subscript 1 out of range [1..0]") even though the variable holds a
 * perfectly good array.  Reload flat memory from the Datum init_var just
 * wrote.  Arrays with no default refresh from a NULL Datum, which yields the
 * same empty state the entry block already established.
 *
 * The driver emits this rather than the core (which would otherwise do it in
 * upl_emit_block) because refreshing native arrays is language-specific.
 */
void
function_compiler::emit_init_vars(llvm::ArrayRef<int> initvarnos)
{
	llvm::Value *estate_ref = ctx->function->getArg(0);

	for (int dno : initvarnos)
	{
		UPLpgSQL_native_array *na;
		llvm::Value *args[] = {
			estate_ref,
			upl_const_int32(ctx, dno)
		};

		call_fn(RT_INIT_VAR, args);

		na = find_native_array(dno);
		if (na != NULL)
		{
			elog(DEBUG1, "uplpgsql: native array from_datum dno %d (init var)",
				 dno);
			emit_refresh_native_array(na);
		}
	}
}

/* ----------------------------------------------------------------
 * Exception handling callback
 *
 * This is called from upl_emit_block() when a block has exception
 * handlers.  It contains the full sigsetjmp pattern, handler dispatch,
 * and rethrow logic — all PL/pgSQL-specific.
 * ----------------------------------------------------------------
 */
void
function_compiler::compile_block_exceptions(void *exception_data)
{
	UPLpgSQL_stmt_block *stmt = (UPLpgSQL_stmt_block *) exception_data;
	llvm::Value		*estate_ref = ctx->estate_ref;
	llvm::Value		*frame_ptr;
	llvm::Value		*sjrc;
	llvm::Value		*is_try;
	llvm::BasicBlock	*try_enter_bb;
	llvm::BasicBlock	*catch_enter_bb;
	llvm::BasicBlock	*try_exit_bb;
	llvm::BasicBlock	*exception_return_bb;
	llvm::BasicBlock	*handler_return_bb;
	llvm::BasicBlock	*handler_exit_bb;
	llvm::BasicBlock	*after_block_bb;
	llvm::BasicBlock	*rethrow_bb;
	llvm::BasicBlock	*saved_return_bb;
	llvm::Value		*handler_idx;
	llvm::SwitchInst   *switch_inst;
	List			   *exc_list = stmt->exceptions->exc_list;
	int					num_handlers;
	int					i;

	ctx->has_exceptions = true;
	num_handlers = list_length(exc_list);

	/* Create all basic blocks upfront */
	try_enter_bb = upl_append_block(ctx, "exc.try_enter");
	catch_enter_bb = upl_append_block(ctx, "exc.catch_enter");
	try_exit_bb = upl_append_block(ctx, "exc.try_exit");
	exception_return_bb = upl_append_block(ctx, "exc.return");
	handler_return_bb = upl_append_block(ctx, "exc.handler_return");
	handler_exit_bb = upl_append_block(ctx, "exc.handler_exit");
	after_block_bb = upl_append_block(ctx, "exc.after");
	rethrow_bb = upl_append_block(ctx, "exc.rethrow");

	/* 1. Push exception frame (allocates frame, begins subtxn) */
	{
		llvm::Value *args[] = {
			estate_ref,
			upl_const_ptr(ctx, stmt)
		};
		frame_ptr = call_fn(RT_EXCEPTION_PUSH_FRAME, args);
	}

	/* 2. Call sigsetjmp(frame, 0) — frame IS the jmpbuf (first field) */
	{
		llvm::Value *args[] = {
			frame_ptr,
			upl_const_int32(ctx, 0)
		};
		sjrc = ctx->builder->CreateCall(ctx->sigsetjmp_fn, args, "sjrc");
	}

	/* 3. Branch: rc == 0 -> try, else -> catch */
	is_try = ctx->builder->CreateICmpEQ(sjrc,
						   upl_const_int32(ctx, 0), "is_try");
	ctx->builder->CreateCondBr(is_try, try_enter_bb, catch_enter_bb);

	/* === TRY ENTER === */
	ctx->builder->SetInsertPoint(try_enter_bb);

	/* Arm the exception frame */
	{
		llvm::Value *args[] = { estate_ref, frame_ptr };
		call_fn(RT_EXCEPTION_ARM, args);
	}

	/* Initialize declared variables */
	emit_init_vars(llvm::ArrayRef<int>(stmt->initvarnos, stmt->n_initvars));

	/*
	 * Compile body statements.  Redirect RETURN to our exception_return_bb
	 * so we can commit the subtransaction before actually returning.
	 */
	saved_return_bb = ctx->return_bb;
	ctx->return_bb = exception_return_bb;

	/*
	 * An EXIT/CONTINUE in the try body whose target lies outside this block
	 * jumps across our frame, so upl_emit_loop_exit() must release it on the
	 * way — via TRY_EXIT, since on that path the subtransaction commits.
	 * Push the cleanup before the block's own label: the label's exit path
	 * (try_exit_bb) releases the frame itself, so an "EXIT <label>" of this
	 * very block must not unwind it a second time.
	 */
	{
		llvm::Value *cleanup_args[] = { frame_ptr };

		upl_push_cleanup(ctx, RT_EXCEPTION_TRY_EXIT, cleanup_args);
	}

	/*
	 * "EXIT <label>" out of the try body must still release the
	 * subtransaction, so it targets try_exit_bb rather than jumping straight
	 * to after_block_bb.
	 */
	if (stmt->label != NULL)
		upl_push_block_label(ctx, stmt->label, try_exit_bb);

	compile_stmts(stmt->body);

	if (stmt->label != NULL)
		upl_pop_loop(ctx);

	upl_pop_cleanup(ctx);

	ctx->return_bb = saved_return_bb;

	/* Fall through to try_exit */
	ctx->builder->CreateBr(try_exit_bb);

	/* === EXCEPTION RETURN (RETURN inside try body) === */
	ctx->builder->SetInsertPoint(exception_return_bb);
	{
		llvm::Value *args[] = { estate_ref, frame_ptr };
		call_fn(RT_EXCEPTION_TRY_EXIT, args);
	}
	ctx->builder->CreateBr(saved_return_bb);

	/* === TRY EXIT (normal completion) === */
	ctx->builder->SetInsertPoint(try_exit_bb);
	{
		llvm::Value *args[] = { estate_ref, frame_ptr };
		call_fn(RT_EXCEPTION_TRY_EXIT, args);
	}
	ctx->builder->CreateBr(after_block_bb);

	/* === CATCH ENTER === */
	ctx->builder->SetInsertPoint(catch_enter_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			upl_const_ptr(ctx, stmt),
			frame_ptr
		};
		handler_idx = call_fn(RT_EXCEPTION_CATCH, args);
	}

	/* Switch on handler_idx: -1 -> rethrow, 0..N-1 -> handler blocks */
	switch_inst = ctx->builder->CreateSwitch(handler_idx,
								   rethrow_bb, num_handlers);

	/*
	 * A RETURN inside a handler body must still run HANDLER_DONE before it
	 * leaves the block: that restores cur_error, pops the stmt_mcontext and
	 * frees the exception frame.  ctx->return_bb was restored to the
	 * function's real return block after the try body above, so a RETURN here
	 * would branch straight past the cleanup.  Point it at a landing pad that
	 * runs HANDLER_DONE first.
	 */
	saved_return_bb = ctx->return_bb;
	ctx->return_bb = handler_return_bb;

	/*
	 * Likewise for the handler bodies: an EXIT/CONTINUE that leaves the block
	 * from inside a handler must run HANDLER_DONE for this frame on its way
	 * out (the subtransaction was already rolled back in CATCH; what remains
	 * is the stmt_mcontext pop, the cur_error restore, and the frame free).
	 */
	{
		llvm::Value *cleanup_args[] = { frame_ptr };

		upl_push_cleanup(ctx, RT_EXCEPTION_HANDLER_DONE, cleanup_args);
	}

	/* === HANDLER BLOCKS === */
	i = 0;
	for (auto *exception : cppgres::list<UPLpgSQL_exception *>(exc_list))
	{
		llvm::BasicBlock	*handler_bb;
		char				bbname[32];

		snprintf(bbname, sizeof(bbname), "exc.handler_%d", i);
		handler_bb = upl_append_block(ctx, bbname);

		/* Add case to switch */
		switch_inst->addCase(llvm::cast<llvm::ConstantInt>(upl_const_int32(ctx, i)),
							 handler_bb);

		/* Compile handler body */
		ctx->builder->SetInsertPoint(handler_bb);

		/* Set SQLSTATE/SQLERRM variables */
		{
			llvm::Value *args[] = {
				estate_ref,
				upl_const_ptr(ctx, stmt),
				upl_const_int32(ctx, i)
			};
			call_fn(RT_EXCEPTION_SET_HANDLER_VARS, args);
		}

		/*
		 * "EXIT <label>" out of a handler body leaves the block, but must
		 * run HANDLER_DONE on the way out.
		 */
		if (stmt->label != NULL)
			upl_push_block_label(ctx, stmt->label, handler_exit_bb);

		/* Compile the handler's statements */
		compile_stmts(exception->action);

		if (stmt->label != NULL)
			upl_pop_loop(ctx);

		/* Clean up after handler (normal, non-RETURN completion) */
		{
			llvm::Value *args[] = { estate_ref, frame_ptr };
			call_fn(RT_EXCEPTION_HANDLER_DONE, args);
		}

		ctx->builder->CreateBr(after_block_bb);

		i++;
	}

	upl_pop_cleanup(ctx);

	ctx->return_bb = saved_return_bb;

	/* === HANDLER RETURN (RETURN inside a handler body) === */
	ctx->builder->SetInsertPoint(handler_return_bb);
	{
		llvm::Value *args[] = { estate_ref, frame_ptr };
		call_fn(RT_EXCEPTION_HANDLER_DONE, args);
	}
	ctx->builder->CreateBr(saved_return_bb);

	/* === HANDLER EXIT (EXIT <label> inside a handler body) === */
	ctx->builder->SetInsertPoint(handler_exit_bb);
	{
		llvm::Value *args[] = { estate_ref, frame_ptr };
		call_fn(RT_EXCEPTION_HANDLER_DONE, args);
	}
	ctx->builder->CreateBr(after_block_bb);

	/* === RETHROW === */
	ctx->builder->SetInsertPoint(rethrow_bb);
	{
		llvm::Value *args[] = { estate_ref, frame_ptr };
		call_fn(RT_EXCEPTION_RETHROW, args);
	}
	ctx->builder->CreateUnreachable();

	/* Continue after the block */
	ctx->builder->SetInsertPoint(after_block_bb);
}

/* ----------------------------------------------------------------
 * Native Local Array Escape Analysis (Phase 7)
 *
 * Identifies local array variables that can be lowered to flat native
 * memory (stack alloca or heap palloc0) instead of going through
 * PostgreSQL's array_get_element / array_set_element on every subscript
 * access.  This eliminates ~50-100ns per array access, replacing it with
 * a single GEP+load/store instruction (~1ns).
 *
 * A variable qualifies if ALL of these hold:
 *   1. dtype == UPLPGSQL_DTYPE_VAR, datatype->typisarray == true
 *   2. Element type is int4, int8, or float8 (Tier 1 types)
 *   3. Not a function parameter (parameters may alias caller data)
 *   4. Never passed to RETURN, RAISE, SPI, PERFORM, EXECUTE, CALL,
 *      FOREACH, OPEN, or FETCH (these would need a real PG array Datum)
 *
 * Key design decision: the ASSIGN statement case does NOT disqualify
 * arrays referenced in other assignments' expressions.  This is because
 * PL/pgSQL's paramnos bitmapset (used by expr_references_dno) cannot
 * distinguish a subscript read like x[i] from a whole-datum reference
 * like x.  Subscript reads are safe — they are compiled to native
 * GEP+load by compile_expr_datum().  True whole-datum escapes
 * (y := x, RETURN x, etc.) are caught by the other statement-type
 * cases in native_array_check_stmt().
 *
 * After analysis, qualifying arrays get:
 *   - data_ptr alloca (ptr, initially NULL) in the LLVM entry block
 *   - len_ptr alloca (i32, initially 0) in the LLVM entry block
 *   - array_fill() interception → stack/heap allocation (upl_compile_expr.cpp)
 *   - arr[i] read/write → inline bounds check + GEP (upl_compile_expr.cpp)
 * ----------------------------------------------------------------
 */

/* Max total stack allocation across all native arrays in one function */
#define NATIVE_ARRAY_TOTAL_STACK_MAX	16384

/*
 * Check if an expression contains a Param reference to a given dno
 * anywhere (used for RETURN, RAISE args, etc. where any reference escapes).
 */
static bool
expr_references_dno(UPLpgSQL_expr *expr, int dno)
{
	Bitmapset *paramnos;

	if (expr == NULL)
		return false;

	paramnos = expr->paramnos;
	return bms_is_member(dno, paramnos);
}

/*
 * Walk statement list and disqualify any candidate arrays that escape.
 */
static void
native_array_check_stmts(UPLpgSQL_function *func, List *stmts,
						 std::vector<bool> &candidates);

static void
native_array_check_stmt(UPLpgSQL_function *func, UPLpgSQL_stmt *stmt,
						std::vector<bool> &candidates)
{
	int		ndatums = (int) candidates.size();
	int		dno;

	switch (stmt->cmd_type)
	{
		case UPLPGSQL_STMT_ASSIGN:
			{
				/*
				 * ASSIGN never disqualifies a candidate; escapes are
				 * handled at IR generation instead of here.
				 *
				 * We cannot decide it here: expr_references_dno uses the
				 * paramnos bitmapset, which can't distinguish a subscript
				 * read (x[i], safe and compiled to native GEP+load) from a
				 * whole-datum read (x, an escape).  Disqualifying on
				 * paramnos would drop every array that is ever read by
				 * subscript — that is, all of them.
				 *
				 * So compile_assign() handles both escape
				 * directions when it falls back to the interpreter: it
				 * syncs flat memory into the PG Datums beforehand (so a
				 * whole-datum read like y := x sees live data), and
				 * reloads flat memory afterwards if the target is itself
				 * a native array (so x := ARRAY[...] is not left stale).
				 */
			}
			break;

		case UPLPGSQL_STMT_RETURN:
		case UPLPGSQL_STMT_RETURN_NEXT:
			{
				UPLpgSQL_stmt_return *r = (UPLpgSQL_stmt_return *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(r->expr, dno))
						candidates[dno] = false;
				}
			}
			break;

		case UPLPGSQL_STMT_RETURN_QUERY:
			{
				UPLpgSQL_stmt_return_query *r =
					(UPLpgSQL_stmt_return_query *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						(expr_references_dno(r->query, dno) ||
						 expr_references_dno(r->dynquery, dno)))
						candidates[dno] = false;
				}
			}
			break;

		case UPLPGSQL_STMT_RAISE:
			{
				UPLpgSQL_stmt_raise *r = (UPLpgSQL_stmt_raise *) stmt;

				for (auto *param : cppgres::list<UPLpgSQL_expr *>(r->params))
				{
					for (dno = 0; dno < ndatums; dno++)
					{
						if (candidates[dno] &&
							expr_references_dno(param, dno))
							candidates[dno] = false;
					}
				}
				for (auto *opt : cppgres::list<UPLpgSQL_raise_option *>(r->options))
				{
					for (dno = 0; dno < ndatums; dno++)
					{
						if (candidates[dno] &&
							expr_references_dno(opt->expr, dno))
							candidates[dno] = false;
					}
				}
			}
			break;

		case UPLPGSQL_STMT_EXECSQL:
			{
				UPLpgSQL_stmt_execsql *e = (UPLpgSQL_stmt_execsql *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(e->sqlstmt, dno))
						candidates[dno] = false;
				}
			}
			break;

		case UPLPGSQL_STMT_DYNEXECUTE:
			{
				UPLpgSQL_stmt_dynexecute *e =
					(UPLpgSQL_stmt_dynexecute *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(e->query, dno))
						candidates[dno] = false;
				}
				for (auto *param : cppgres::list<UPLpgSQL_expr *>(e->params))
				{
					for (dno = 0; dno < ndatums; dno++)
					{
						if (candidates[dno] &&
							expr_references_dno(param, dno))
							candidates[dno] = false;
					}
				}
			}
			break;

		case UPLPGSQL_STMT_PERFORM:
			{
				UPLpgSQL_stmt_perform *p = (UPLpgSQL_stmt_perform *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(p->expr, dno))
						candidates[dno] = false;
				}
			}
			break;

		case UPLPGSQL_STMT_CALL:
			{
				UPLpgSQL_stmt_call *c = (UPLpgSQL_stmt_call *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(c->expr, dno))
						candidates[dno] = false;
				}
			}
			break;

		case UPLPGSQL_STMT_FOREACH_A:
			{
				UPLpgSQL_stmt_foreach_a *f =
					(UPLpgSQL_stmt_foreach_a *) stmt;

				/* If iterating over a candidate array, disqualify */
				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(f->expr, dno))
						candidates[dno] = false;
				}
				native_array_check_stmts(func, f->body, candidates);
			}
			break;

		/* Statements with sub-statement lists — recurse */
		case UPLPGSQL_STMT_BLOCK:
			{
				UPLpgSQL_stmt_block *b = (UPLpgSQL_stmt_block *) stmt;

				native_array_check_stmts(func, b->body, candidates);
				if (b->exceptions)
				{
					for (auto *exc : cppgres::list<UPLpgSQL_exception *>(b->exceptions->exc_list))
					{
						native_array_check_stmts(func, exc->action,
												 candidates);
					}
				}
			}
			break;

		case UPLPGSQL_STMT_IF:
			{
				UPLpgSQL_stmt_if *i = (UPLpgSQL_stmt_if *) stmt;

				/* condition can reference arrays */
				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(i->cond, dno))
						candidates[dno] = false;
				}
				native_array_check_stmts(func, i->then_body,
										 candidates);
				for (auto *elif : cppgres::list<UPLpgSQL_if_elsif *>(i->elsif_list))
				{
					for (dno = 0; dno < ndatums; dno++)
					{
						if (candidates[dno] &&
							expr_references_dno(elif->cond, dno))
							candidates[dno] = false;
					}
					native_array_check_stmts(func, elif->stmts,
											 candidates);
				}
				native_array_check_stmts(func, i->else_body,
										 candidates);
			}
			break;

		case UPLPGSQL_STMT_CASE:
			{
				UPLpgSQL_stmt_case *c = (UPLpgSQL_stmt_case *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(c->t_expr, dno))
						candidates[dno] = false;
				}
				for (auto *w : cppgres::list<UPLpgSQL_case_when *>(c->case_when_list))
				{
					for (dno = 0; dno < ndatums; dno++)
					{
						if (candidates[dno] &&
							expr_references_dno(w->expr, dno))
							candidates[dno] = false;
					}
					native_array_check_stmts(func, w->stmts,
											 candidates);
				}
				native_array_check_stmts(func, c->else_stmts,
										 candidates);
			}
			break;

		case UPLPGSQL_STMT_LOOP:
			{
				UPLpgSQL_stmt_loop *l = (UPLpgSQL_stmt_loop *) stmt;

				native_array_check_stmts(func, l->body,
										 candidates);
			}
			break;

		case UPLPGSQL_STMT_WHILE:
			{
				UPLpgSQL_stmt_while *w = (UPLpgSQL_stmt_while *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(w->cond, dno))
						candidates[dno] = false;
				}
				native_array_check_stmts(func, w->body,
										 candidates);
			}
			break;

		case UPLPGSQL_STMT_FORI:
			{
				UPLpgSQL_stmt_fori *f = (UPLpgSQL_stmt_fori *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						(expr_references_dno(f->lower, dno) ||
						 expr_references_dno(f->upper, dno) ||
						 expr_references_dno(f->step, dno)))
						candidates[dno] = false;
				}
				native_array_check_stmts(func, f->body,
										 candidates);
			}
			break;

		case UPLPGSQL_STMT_FORS:
		case UPLPGSQL_STMT_DYNFORS:
			{
				/* These have a query and a body */
				UPLpgSQL_stmt_fors *f = (UPLpgSQL_stmt_fors *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(f->query, dno))
						candidates[dno] = false;
				}
				native_array_check_stmts(func, f->body,
										 candidates);
			}
			break;

		case UPLPGSQL_STMT_FORC:
			{
				UPLpgSQL_stmt_forc *f = (UPLpgSQL_stmt_forc *) stmt;

				for (dno = 0; dno < ndatums; dno++)
				{
					if (candidates[dno] &&
						expr_references_dno(f->argquery, dno))
						candidates[dno] = false;
				}
				native_array_check_stmts(func, f->body,
										 candidates);
			}
			break;

		default:
			/* EXIT, CLOSE, FETCH, GETDIAG, COMMIT, ROLLBACK, ASSERT, OPEN
			 * — no expression that could reference an array */
			break;
	}
}

static void
native_array_check_stmts(UPLpgSQL_function *func, List *stmts,
						 std::vector<bool> &candidates)
{
	for (auto *stmt : cppgres::list<UPLpgSQL_stmt *>(stmts))
		native_array_check_stmt(func, stmt, candidates);
}

/*
 * Main escape analysis entry point for native local arrays.
 *
 * Three-step process:
 *   Step 1: Identify candidate arrays (local vars with int4/int8/float8
 *           element types, excluding function parameters).
 *   Step 2: Walk the entire AST via native_array_check_stmts() and
 *           disqualify any candidate that appears in an escaping context
 *           (RETURN, RAISE, SPI, etc.).
 *   Step 3: Build the UPLpgSQL_native_array metadata array from survivors.
 *
 * Populates native_arrays_.
 * Called from setup_entry(), before IR generation.
 */
void
function_compiler::analyze_native_arrays(UPLpgSQL_function *func)
{
	int			ndatums = func->ndatums;
	int			i, count;

	native_arrays_.clear();

	if (ndatums == 0)
		return;

	std::vector<bool> candidates(ndatums, false);

	/* Step 1: Identify candidate array variables */
	for (i = 0; i < ndatums; i++)
	{
		UPLpgSQL_datum *d = func->datums[i];
		UPLpgSQL_var   *var;
		Oid				elemtype;
		bool			is_param = false;
		int				j;

		if (d->dtype != UPLPGSQL_DTYPE_VAR)
			continue;

		var = (UPLpgSQL_var *) d;

		if (!var->datatype->typisarray)
			continue;

		/* Check element type */
		elemtype = get_element_type(var->datatype->typoid);
		if (elemtype != INT4OID && elemtype != INT8OID && elemtype != FLOAT8OID)
			continue;

		/* Exclude function parameters */
		for (j = 0; j < func->fn_nargs; j++)
		{
			if (func->fn_argvarnos[j] == i)
			{
				is_param = true;
				break;
			}
		}
		if (is_param)
			continue;

		candidates[i] = true;
	}

	/* Step 2: Walk AST and disqualify arrays that escape */
	native_array_check_stmts(func, func->action->body, candidates);

	/* Step 3: Build native_arrays list from survivors */
	count = (int) std::ranges::count(candidates, true);

	if (count > 0)
	{
		native_arrays_.reserve(count);

		for (i = 0; i < ndatums; i++)
		{
			UPLpgSQL_var		   *var;
			Oid						elemtype;
			UPLpgSQL_native_array	na = {};

			if (!candidates[i])
				continue;

			var = (UPLpgSQL_var *) func->datums[i];
			elemtype = get_element_type(var->datatype->typoid);

			na.dno = i;
			na.elemtype = elemtype;

			if (elemtype == INT4OID)
				na.elem_size = 4;
			else
				na.elem_size = 8;	/* int8, float8 */

			/* llvm_elemtype and data_ptr/len_ptr set later during IR gen */

			native_arrays_.push_back(na);

			elog(DEBUG1, "uplpgsql: native array candidate dno %d (%s), "
				 "elemtype %u, elem_size %d",
				 i, var->refname, elemtype, na.elem_size);
		}
	}
}

/*
 * uplpgsql_jit_score_stmts - Recursively score a statement list for JIT
 *                            suitability.
 *
 * Returns a score where positive values favor JIT compilation and negative
 * values favor the interpreter.  The loop_depth multiplier amplifies scores
 * inside loops since those statements execute many times.
 */
static int
uplpgsql_jit_score_stmts(List *stmts, int loop_depth);

static int
uplpgsql_jit_score_stmt(UPLpgSQL_stmt *stmt, int loop_depth)
{
	int		score = 0;

	/*
	 * Only statements inside loops score positively.  Top-level statements
	 * (loop_depth == 0) don't benefit from JIT because the per-call
	 * overhead dominates — the function body executes once per call
	 * regardless of whether it's interpreted or JIT'd.
	 */
	int		mult = (loop_depth > 0) ? 3 * loop_depth : 0;

	switch (stmt->cmd_type)
	{
		case UPLPGSQL_STMT_ASSIGN:
			/* Scalar assignment may be inlined to native IR */
			score = 1 * mult;
			break;

		case UPLPGSQL_STMT_IF:
			{
				UPLpgSQL_stmt_if *ifstmt = (UPLpgSQL_stmt_if *) stmt;

				score = 1 * mult;	/* the branch itself */
				score += uplpgsql_jit_score_stmts(ifstmt->then_body,
												  loop_depth);
				for (auto *ei : cppgres::list<UPLpgSQL_if_elsif *>(ifstmt->elsif_list))
				{
					score += 1 * mult;
					score += uplpgsql_jit_score_stmts(ei->stmts, loop_depth);
				}
				if (ifstmt->else_body)
					score += uplpgsql_jit_score_stmts(ifstmt->else_body,
													  loop_depth);
			}
			break;

		case UPLPGSQL_STMT_CASE:
			{
				UPLpgSQL_stmt_case *cs = (UPLpgSQL_stmt_case *) stmt;

				score = 1 * mult;
				for (auto *cw : cppgres::list<UPLpgSQL_case_when *>(cs->case_when_list))
				{
					score += 1 * mult;
					score += uplpgsql_jit_score_stmts(cw->stmts, loop_depth);
				}
				if (cs->else_stmts)
					score += uplpgsql_jit_score_stmts(cs->else_stmts,
													  loop_depth);
			}
			break;

		case UPLPGSQL_STMT_WHILE:
			{
				UPLpgSQL_stmt_while *ws = (UPLpgSQL_stmt_while *) stmt;

				score = 3;	/* loops always score positive */
				score += uplpgsql_jit_score_stmts(ws->body, loop_depth + 1);
			}
			break;

		case UPLPGSQL_STMT_LOOP:
			{
				UPLpgSQL_stmt_loop *ls = (UPLpgSQL_stmt_loop *) stmt;

				score = 3;
				score += uplpgsql_jit_score_stmts(ls->body, loop_depth + 1);
			}
			break;

		case UPLPGSQL_STMT_FORI:
			{
				UPLpgSQL_stmt_fori *fi = (UPLpgSQL_stmt_fori *) stmt;

				score = 3;
				score += uplpgsql_jit_score_stmts(fi->body, loop_depth + 1);
			}
			break;

		case UPLPGSQL_STMT_EXIT:
			score = 1 * mult;
			break;

		/* SPI-dominated: cursor-based loops */
		case UPLPGSQL_STMT_FORS:
			{
				UPLpgSQL_stmt_fors *fs = (UPLpgSQL_stmt_fors *) stmt;

				score = -3;	/* SPI cursor always penalizes */
				score += uplpgsql_jit_score_stmts(fs->body, loop_depth + 1);
			}
			break;

		case UPLPGSQL_STMT_FORC:
			{
				UPLpgSQL_stmt_forc *fc = (UPLpgSQL_stmt_forc *) stmt;

				score = -3;
				score += uplpgsql_jit_score_stmts(fc->body, loop_depth + 1);
			}
			break;

		case UPLPGSQL_STMT_DYNFORS:
			{
				UPLpgSQL_stmt_dynfors *df = (UPLpgSQL_stmt_dynfors *) stmt;

				score = -3;
				score += uplpgsql_jit_score_stmts(df->body, loop_depth + 1);
			}
			break;

		/* SPI-dominated: SQL execution — penalize more inside loops */
		case UPLPGSQL_STMT_PERFORM:
		case UPLPGSQL_STMT_EXECSQL:
		case UPLPGSQL_STMT_DYNEXECUTE:
		case UPLPGSQL_STMT_CALL:
			score = (loop_depth > 0) ? -2 * 3 * loop_depth : -2;
			break;

		/* Moderate SPI cost */
		case UPLPGSQL_STMT_FOREACH_A:
		case UPLPGSQL_STMT_RETURN_NEXT:
		case UPLPGSQL_STMT_RETURN_QUERY:
			score = (loop_depth > 0) ? -1 * 3 * loop_depth : -1;
			break;

		/* Neutral */
		case UPLPGSQL_STMT_RETURN:
		case UPLPGSQL_STMT_RAISE:
		case UPLPGSQL_STMT_ASSERT:
		case UPLPGSQL_STMT_GETDIAG:
		case UPLPGSQL_STMT_OPEN:
		case UPLPGSQL_STMT_FETCH:
		case UPLPGSQL_STMT_CLOSE:
		case UPLPGSQL_STMT_COMMIT:
		case UPLPGSQL_STMT_ROLLBACK:
			score = 0;
			break;

		case UPLPGSQL_STMT_BLOCK:
			{
				UPLpgSQL_stmt_block *blk = (UPLpgSQL_stmt_block *) stmt;

				if (blk->exceptions)
					score = -2;

				score += uplpgsql_jit_score_stmts(blk->body, loop_depth);
			}
			break;

		default:
			score = 0;
			break;
	}

	return score;
}

static int
uplpgsql_jit_score_stmts(List *stmts, int loop_depth)
{
	int			total = 0;

	for (auto *stmt : cppgres::list<UPLpgSQL_stmt *>(stmts))
		total += uplpgsql_jit_score_stmt(stmt, loop_depth);

	return total;
}

/*
 * uplpgsql_should_jit - Decide whether to JIT-compile a function.
 *
 * Walks the AST and scores the function based on statement mix.
 * Returns true if JIT is expected to help.
 */
extern "C" bool
uplpgsql_should_jit(UPLpgSQL_function *func)
{
	int		score;

	score = uplpgsql_jit_score_stmts(func->action->body, 0);

	elog(DEBUG1, "uplpgsql: JIT score for %s = %d",
		 func->fn_signature, score);

	return (score > 0);
}

/*
 * function_compiler - Compile a UPLpgSQL_function to native code via LLVM.
 *
 * The constructor prepares the core compile context (callbacks, datum
 * offsets, runtime function table); compile() fills in the pipeline hooks
 * and runs the core pipeline (upl_compile_function), which orchestrates
 * LLVM setup, IR generation, verification, optimization (O3), and OrcJIT
 * compilation.
 *
 * Input:  UPLpgSQL_function (AST from the forked PL/pgSQL parser)
 * Output: UPLpgSQL_func (contains the native function pointer)
 *
 * The generated LLVM function has signature: i32 func(ptr estate)
 * where estate is a UPLpgSQL_exec_state*.  The return value is one of
 * UPLPGSQL_RC_OK, RC_EXIT, RC_RETURN, RC_CONTINUE.
 *
 * Unique symbol names (uplpgsql_fn_<oid>_g<N>) prevent collisions when
 * a function is recompiled after CREATE OR REPLACE — the old symbol
 * remains in OrcJIT but is no longer referenced.
 */
function_compiler::function_compiler(UPLpgSQL_function *func) : func_(func)
{
	/* Let the core-callback trampolines recover this compiler from ctx */
	ctx_.lang_data = this;

	/* Allocate RT function arrays */
	ctx_.rt_funcs.assign(UPLPGSQL_NUM_RT_FUNCS, nullptr);
	ctx_.rt_fntypes.assign(UPLPGSQL_NUM_RT_FUNCS, nullptr);

	/* Setup callbacks */
	ctx_.callbacks.compile_stmts = cb_compile_stmts;
	ctx_.callbacks.try_compile_bool = cb_try_compile_bool;
	ctx_.callbacks.assign_expr = cb_assign_expr;
	ctx_.callbacks.rt_eval_bool = RT_EVAL_BOOL;
	ctx_.callbacks.rt_eval_int = RT_EVAL_INT;
	ctx_.callbacks.rt_set_found = RT_SET_FOUND;
	ctx_.callbacks.rt_assign_int = RT_ASSIGN_INT;
	ctx_.callbacks.rt_case_error = RT_CASE_ERROR;
	ctx_.callbacks.rt_assign_null = RT_ASSIGN_NULL;
	ctx_.callbacks.rt_init_var = RT_INIT_VAR;

	/* Setup datum offsets */
	ctx_.datum_offsets.estate_to_lang_state =
		offsetof(UPLpgSQL_exec_state, uplpgsql_estate);
	ctx_.datum_offsets.lang_state_to_datums =
		offsetof(UPLpgSQL_execstate, datums);
	ctx_.datum_offsets.var_to_value = offsetof(UPLpgSQL_var, value);
	ctx_.datum_offsets.var_to_isnull = offsetof(UPLpgSQL_var, isnull);
}

UPLpgSQL_func *
function_compiler::compile()
{
	UPL_compile_hooks	hooks = {
		.register_rt_funcs = cb_register_rt_funcs,
		.setup_entry = cb_setup_entry,
		.compile_body = cb_compile_body,
		.func_name_prefix = "uplpgsql_fn",
		.fn_oid = func_->fn_oid,
		.fn_xmin = func_->cfunc.fn_xmin,
		.fn_tid = func_->cfunc.fn_tid,
		.default_rc = UPLPGSQL_RC_OK,
		.dump_ir = uplpgsql_dump_ir,
	};
	UPLpgSQL_func	   *result;
	void			   *fn_ptr;

	/* Run the core compilation pipeline */
	fn_ptr = upl_compile_function(&ctx_, hooks);

	/* Create cached function result */
	result = (UPLpgSQL_func *) MemoryContextAllocZero(TopMemoryContext,
													   sizeof(UPLpgSQL_func));
	result->jit_func = (uplpgsql_jit_func) fn_ptr;
	result->fn_oid = func_->fn_oid;
	result->fn_xmin = func_->cfunc.fn_xmin;
	result->fn_tid = func_->cfunc.fn_tid;

	return result;
}

/*
 * uplpgsql_compile_function - Public entry point (C linkage; called from
 * the language handler).
 */
extern "C" UPLpgSQL_func *
uplpgsql_compile_function(UPLpgSQL_function *func)
{
	return function_compiler(func).compile();
}

/*
 * Register all runtime helper function declarations in the LLVM module.
 *
 * This creates LLVM function declarations (not definitions) for every
 * uplpgsql_rt_* runtime helper.  The declarations provide the type
 * information LLVM needs to generate correct call instructions.  The
 * actual function bodies are not in the LLVM module — they are resolved
 * at link time by OrcJIT's process symbol search generator, which finds
 * them in the process's symbol table (they're in our .so with default
 * visibility via UPL_RT_EXPORT).
 *
 * Each registration stores both the llvm::Function * (declaration) and
 * llvm::FunctionType * in ctx->rt_funcs[] and ctx->rt_fntypes[],
 * indexed by the UPLpgSQL_rt_func enum.  These are used by
 * call_fn() to emit call instructions.
 */
void
function_compiler::register_runtime_funcs()
{
	llvm::Type *ptr = ctx->types[UPL_PTR];
	llvm::Type *i1  = ctx->types[UPL_INT1];
	llvm::Type *i8  = ctx->types[UPL_INT8];
	llvm::Type *i32 = ctx->types[UPL_INT32];
	llvm::Type *i64 = ctx->types[UPL_INT64];
	llvm::Type *vd  = ctx->types[UPL_VOID];

	/* Datum uplpgsql_rt_eval_expr(ptr estate, ptr expr, ptr isNull_out) */
	{
		llvm::Type *params[] = { ptr, ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i64, params, false);

		ctx->rt_fntypes[RT_EVAL_EXPR] = ft;
		ctx->rt_funcs[RT_EVAL_EXPR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_eval_expr", ctx->module.get());
	}

	/* bool uplpgsql_rt_eval_bool(ptr estate, ptr expr) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i1, params, false);

		ctx->rt_fntypes[RT_EVAL_BOOL] = ft;
		ctx->rt_funcs[RT_EVAL_BOOL] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_eval_bool", ctx->module.get());
	}

	/* int32 uplpgsql_rt_eval_int(ptr estate, ptr expr) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EVAL_INT] = ft;
		ctx->rt_funcs[RT_EVAL_INT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_eval_int", ctx->module.get());
	}

	/* void uplpgsql_rt_init_var(ptr estate, i32 dno) */
	{
		llvm::Type *params[] = { ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_INIT_VAR] = ft;
		ctx->rt_funcs[RT_INIT_VAR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_init_var", ctx->module.get());
	}

	/* void uplpgsql_rt_assign_expr(ptr estate, i32 target_dno, ptr expr) */
	{
		llvm::Type *params[] = { ptr, i32, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_ASSIGN_EXPR] = ft;
		ctx->rt_funcs[RT_ASSIGN_EXPR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_assign_expr", ctx->module.get());
	}

	/* void uplpgsql_rt_set_found(ptr estate, i1 value) */
	{
		llvm::Type *params[] = { ptr, i1 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_SET_FOUND] = ft;
		ctx->rt_funcs[RT_SET_FOUND] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_set_found", ctx->module.get());
	}

	/* void uplpgsql_rt_assign_int(ptr estate, i32 dno, i32 value) */
	{
		llvm::Type *params[] = { ptr, i32, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_ASSIGN_INT] = ft;
		ctx->rt_funcs[RT_ASSIGN_INT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_assign_int", ctx->module.get());
	}

	/* int32 uplpgsql_rt_exec_return(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_RETURN] = ft;
		ctx->rt_funcs[RT_EXEC_RETURN] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_return", ctx->module.get());
	}

	/* int32 uplpgsql_rt_exec_perform(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_PERFORM] = ft;
		ctx->rt_funcs[RT_EXEC_PERFORM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_perform", ctx->module.get());
	}

	/* int32 uplpgsql_rt_exec_sql(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_SQL] = ft;
		ctx->rt_funcs[RT_EXEC_SQL] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_sql", ctx->module.get());
	}

	/* void uplpgsql_rt_exec_raise(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXEC_RAISE] = ft;
		ctx->rt_funcs[RT_EXEC_RAISE] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_raise", ctx->module.get());
	}

	/* void uplpgsql_rt_assign_null(ptr estate, i32 dno) */
	{
		llvm::Type *params[] = { ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_ASSIGN_NULL] = ft;
		ctx->rt_funcs[RT_ASSIGN_NULL] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_assign_null", ctx->module.get());
	}

	/* void uplpgsql_rt_case_error(ptr estate, i32 lineno) */
	{
		llvm::Type *params[] = { ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_CASE_ERROR] = ft;
		ctx->rt_funcs[RT_CASE_ERROR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_case_error", ctx->module.get());
	}

	/* void uplpgsql_rt_exec_assert(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXEC_ASSERT_FAIL] = ft;
		ctx->rt_funcs[RT_EXEC_ASSERT_FAIL] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_assert", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_open(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_OPEN] = ft;
		ctx->rt_funcs[RT_EXEC_OPEN] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_open", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_fetch(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_FETCH] = ft;
		ctx->rt_funcs[RT_EXEC_FETCH] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_fetch", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_close(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_CLOSE] = ft;
		ctx->rt_funcs[RT_EXEC_CLOSE] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_close", ctx->module.get());
	}

	/* ptr uplpgsql_rt_open_query_cursor(ptr estate, ptr query) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_OPEN_QUERY_CURSOR] = ft;
		ctx->rt_funcs[RT_OPEN_QUERY_CURSOR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_open_query_cursor", ctx->module.get());
	}

	/* i1 uplpgsql_rt_fetch_cursor_row(ptr estate, ptr portal, i32 target_dno) */
	{
		llvm::Type *params[] = { ptr, ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(i1, params, false);

		ctx->rt_fntypes[RT_FETCH_CURSOR_ROW] = ft;
		ctx->rt_funcs[RT_FETCH_CURSOR_ROW] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_fetch_cursor_row", ctx->module.get());
	}

	/* void uplpgsql_rt_close_portal(ptr estate, ptr portal) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_CLOSE_PORTAL] = ft;
		ctx->rt_funcs[RT_CLOSE_PORTAL] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_close_portal", ctx->module.get());
	}

	/* ptr uplpgsql_rt_open_forc_cursor(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_OPEN_FORC_CURSOR] = ft;
		ctx->rt_funcs[RT_OPEN_FORC_CURSOR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_open_forc_cursor", ctx->module.get());
	}

	/* void uplpgsql_rt_close_forc_cursor(ptr estate, ptr stmt, ptr portal) */
	{
		llvm::Type *params[] = { ptr, ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_CLOSE_FORC_CURSOR] = ft;
		ctx->rt_funcs[RT_CLOSE_FORC_CURSOR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_close_forc_cursor", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_block_protected(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_BLOCK_PROTECTED] = ft;
		ctx->rt_funcs[RT_EXEC_BLOCK_PROTECTED] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_block_protected", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_dynexecute(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_DYNEXECUTE] = ft;
		ctx->rt_funcs[RT_EXEC_DYNEXECUTE] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_dynexecute", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_call(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_CALL] = ft;
		ctx->rt_funcs[RT_EXEC_CALL] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_call", ctx->module.get());
	}

	/* void uplpgsql_rt_exec_getdiag(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXEC_GETDIAG] = ft;
		ctx->rt_funcs[RT_EXEC_GETDIAG] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_getdiag", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_return_next(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_RETURN_NEXT] = ft;
		ctx->rt_funcs[RT_EXEC_RETURN_NEXT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_return_next", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_return_query(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_RETURN_QUERY] = ft;
		ctx->rt_funcs[RT_EXEC_RETURN_QUERY] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_return_query", ctx->module.get());
	}

	/* void uplpgsql_rt_exec_commit(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXEC_COMMIT] = ft;
		ctx->rt_funcs[RT_EXEC_COMMIT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_commit", ctx->module.get());
	}

	/* void uplpgsql_rt_exec_rollback(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXEC_ROLLBACK] = ft;
		ctx->rt_funcs[RT_EXEC_ROLLBACK] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_rollback", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exec_foreach_a(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXEC_FOREACH_A] = ft;
		ctx->rt_funcs[RT_EXEC_FOREACH_A] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exec_foreach_a", ctx->module.get());
	}

	/* ptr uplpgsql_rt_open_dynfors_cursor(ptr estate, ptr stmt) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_OPEN_DYNFORS_CURSOR] = ft;
		ctx->rt_funcs[RT_OPEN_DYNFORS_CURSOR] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_open_dynfors_cursor", ctx->module.get());
	}

	/* --- Exception handling runtime functions --- */

	/* ptr uplpgsql_rt_exception_push_frame(ptr estate, ptr block) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_PUSH_FRAME] = ft;
		ctx->rt_funcs[RT_EXCEPTION_PUSH_FRAME] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_push_frame", ctx->module.get());
	}

	/* void uplpgsql_rt_exception_arm(ptr estate, ptr frame) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_ARM] = ft;
		ctx->rt_funcs[RT_EXCEPTION_ARM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_arm", ctx->module.get());
	}

	/* void uplpgsql_rt_exception_try_exit(ptr estate, ptr frame) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_TRY_EXIT] = ft;
		ctx->rt_funcs[RT_EXCEPTION_TRY_EXIT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_try_exit", ctx->module.get());
	}

	/* i32 uplpgsql_rt_exception_catch(ptr estate, ptr block, ptr frame) */
	{
		llvm::Type *params[] = { ptr, ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i32, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_CATCH] = ft;
		ctx->rt_funcs[RT_EXCEPTION_CATCH] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_catch", ctx->module.get());
	}

	/* void uplpgsql_rt_exception_set_handler_vars(ptr estate, ptr block, i32 idx) */
	{
		llvm::Type *params[] = { ptr, ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_SET_HANDLER_VARS] = ft;
		ctx->rt_funcs[RT_EXCEPTION_SET_HANDLER_VARS] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_set_handler_vars", ctx->module.get());
	}

	/* void uplpgsql_rt_exception_handler_done(ptr estate, ptr frame) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_HANDLER_DONE] = ft;
		ctx->rt_funcs[RT_EXCEPTION_HANDLER_DONE] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_handler_done", ctx->module.get());
	}

	/* void uplpgsql_rt_exception_rethrow(ptr estate, ptr frame) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_EXCEPTION_RETHROW] = ft;
		ctx->rt_funcs[RT_EXCEPTION_RETHROW] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_exception_rethrow", ctx->module.get());
	}

	/* void uplpgsql_rt_assign_var_datum(ptr estate, i32 dno, i64 value, i8 isnull) */
	{
		llvm::Type *i8t = ctx->types[UPL_INT8];
		llvm::Type *params[] = { ptr, i32, i64, i8t };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_ASSIGN_VAR_DATUM] = ft;
		ctx->rt_funcs[RT_ASSIGN_VAR_DATUM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_assign_var_datum", ctx->module.get());
	}

	/* void uplpgsql_rt_copy_assign_var_datum(ptr estate, i32 dno, i64 value, i8 isnull) */
	{
		llvm::Type *i8t = ctx->types[UPL_INT8];
		llvm::Type *params[] = { ptr, i32, i64, i8t };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_COPY_ASSIGN_VAR_DATUM] = ft;
		ctx->rt_funcs[RT_COPY_ASSIGN_VAR_DATUM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_copy_assign_var_datum", ctx->module.get());
	}

	/* ptr uplpgsql_rt_alloc_scope_enter(ptr estate) */
	{
		llvm::Type *params[] = { ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_ALLOC_SCOPE_ENTER] = ft;
		ctx->rt_funcs[RT_ALLOC_SCOPE_ENTER] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_alloc_scope_enter", ctx->module.get());
	}

	/* void uplpgsql_rt_alloc_scope_exit(ptr estate, ptr old) */
	{
		llvm::Type *params[] = { ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_ALLOC_SCOPE_EXIT] = ft;
		ctx->rt_funcs[RT_ALLOC_SCOPE_EXIT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_alloc_scope_exit", ctx->module.get());
	}

	/* void uplpgsql_rt_copy_assign_var_datum_scoped(ptr, i32, i64, i8, ptr) */
	{
		llvm::Type *i8t = ctx->types[UPL_INT8];
		llvm::Type *params[] = { ptr, i32, i64, i8t, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_COPY_ASSIGN_VAR_DATUM_SCOPED] = ft;
		ctx->rt_funcs[RT_COPY_ASSIGN_VAR_DATUM_SCOPED] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_copy_assign_var_datum_scoped", ctx->module.get());
	}

	/* i64 uplpgsql_rt_get_recfield(ptr estate, i32 recfield_dno) */
	{
		llvm::Type *params[] = { ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(i64, params, false);

		ctx->rt_fntypes[RT_GET_RECFIELD] = ft;
		ctx->rt_funcs[RT_GET_RECFIELD] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_get_recfield", ctx->module.get());
	}

	/*
	 * Datum uplpgsql_rt_get_recfield_fast(ptr estate, i32 rec_dno,
	 *     i32 fnumber, ptr isnull_out)
	 */
	{
		llvm::Type *params[] = { ptr, i32, i32, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i64, params, false);

		ctx->rt_fntypes[RT_GET_RECFIELD_FAST] = ft;
		ctx->rt_funcs[RT_GET_RECFIELD_FAST] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_get_recfield_fast", ctx->module.get());
	}

	/*
	 * Datum uplpgsql_rt_array_get_element(ptr estate, i32 dno, i32 subscript,
	 *     i32 typlen, i32 elmlen, i1 elmbyval, i8 elmalign, ptr isNull)
	 */
	{
		llvm::Type *params[] = { ptr, i32, i32, i32, i32, i1, i8, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(i64, params, false);

		ctx->rt_fntypes[RT_ARRAY_GET_ELEMENT] = ft;
		ctx->rt_funcs[RT_ARRAY_GET_ELEMENT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_array_get_element", ctx->module.get());
	}

	/*
	 * void uplpgsql_rt_array_set_element(ptr estate, i32 dno, i32 subscript,
	 *     i64 value, i1 isnull, i32 typlen, i32 elemtype,
	 *     i32 elmlen, i1 elmbyval, i8 elmalign)
	 */
	{
		llvm::Type *params[] = { ptr, i32, i32, i64, i1, i32, i32, i32, i1, i8 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_ARRAY_SET_ELEMENT] = ft;
		ctx->rt_funcs[RT_ARRAY_SET_ELEMENT] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_array_set_element", ctx->module.get());
	}

	/*
	 * Phase 7: Native local array heap allocation.
	 * Called when byte_size > NATIVE_ARRAY_STACK_THRESHOLD.
	 * void *uplpgsql_rt_native_array_alloc(ptr estate, i64 byte_size)
	 */
	{
		llvm::Type *params[] = { ptr, i64 };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_NATIVE_ARRAY_ALLOC] = ft;
		ctx->rt_funcs[RT_NATIVE_ARRAY_ALLOC] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_native_array_alloc", ctx->module.get());
	}

	/*
	 * Phase 7: Native local array bounds check error path.
	 * Called from inline bounds checks when subscript is out of [1, length].
	 * void uplpgsql_rt_native_array_bounds_check(i32 subscript, i32 length)
	 */
	{
		llvm::Type *params[] = { i32, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_NATIVE_ARRAY_BOUNDS_CHECK] = ft;
		ctx->rt_funcs[RT_NATIVE_ARRAY_BOUNDS_CHECK] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_native_array_bounds_check", ctx->module.get());
	}

	/* ptr uplpgsql_rt_native_array_from_datum(ptr estate, i64 datum, i32 isnull, i32 elem_size, ptr out_nelems, ptr out_lb) */
	{
		llvm::Type *params[] = { ptr, i64, i32, i32, ptr, ptr, ptr };
		llvm::FunctionType *ft = llvm::FunctionType::get(ptr, params, false);

		ctx->rt_fntypes[RT_NATIVE_ARRAY_FROM_DATUM] = ft;
		ctx->rt_funcs[RT_NATIVE_ARRAY_FROM_DATUM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_native_array_from_datum", ctx->module.get());
	}

	/* void uplpgsql_rt_free_var_datum(ptr estate, i32 dno) */
	{
		llvm::Type *params[] = { ptr, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_FREE_VAR_DATUM] = ft;
		ctx->rt_funcs[RT_FREE_VAR_DATUM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_free_var_datum", ctx->module.get());
	}

	/* void uplpgsql_rt_native_array_to_datum(ptr estate, i32 varno, ptr data, i32 nelems, i32 lb, i32 elemtype, i32 elem_size) */
	{
		llvm::Type *params[] = { ptr, i32, ptr, i32, i32, ptr, i32, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_NATIVE_ARRAY_TO_DATUM] = ft;
		ctx->rt_funcs[RT_NATIVE_ARRAY_TO_DATUM] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_native_array_to_datum", ctx->module.get());
	}

	/*
	 * void uplpgsql_rt_native_array_reserve(ptr estate, ptr data_io,
	 *     ptr nulls_io, ptr is_heap_io, ptr cap_io, i32 len, i32 elem_size)
	 *
	 * The pointer arguments are the JIT'd code's own alloca slots: the
	 * helper grows the buffers and writes the new pointers/capacity back
	 * through them.
	 */
	{
		llvm::Type *params[] = { ptr, ptr, ptr, ptr, ptr, i32, i32 };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_NATIVE_ARRAY_RESERVE] = ft;
		ctx->rt_funcs[RT_NATIVE_ARRAY_RESERVE] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_native_array_reserve", ctx->module.get());
	}

	/* void uplpgsql_rt_native_array_release(ptr data, ptr nulls, i8 is_heap) */
	{
		llvm::Type *i8t = ctx->types[UPL_INT8];
		llvm::Type *params[] = { ptr, ptr, i8t };
		llvm::FunctionType *ft = llvm::FunctionType::get(vd, params, false);

		ctx->rt_fntypes[RT_NATIVE_ARRAY_RELEASE] = ft;
		ctx->rt_funcs[RT_NATIVE_ARRAY_RELEASE] = llvm::Function::Create(ft,
			llvm::Function::ExternalLinkage, "uplpgsql_rt_native_array_release", ctx->module.get());
	}
}

/*
 * Compile a list of statements
 */
void
function_compiler::compile_stmts(List *stmts)
{

	for (auto *stmt : cppgres::list<UPLpgSQL_stmt *>(stmts))
	{
		compile_stmt(stmt);
	}
}

/*
 * Compile a single statement (dispatch)
 */
void
function_compiler::compile_stmt(UPLpgSQL_stmt *stmt)
{
	/*
	 * Store the current statement pointer into plstate->err_stmt so that
	 * the error context callback can report the line number if a runtime
	 * helper raises an error.  This mirrors exec_stmts() in the interpreter.
	 */
	{
		llvm::Value *off, *gep, *stmt_ptr;

		off = llvm::ConstantInt::get(ctx->types[UPL_INT64],
						   offsetof(UPLpgSQL_execstate, err_stmt),
						   false);
		gep = ctx->builder->CreateGEP(ctx->types[UPL_INT8],
							 plstate_ref_, off, "err_stmt.ptr");
		stmt_ptr = llvm_const_ptr(ctx, (void *) stmt);
		ctx->builder->CreateStore(stmt_ptr, gep);
	}

	switch (stmt->cmd_type)
	{
		case UPLPGSQL_STMT_BLOCK:
			compile_block((UPLpgSQL_stmt_block *) stmt);
			break;
		case UPLPGSQL_STMT_RETURN:
			compile_return((UPLpgSQL_stmt_return *) stmt);
			break;
		case UPLPGSQL_STMT_ASSIGN:
			compile_assign((UPLpgSQL_stmt_assign *) stmt);
			break;
		case UPLPGSQL_STMT_IF:
			compile_if((UPLpgSQL_stmt_if *) stmt);
			break;
		case UPLPGSQL_STMT_WHILE:
			compile_while((UPLpgSQL_stmt_while *) stmt);
			break;
		case UPLPGSQL_STMT_LOOP:
			compile_loop((UPLpgSQL_stmt_loop *) stmt);
			break;
		case UPLPGSQL_STMT_FORI:
			compile_fori((UPLpgSQL_stmt_fori *) stmt);
			break;
		case UPLPGSQL_STMT_EXIT:
			compile_exit((UPLpgSQL_stmt_exit *) stmt);
			break;
		case UPLPGSQL_STMT_PERFORM:
			compile_perform((UPLpgSQL_stmt_perform *) stmt);
			break;
		case UPLPGSQL_STMT_EXECSQL:
			compile_execsql((UPLpgSQL_stmt_execsql *) stmt);
			break;
		case UPLPGSQL_STMT_RAISE:
			compile_raise((UPLpgSQL_stmt_raise *) stmt);
			break;
		case UPLPGSQL_STMT_CASE:
			compile_case((UPLpgSQL_stmt_case *) stmt);
			break;
		case UPLPGSQL_STMT_ASSERT:
			compile_assert((UPLpgSQL_stmt_assert *) stmt);
			break;
		case UPLPGSQL_STMT_OPEN:
			compile_open((UPLpgSQL_stmt_open *) stmt);
			break;
		case UPLPGSQL_STMT_FETCH:
			compile_fetch((UPLpgSQL_stmt_fetch *) stmt);
			break;
		case UPLPGSQL_STMT_CLOSE:
			compile_close((UPLpgSQL_stmt_close *) stmt);
			break;
		case UPLPGSQL_STMT_FORS:
			compile_fors((UPLpgSQL_stmt_fors *) stmt);
			break;
		case UPLPGSQL_STMT_FORC:
			compile_forc((UPLpgSQL_stmt_forc *) stmt);
			break;
		case UPLPGSQL_STMT_DYNEXECUTE:
			compile_dynexecute((UPLpgSQL_stmt_dynexecute *) stmt);
			break;
		case UPLPGSQL_STMT_DYNFORS:
			compile_dynfors((UPLpgSQL_stmt_dynfors *) stmt);
			break;
		case UPLPGSQL_STMT_FOREACH_A:
			compile_foreach_a((UPLpgSQL_stmt_foreach_a *) stmt);
			break;
		case UPLPGSQL_STMT_RETURN_NEXT:
			compile_return_next((UPLpgSQL_stmt_return_next *) stmt);
			break;
		case UPLPGSQL_STMT_RETURN_QUERY:
			compile_return_query((UPLpgSQL_stmt_return_query *) stmt);
			break;
		case UPLPGSQL_STMT_CALL:
			compile_call((UPLpgSQL_stmt_call *) stmt);
			break;
		case UPLPGSQL_STMT_GETDIAG:
			compile_getdiag((UPLpgSQL_stmt_getdiag *) stmt);
			break;
		case UPLPGSQL_STMT_COMMIT:
			compile_commit((UPLpgSQL_stmt_commit *) stmt);
			break;
		case UPLPGSQL_STMT_ROLLBACK:
			compile_rollback((UPLpgSQL_stmt_rollback *) stmt);
			break;
		default:
			elog(ERROR, "uplpgsql: unsupported statement type %d",
				 stmt->cmd_type);
			break;
	}
}

/*
 * Compile a BLOCK statement.
 *
 * If the block has exception handlers, we emit native LLVM IR that calls
 * sigsetjmp directly.  The pattern is:
 *
 *   frame = rt_exception_push_frame(estate, block)   // alloc, begin subtxn
 *   rc = sigsetjmp(frame, 0)                         // returns_twice
 *   if (rc == 0) goto try_enter else goto catch_enter
 *
 *   try_enter:
 *     rt_exception_arm(estate, frame)                 // set PG_exception_stack
 *     <init vars, compiled body>
 *     goto try_exit
 *
 *   exception_return_bb:                              // RETURN inside try body
 *     rt_exception_try_exit(estate, frame)
 *     goto real_return_bb
 *
 *   try_exit:
 *     rt_exception_try_exit(estate, frame)
 *     goto after_block
 *
 *   catch_enter:
 *     handler_idx = rt_exception_catch(estate, block, frame)
 *     switch handler_idx [0->h0, 1->h1, ..., default->rethrow]
 *
 *   handler_N:
 *     rt_exception_set_handler_vars(estate, block, N)
 *     <compiled handler body>
 *     rt_exception_handler_done(estate, frame)
 *     goto after_block
 *
 *   rethrow:
 *     rt_exception_rethrow(estate, frame)
 *     unreachable
 *
 * Otherwise, variables are initialized inline and body is compiled normally.
 */
void
function_compiler::compile_block(UPLpgSQL_stmt_block *stmt)
{
	/*
	 * Initialize the block's variables here instead of letting the core do
	 * it, so native arrays get refreshed from the Datum their DECLARE
	 * default writes; then pass an empty initvarnos so the core does not
	 * repeat the loop.  Blocks with exception handlers are delegated whole
	 * to compile_block_exceptions(), which does its own init inside
	 * the TRY — the core ignores initvarnos on that path.
	 */
	if (stmt->exceptions != NULL)
	{
		/*
		 * The exception path pushes its own block label, because there an
		 * EXIT must route through the subtransaction release rather than
		 * branch past it.
		 */
		upl_emit_block(ctx, {}, stmt->body, true, stmt,
					   cb_compile_block_exceptions);
		return;
	}

	emit_init_vars(llvm::ArrayRef<int>(stmt->initvarnos, stmt->n_initvars));

	/*
	 * A labeled block is an EXIT target as well as a loop is: "EXIT <label>"
	 * on a named block leaves the block and resumes after it.
	 */
	if (stmt->label != NULL)
	{
		llvm::BasicBlock *end_bb = upl_append_block(ctx, "block.end");

		upl_push_block_label(ctx, stmt->label, end_bb);

		upl_emit_block(ctx, {}, stmt->body, false, stmt,
					   cb_compile_block_exceptions);

		upl_pop_loop(ctx);

		if (ctx->builder->GetInsertBlock()->getTerminator() == NULL)
			ctx->builder->CreateBr(end_bb);
		ctx->builder->SetInsertPoint(end_bb);
	}
	else
		upl_emit_block(ctx, {}, stmt->body, false, stmt,
					   cb_compile_block_exceptions);
}

/*
 * Compile a RETURN statement
 *
 * Delegates to uplpgsql_rt_exec_return which calls exec_stmt_return.
 * This handles all RETURN variants (expression, variable, void, composite).
 * The runtime helper sets retval/retisnull/rettype in the PL/pgSQL estate.
 */

/*
 * Sync all active native arrays to PG Datum variables.
 *
 * Called lazily before runtime helpers that evaluate expressions via
 * the interpreter (RETURN, RAISE, SPI).  Builds a PG array Datum
 * from the flat native memory so the interpreter sees valid data.
 * This avoids the cost of syncing on every subscript write.
 */
void
function_compiler::emit_sync_native_array(UPLpgSQL_native_array *na)
{
	llvm::Value *estate_ref = ctx->function->getArg(0);
	llvm::Type *i32_ty = ctx->types[UPL_INT32];
	llvm::Value *data, *len, *lb, *nulls;
	llvm::Value *args[8];

	data = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
						  na->data_ptr, "sync.data");
	len = ctx->builder->CreateLoad(i32_ty,
						 na->len_ptr, "sync.len");
	lb = ctx->builder->CreateLoad(i32_ty, na->lb_ptr, "sync.lb");
	nulls = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
						   na->nulls_ptr, "sync.nulls");

	args[0] = estate_ref;
	args[1] = llvm::ConstantInt::get(i32_ty, na->dno, false);
	args[2] = data;
	args[3] = len;
	args[4] = lb;
	args[5] = nulls;
	args[6] = llvm::ConstantInt::get(i32_ty, na->elemtype, false);
	args[7] = llvm::ConstantInt::get(i32_ty, na->elem_size, false);

	ctx->builder->CreateCall(ctx->rt_funcs[RT_NATIVE_ARRAY_TO_DATUM],
							 args, "");
}

void
function_compiler::sync_native_arrays()
{
	for (UPLpgSQL_native_array &na : native_arrays_)
		emit_sync_native_array(&na);
}

/*
 * Marshal only the native arrays that an expression reads.
 *
 * The interpreter reads variables as Datums, so any native array the
 * expression touches must be marshalled out to its Datum first — but only
 * those.  Syncing the rest costs a full copy of each, per execution, for
 * data the callee cannot even see: three "least(greatest(x,0),1)" clamps per
 * pixel dragged a 6912-element image array through construct_md_array every
 * time, and cost more than the path tracer around them.
 *
 * paramnos is exactly the set of variables the expression reads, and it is
 * what escape analysis already trusts for RETURN/RAISE/FOREACH.  A NULL expr
 * means we cannot tell, so fall back to marshalling everything.
 *
 * paramnos is only trustworthy once the expression has been through parse
 * analysis, which fills it in as each variable reference is resolved
 * (make_datum_param).  Compilation runs before the function has ever
 * executed, and the only expressions analyzed by then are the ones
 * try_compile_assign()/_bool() got as far as preparing.  An ASSIGN
 * whose target is not a plain variable bails out before preparing, and a
 * prepare that fails mid-analysis (a record field, say) leaves paramnos
 * partially filled.  Either way expr->plan is still NULL — the plan is only
 * set after a successful prepare — and a read-set taken from it would be
 * empty or incomplete, baking a stale-array read into the compiled code for
 * good.  Treat "no plan" as "read set unknown" and marshal everything.
 */
void
function_compiler::sync_native_arrays_for_expr(UPLpgSQL_expr *expr)
{
	if (expr == NULL || expr->plan == NULL)
	{
		sync_native_arrays();
		return;
	}

	for (UPLpgSQL_native_array &na : native_arrays_)
	{
		if (expr_references_dno(expr, na.dno))
			emit_sync_native_array(&na);
	}
}

/*
 * Look up native array metadata by dno, or NULL if dno is not one.
 *
 * The returned pointer stays valid for the whole compilation: the vector is
 * never resized after analyze_native_arrays() has built it.
 */
UPLpgSQL_native_array *
function_compiler::find_native_array(int dno)
{
	for (UPLpgSQL_native_array &na : native_arrays_)
	{
		if (na.dno == dno)
			return &na;
	}

	return NULL;
}

/*
 * Reload one native array's flat memory from its PG Datum variable.
 *
 * The inverse of sync_native_arrays().  Called after the
 * interpreter has written a whole array Datum into the variable (SELECT
 * INTO, or an ASSIGN that fell back to exec_assign_expr), which leaves
 * data_ptr/len_ptr pointing at stale memory.  Without this, a later
 * native subscript read would return the pre-assignment contents.
 */
void
function_compiler::emit_refresh_native_array(UPLpgSQL_native_array *na)
{
	llvm::Value *estate_ref = ctx->function->getArg(0);
	llvm::Type	 *i32_ty = ctx->types[UPL_INT32];
	llvm::Value *datum_val, *isnull_val, *new_data, *new_len, *new_lb;
	llvm::Value *nelems_ptr, *lb_ptr, *nulls_out_ptr, *new_nulls;

	datum_val = upl_emit_load_var_datum(ctx, estate_ref, na->dno);
	isnull_val = upl_emit_load_var_isnull(ctx, estate_ref, na->dno);

	nelems_ptr = ctx->builder->CreateAlloca(i32_ty, nullptr, "na_nelems");
	lb_ptr = ctx->builder->CreateAlloca(i32_ty, nullptr, "na_lb");
	nulls_out_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_PTR], nullptr, "na_nulls_out");
	isnull_val = ctx->builder->CreateZExt(isnull_val, i32_ty, "isnull_i32");

	/*
	 * The mirror is being replaced wholesale, so release the old buffers
	 * first — they have no other referent.  Without this every refresh
	 * leaked the previous copy into datum_context, and a loop of
	 * out-of-range writes (each one a sync/set/refresh round trip)
	 * accumulated O(n^2) bytes over the call.  The array_fill stack buffer
	 * is skipped via the is_heap flag.
	 */
	{
		llvm::Value *rel_args[3];

		rel_args[0] = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
									 na->data_ptr, "na.old_data");
		rel_args[1] = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
									 na->nulls_ptr, "na.old_nulls");
		rel_args[2] = ctx->builder->CreateLoad(ctx->types[UPL_INT8],
									 na->is_heap_ptr, "na.old_onheap");

		ctx->builder->CreateCall(ctx->rt_funcs[RT_NATIVE_ARRAY_RELEASE],
							 rel_args, "");
	}

	{
		llvm::Value *call_args[] = {
			estate_ref,
			datum_val,
			isnull_val,
			llvm::ConstantInt::get(i32_ty, na->elem_size, false),
			nelems_ptr,
			lb_ptr,
			nulls_out_ptr
		};

		new_data = ctx->builder->CreateCall(ctx->rt_funcs[RT_NATIVE_ARRAY_FROM_DATUM],
							 call_args, "na.from_datum");
	}

	ctx->builder->CreateStore(new_data, na->data_ptr);
	new_len = ctx->builder->CreateLoad(i32_ty, nelems_ptr, "na.new_len");
	ctx->builder->CreateStore(new_len, na->len_ptr);
	new_lb = ctx->builder->CreateLoad(i32_ty, lb_ptr, "na.new_lb");
	ctx->builder->CreateStore(new_lb, na->lb_ptr);
	new_nulls = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
							   nulls_out_ptr, "na.new_nulls");
	ctx->builder->CreateStore(new_nulls, na->nulls_ptr);

	/*
	 * from_datum allocates exactly len elements, on the heap.  (len can be 0
	 * or -1 with a NULL data pointer; the capacity is never consulted then,
	 * because the append test requires len >= 0 and a reserve call fixes the
	 * buffers up before anything is stored.)
	 */
	ctx->builder->CreateStore(new_len, na->cap_ptr);
	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT8], 1, false),
				   na->is_heap_ptr);
}

void
function_compiler::compile_return(UPLpgSQL_stmt_return *stmt)
{
	/* Sync native arrays before interpreter evaluates RETURN expression */
	if (!native_arrays_.empty())
		sync_native_arrays();

	upl_emit_return(ctx, RT_EXEC_RETURN, stmt);
}

/*
 * Compile an ASSIGN statement: varno := expr
 */
void
function_compiler::compile_assign(UPLpgSQL_stmt_assign *stmt)
{
	UPLpgSQL_native_array *target_na;

	/* Try to inline as native int4 arithmetic */
	if (try_compile_assign(stmt))
		return;

	/*
	 * Read escape (y := x) is handled by call_exec below, which
	 * syncs native arrays before entering the interpreter.
	 */

	/* Fall back to direct exec_assign_expr(plstate, datums[dno], expr) */
	{
		llvm::Value *off, *datums_gep, *datums_ptr, *elem_gep, *datum_ptr;
		llvm::Value *args[3];

		/* Load plstate->datums */
		off = llvm::ConstantInt::get(ctx->types[UPL_INT64],
						   offsetof(UPLpgSQL_execstate, datums), false);
		datums_gep = ctx->builder->CreateGEP(ctx->types[UPL_INT8],
							 plstate_ref_, off, "datums.ptr");
		datums_ptr = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
									datums_gep, "datums");

		/* Load datums[dno] */
		off = llvm::ConstantInt::get(ctx->types[UPL_INT64],
						   stmt->varno * sizeof(void *), false);
		elem_gep = ctx->builder->CreateGEP(ctx->types[UPL_INT8],
							 datums_ptr, off, "datum.gep");
		datum_ptr = ctx->builder->CreateLoad(ctx->types[UPL_PTR],
								  elem_gep, "datum");

		args[0] = plstate_ref_;
		args[1] = datum_ptr;
		args[2] = llvm_const_ptr(ctx, stmt->expr);

		/*
		 * exec_assign_expr reads exactly what stmt->expr reads, so marshal
		 * only that.  An assignment whose value happens not to compile must
		 * not drag every other array in the function through a round trip.
		 */
		sync_native_arrays_for_expr(stmt->expr);
		call_exec_nosync((void *) exec_assign_expr,
								  ctx->types[UPL_VOID], args);

		/*
		 * Clear any plan created at compile time so that exec_assign_expr
		 * can re-prepare it with exec_simple_check_plan, enabling the fast
		 * "simple expression" evaluation path.  Without this, expressions
		 * that were SPI_prepare'd but couldn't inline would be stuck on the
		 * slow full-SPI executor path (30x+ overhead).  This must come after
		 * the marshal decision above, which reads the plan as evidence that
		 * paramnos reflects a completed parse analysis.
		 */
		if (stmt->expr->plan != NULL)
		{
			cppgres::ffi_guard{::SPI_freeplan}(stmt->expr->plan);
			stmt->expr->plan = NULL;
		}
	}

	/*
	 * Write escape: if the target is itself a native array, the interpreter
	 * just stored a whole PG array Datum into it (x := ARRAY[...]), leaving
	 * data_ptr/len_ptr stale.  Reload flat memory from the new Datum so
	 * later native subscript reads see it.
	 */
	target_na = find_native_array(stmt->varno);
	if (target_na != NULL)
	{
		elog(DEBUG1, "uplpgsql: native array from_datum dno %d (ASSIGN)",
			 target_na->dno);
		emit_refresh_native_array(target_na);
	}
}

/*
 * Compile an IF / ELSIF / ELSE statement
 */
void
function_compiler::compile_if(UPLpgSQL_stmt_if *stmt)
{
	llvm::SmallVector<UPL_branch, 8> elsifs;

	for (auto *elsif : cppgres::list<UPLpgSQL_if_elsif *>(stmt->elsif_list))
		elsifs.push_back({elsif->cond, elsif->stmts});

	upl_emit_if(ctx,
				stmt->cond,
				stmt->then_body,
				elsifs,
				stmt->else_body != NIL ? stmt->else_body : NULL);
}

/*
 * Compile a WHILE loop — native LLVM loop.
 *
 * Structure:
 *   while.cond:  eval condition via RT_EVAL_BOOL → branch
 *   while.body:  compiled statements → branch to while.cond
 *   while.exit:  continue after loop
 */
void
function_compiler::compile_while(UPLpgSQL_stmt_while *stmt)
{
	upl_emit_cond_loop(ctx, stmt->label, stmt->cond, stmt->test_at_top, stmt->body);
}

/*
 * Compile a LOOP (unconditional) — native LLVM loop.
 *
 * Structure:
 *   loop.body:  compiled statements → branch to loop.body
 *   loop.exit:  continue after loop (reached via EXIT statement)
 */
void
function_compiler::compile_loop(UPLpgSQL_stmt_loop *stmt)
{
	upl_emit_loop(ctx, stmt->label, stmt->body);
}

/*
 * Compile a FOR integer-range loop — native LLVM loop.
 *
 * Evaluates lower/upper/step bounds once via runtime helpers, then the
 * actual loop (compare, branch, increment) is pure LLVM IR.
 *
 * Structure:
 *   fori.setup:  evaluate bounds, initialize loop_val alloca
 *   fori.cond:   compare loop_val vs upper → branch
 *   fori.body:   assign loop var, compile body
 *   fori.step:   increment/decrement with overflow check → branch to cond
 *   fori.exit:   set FOUND, continue
 */
void
function_compiler::compile_fori(UPLpgSQL_stmt_fori *stmt)
{
	upl_emit_fori(ctx, stmt->label,
				  stmt->var->dno,
				  stmt->lower, stmt->upper, stmt->step,
				  stmt->reverse,
				  stmt->body);
}

/*
 * Compile EXIT / CONTINUE statement.
 *
 * EXIT branches to the loop's exit_bb.
 * CONTINUE branches to the loop's continue_bb.
 * If there's a condition, wrap in a conditional branch.
 * If there's a label, find the matching loop on the stack.
 */
void
function_compiler::compile_exit(UPLpgSQL_stmt_exit *stmt)
{
	upl_emit_loop_exit(ctx, stmt->label, stmt->is_exit, stmt->cond);
}

/*
 * Compile PERFORM statement — delegates to runtime.
 */
void
function_compiler::compile_perform(UPLpgSQL_stmt_perform *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_perform,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile EXECSQL statement
 */
void
function_compiler::compile_execsql(UPLpgSQL_stmt_execsql *stmt)
{
	llvm::Value *exec_args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_execsql,
					   ctx->types[UPL_INT32], exec_args);

	/*
	 * If the SELECT INTO target is a native array, decompose the PG array
	 * Datum that exec_stmt_execsql stored into the variable into flat
	 * native memory.  The interpreter stores a normal PG array Datum, but
	 * native array subscript reads expect data_ptr/len_ptr to point to
	 * flat element memory.
	 */
	if (stmt->into && stmt->target != NULL)
	{

		/*
		 * Check if the INTO target (or any variable it wraps) is a
		 * native array.  The target may be a VAR (direct) or a ROW
		 * wrapping one or more VARs.  For multi-target INTO (ROW with
		 * multiple fields), check each field.
		 */
		int		target_dnos[64];
		int		num_targets = 0;
		int		ti;

		if (stmt->target->dtype == UPLPGSQL_DTYPE_ROW)
		{
			UPLpgSQL_row *row = (UPLpgSQL_row *) stmt->target;
			int		fi;

			for (fi = 0; fi < row->nfields && fi < 64; fi++)
				target_dnos[num_targets++] = row->varnos[fi];
		}
		else
		{
			target_dnos[num_targets++] = stmt->target->dno;
		}

		for (ti = 0; ti < num_targets; ti++)
		{
			UPLpgSQL_native_array *na;

			na = find_native_array(target_dnos[ti]);
			if (na != NULL)
			{
				elog(DEBUG1, "uplpgsql: native array from_datum dno %d "
					 "(SELECT INTO)", na->dno);
				emit_refresh_native_array(na);
			}
		}
	}
}

/*
 * Compile RAISE statement
 */
void
function_compiler::compile_raise(UPLpgSQL_stmt_raise *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_raise,
					   ctx->types[UPL_VOID], args);
}

/*
 * Compile a cursor OPEN statement.
 *
 * Delegates directly to exec_stmt_open for portal creation. The three
 * OPEN variants (static query, dynamic query, explicit cursor) are
 * handled by the exec function based on stmt fields set at parse time.
 */
void
function_compiler::compile_open(UPLpgSQL_stmt_open *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_open,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile a cursor FETCH/MOVE statement.
 *
 * Delegates directly to exec_stmt_fetch which handles cursor lookup,
 * row fetching, target assignment, and FOUND variable setting.
 */
void
function_compiler::compile_fetch(UPLpgSQL_stmt_fetch *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_fetch,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile a cursor CLOSE statement.
 */
void
function_compiler::compile_close(UPLpgSQL_stmt_close *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_close,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile an ASSERT statement.
 *
 * Evaluates the condition via RT_EVAL_BOOL, then branches:
 * true → continue, false/null → call RT_EXEC_ASSERT_FAIL which
 * checks the GUC, evaluates the optional message, and raises ERROR.
 *
 * Note: we always evaluate the condition even if asserts are disabled.
 * The runtime helper swallows the failure in that case. This avoids
 * needing to embed a GUC pointer in the IR.
 */
void
function_compiler::compile_assert(UPLpgSQL_stmt_assert *stmt)
{
	llvm::Value		*estate_ref = ctx->function->getArg(0);
	llvm::Value		*cond;
	llvm::BasicBlock	*fail_bb;
	llvm::BasicBlock	*cont_bb;

	/* Evaluate condition — try native inlining first */
	cond = try_compile_bool(stmt->cond);
	if (cond == NULL)
	{
		llvm::Value *args[] = {
			estate_ref,
			llvm_const_ptr(ctx, stmt->cond)
		};
		cond = call_fn(RT_EVAL_BOOL, args);
	}

	fail_bb = append_block(ctx, "assert.fail");
	cont_bb = append_block(ctx, "assert.cont");

	ctx->builder->CreateCondBr(cond, cont_bb, fail_bb);

	/* Failure path: delegate to runtime for message eval + error */
	ctx->builder->SetInsertPoint(fail_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			llvm_const_ptr(ctx, stmt)
		};
		call_fn(RT_EXEC_ASSERT_FAIL, args);
	}
	/* RT_EXEC_ASSERT_FAIL either errors or returns (if asserts disabled) */
	ctx->builder->CreateBr(cont_bb);

	/* Continue after assert */
	ctx->builder->SetInsertPoint(cont_bb);
}

/*
 * Compile a CASE statement — native LLVM IR.
 *
 * Two variants:
 * 1. Simple CASE (t_expr != NULL): evaluate search expression once, store in
 *    temp variable, then each WHEN condition compares against it.
 * 2. Searched CASE (t_expr == NULL): each WHEN is an independent boolean.
 *
 * Both compile to a chain of conditional branches, like IF/ELSIF/ELSE.
 * The PL/pgSQL parser rewrites both forms so that each WHEN expr is a boolean
 * expression — the simple CASE's WHEN exprs contain implicit "t_var = value"
 * comparisons. So we just evaluate each WHEN as a boolean.
 */
void
function_compiler::compile_case(UPLpgSQL_stmt_case *stmt)
{
	llvm::SmallVector<UPL_branch, 8> whens;

	for (auto *cwt : cppgres::list<UPLpgSQL_case_when *>(stmt->case_when_list))
		whens.push_back({cwt->expr, cwt->stmts});

	upl_emit_case(ctx,
				  stmt->t_expr != NULL,
				  stmt->t_varno,
				  stmt->t_expr,
				  whens,
				  stmt->have_else,
				  stmt->else_stmts,
				  stmt->lineno);
}

/*
 * Compile a FOR-query (FORS) loop — native LLVM IR loop.
 *
 * Pattern:
 *   portal = rt_open_query_cursor(estate, query)
 *   found = false
 *   goto cond
 * cond:
 *   has_row = rt_fetch_cursor_row(estate, portal, target_dno)
 *   br has_row → body, exit
 * body:
 *   found = true
 *   <body statements>
 *   goto cond    (CONTINUE also goes here)
 * fors_return:
 *   rt_close_portal(estate, portal)
 *   goto real_return_bb
 * exit:
 *   rt_close_portal(estate, portal)
 *   rt_set_found(estate, found)
 *   <continue after loop>
 *
 * RETURN inside the body is redirected to fors_return_bb which closes
 * the portal before jumping to the function's return block.  EXIT
 * branches to exit_bb which also closes the portal.
 */
void
function_compiler::compile_fors(UPLpgSQL_stmt_fors *stmt)
{
	llvm::Value		*estate_ref = ctx->function->getArg(0);
	llvm::Value		*portal_val, *has_row, *found_val;
	llvm::Value		*found_ptr;
	llvm::BasicBlock	*cond_bb, *body_bb, *exit_bb, *fors_return_bb, *cont_bb;
	llvm::BasicBlock	*saved_return_bb;
	int					target_dno = stmt->var->dno;

	/*
	 * Opening the cursor evaluates the loop query in the interpreter, which
	 * reads variables as PG Datums.  This goes through call_fn
	 * rather than call_exec, so it does not get that function's
	 * sync — do it here, or a query over a native array (FOR r IN SELECT
	 * unnest(x)) sees a stale NULL and returns no rows.
	 */
	sync_native_arrays();

	/* Open the query cursor */
	{
		llvm::Value *args[] = {
			estate_ref,
			llvm_const_ptr(ctx, stmt->query)
		};
		portal_val = call_fn(RT_OPEN_QUERY_CURSOR, args);
	}

	/* Allocate found flag */
	found_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT1], nullptr, "fors_found");
	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT1], 0, false),
				   found_ptr);

	cond_bb = append_block(ctx, "fors.cond");
	body_bb = append_block(ctx, "fors.body");
	exit_bb = append_block(ctx, "fors.exit");
	fors_return_bb = append_block(ctx, "fors.return");
	cont_bb = append_block(ctx, "fors.cont");

	ctx->builder->CreateBr(cond_bb);

	/* --- Condition: fetch next row --- */
	ctx->builder->SetInsertPoint(cond_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			portal_val,
			llvm_const_int32(ctx, target_dno)
		};
		has_row = call_fn(RT_FETCH_CURSOR_ROW, args);
	}
	ctx->builder->CreateCondBr(has_row, body_bb, exit_bb);

	/* --- Body --- */
	ctx->builder->SetInsertPoint(body_bb);

	/* Set found = true */
	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT1], 1, false),
				   found_ptr);

	/* Redirect RETURN to fors_return_bb for portal cleanup */
	saved_return_bb = ctx->return_bb;
	ctx->return_bb = fors_return_bb;

	/*
	 * An EXIT/CONTINUE in the body whose target lies outside this loop jumps
	 * across the open portal, so upl_emit_loop_exit() must close it on the
	 * way — a pinned portal left behind makes the surrounding transaction
	 * unable to commit.  Push the cleanup before the loop's own entry: EXIT
	 * of this loop lands on exit_bb, which closes the portal itself.
	 */
	{
		llvm::Value *cleanup_args[] = { portal_val };

		upl_push_cleanup(ctx, RT_CLOSE_PORTAL, cleanup_args);
	}

	/* Push loop: CONTINUE → cond_bb, EXIT → exit_bb */
	upl_push_loop(ctx, stmt->label, cond_bb, exit_bb);

	/*
	 * Phase 5d: Record field inlining for FOR-query loops.
	 *
	 * If the loop variable is a RECORD-typed rec with no erh, the
	 * expression compiler can't resolve field references like r.a because
	 * the parser callback chain needs a TupleDesc.  We SPI_prepare the
	 * query at compile time, extract resultDesc from the CachedPlanSource,
	 * and create a temporary ExpandedRecordHeader so that body expressions
	 * can resolve record fields through Tiers 1/2.
	 */
	{
		UPLpgSQL_datum	   *target_datum;
		UPLpgSQL_rec	   *rec = NULL;
		ExpandedRecordHeader *saved_erh = NULL;
		bool				installed_erh = false;

		target_datum = func_->datums[target_dno];

		if (target_datum->dtype == UPLPGSQL_DTYPE_REC)
		{
			rec = (UPLpgSQL_rec *) target_datum;

			if (rec->rectypeid == RECORDOID && rec->erh == NULL &&
				stmt->query->plan == NULL)
			{
				SPIPlanPtr			plan;

				plan = prepare_plan_compile_time(stmt->query);

				if (plan != NULL)
				{
					List		   *plansources;
					CachedPlanSource *plansource;
					TupleDesc		tupdesc;

					stmt->query->plan = plan;

					plansources = SPI_plan_get_plan_sources(plan);
					if (list_length(plansources) == 1)
					{
						plansource = (CachedPlanSource *) linitial(plansources);
						tupdesc = plansource->resultDesc;

						if (tupdesc != NULL)
						{
							saved_erh = rec->erh;
							rec->erh = make_expanded_record_from_tupdesc(
								tupdesc, CurrentMemoryContext);
							installed_erh = true;

							elog(DEBUG1,
								 "uplpgsql: installed temporary erh for "
								 "record \"%s\" (dno %d) with %d columns",
								 rec->refname, target_dno,
								 tupdesc->natts);
						}
					}
				}
			}
		}

		/* Compile loop body */
		compile_stmts(stmt->body);

		/* Clean up temporary ExpandedRecordHeader */
		if (installed_erh)
		{
			/*
			 * We don't pfree the expanded record — it lives in
			 * CurrentMemoryContext and will be cleaned up with the
			 * compilation memory context.  Just restore the original
			 * (NULL) erh so runtime doesn't see stale compile-time data.
			 */
			rec->erh = saved_erh;
		}
	}

	upl_pop_loop(ctx);
	upl_pop_cleanup(ctx);

	/* Restore real return block */
	ctx->return_bb = saved_return_bb;

	/* Fall through to next iteration */
	ctx->builder->CreateBr(cond_bb);

	/* --- fors_return_bb: close portal then jump to real return --- */
	ctx->builder->SetInsertPoint(fors_return_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			portal_val
		};
		call_fn(RT_CLOSE_PORTAL, args);
	}
	ctx->builder->CreateBr(saved_return_bb);

	/* --- Exit: close portal, set FOUND, continue --- */
	ctx->builder->SetInsertPoint(exit_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			portal_val
		};
		call_fn(RT_CLOSE_PORTAL, args);
	}

	/* Set FOUND */
	found_val = ctx->builder->CreateLoad(ctx->types[UPL_INT1],
							   found_ptr, "found");
	{
		llvm::Value *args[] = {
			plstate_ref_,
			found_val
		};
		call_exec((void *) exec_set_found,
						   ctx->types[UPL_VOID], args);
	}

	ctx->builder->CreateBr(cont_bb);

	/* Continue after loop */
	ctx->builder->SetInsertPoint(cont_bb);
}

/*
 * Compile a FOR-cursor (FORC) loop — native LLVM IR loop.
 *
 * Same loop pattern as FORS, but cursor open/close uses FORC-specific
 * runtime helpers that handle argument evaluation, plan preparation,
 * and cursor variable management.
 */
void
function_compiler::compile_forc(UPLpgSQL_stmt_forc *stmt)
{
	llvm::Value		*estate_ref = ctx->function->getArg(0);
	llvm::Value		*portal_val, *has_row, *found_val;
	llvm::Value		*found_ptr;
	llvm::Value		*stmt_ptr = llvm_const_ptr(ctx, stmt);
	llvm::BasicBlock	*cond_bb, *body_bb, *exit_bb, *forc_return_bb, *cont_bb;
	llvm::BasicBlock	*saved_return_bb;
	int					target_dno = stmt->var->dno;

	/*
	 * Open the cursor.
	 *
	 * This must go through uplpgsql_rt_open_forc_cursor(), not straight to
	 * exec_open_forc_cursor(): the latter returns a bare Portal, but the
	 * fetch below reads the result as a UPLpgSQL_cursor_ctx (portal plus
	 * prefetch batch state).  The rt wrapper allocates that context and is
	 * what uplpgsql_rt_close_forc_cursor() expects to free.
	 *
	 * Opening evaluates the cursor's query and any arguments in the
	 * interpreter, and call_fn does not sync native arrays the way
	 * call_exec does, so sync explicitly first.
	 */
	sync_native_arrays();
	{
		llvm::Value *args[] = {
			ctx->function->getArg(0),
			stmt_ptr
		};
		portal_val = call_fn(RT_OPEN_FORC_CURSOR, args);
	}

	/* Allocate found flag */
	found_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT1], nullptr, "forc_found");
	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT1], 0, false),
				   found_ptr);

	cond_bb = append_block(ctx, "forc.cond");
	body_bb = append_block(ctx, "forc.body");
	exit_bb = append_block(ctx, "forc.exit");
	forc_return_bb = append_block(ctx, "forc.return");
	cont_bb = append_block(ctx, "forc.cont");

	ctx->builder->CreateBr(cond_bb);

	/* --- Condition: fetch next row --- */
	ctx->builder->SetInsertPoint(cond_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			portal_val,
			llvm_const_int32(ctx, target_dno)
		};
		has_row = call_fn(RT_FETCH_CURSOR_ROW, args);
	}
	ctx->builder->CreateCondBr(has_row, body_bb, exit_bb);

	/* --- Body --- */
	ctx->builder->SetInsertPoint(body_bb);

	/* Set found = true */
	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT1], 1, false),
				   found_ptr);

	/* Redirect RETURN to forc_return_bb for portal cleanup */
	saved_return_bb = ctx->return_bb;
	ctx->return_bb = forc_return_bb;

	/*
	 * As in FORS: an EXIT/CONTINUE leaving this loop must close the cursor
	 * on the way out.  The FORC close helper also takes the statement, to
	 * unbind the cursor variable and free the prefetch context.
	 */
	{
		llvm::Value *cleanup_args[] = { stmt_ptr, portal_val };

		upl_push_cleanup(ctx, RT_CLOSE_FORC_CURSOR, cleanup_args);
	}

	/* Push loop: CONTINUE → cond_bb, EXIT → exit_bb */
	upl_push_loop(ctx, stmt->label, cond_bb, exit_bb);

	/* Compile loop body */
	compile_stmts(stmt->body);

	upl_pop_loop(ctx);
	upl_pop_cleanup(ctx);

	/* Restore real return block */
	ctx->return_bb = saved_return_bb;

	/* Fall through to next iteration */
	ctx->builder->CreateBr(cond_bb);

	/* --- forc_return_bb: close cursor then jump to real return --- */
	ctx->builder->SetInsertPoint(forc_return_bb);
	{
		llvm::Value *args[] = {
			ctx->function->getArg(0),
			stmt_ptr,
			portal_val
		};
		call_fn(RT_CLOSE_FORC_CURSOR, args);
	}
	ctx->builder->CreateBr(saved_return_bb);

	/* --- Exit: close cursor, set FOUND, continue --- */
	ctx->builder->SetInsertPoint(exit_bb);
	{
		llvm::Value *args[] = {
			ctx->function->getArg(0),
			stmt_ptr,
			portal_val
		};
		call_fn(RT_CLOSE_FORC_CURSOR, args);
	}

	/* Set FOUND */
	found_val = ctx->builder->CreateLoad(ctx->types[UPL_INT1],
							   found_ptr, "found");
	{
		llvm::Value *args[] = {
			plstate_ref_,
			found_val
		};
		call_exec((void *) exec_set_found,
						   ctx->types[UPL_VOID], args);
	}

	ctx->builder->CreateBr(cont_bb);

	/* Continue after loop */
	ctx->builder->SetInsertPoint(cont_bb);
}

/*
 * Compile EXECUTE (dynamic SQL) — delegate to runtime.
 */
void
function_compiler::compile_dynexecute(UPLpgSQL_stmt_dynexecute *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_dynexecute,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile DYNFORS (FOR ... IN EXECUTE) — native LLVM IR loop.
 *
 * Same pattern as FORS, but uses a dynamic query cursor opener.
 */
void
function_compiler::compile_dynfors(UPLpgSQL_stmt_dynfors *stmt)
{
	llvm::Value		*estate_ref = ctx->function->getArg(0);
	llvm::Value		*portal_val, *has_row, *found_val;
	llvm::Value		*found_ptr;
	llvm::BasicBlock	*cond_bb, *body_bb, *exit_bb, *dynfors_return_bb, *cont_bb;
	llvm::BasicBlock	*saved_return_bb;
	int					target_dno = stmt->var->dno;

	/*
	 * As in compile_fors(): the open evaluates the query text and
	 * its USING arguments in the interpreter, and reaches it via
	 * call_fn, which does not sync.
	 */
	sync_native_arrays();

	/* Open the dynamic cursor */
	{
		llvm::Value *args[] = {
			estate_ref,
			llvm_const_ptr(ctx, stmt)
		};
		portal_val = call_fn(RT_OPEN_DYNFORS_CURSOR, args);
	}

	/* Allocate found flag */
	found_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT1], nullptr, "dynfors_found");
	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT1], 0, false),
				   found_ptr);

	cond_bb = append_block(ctx, "dynfors.cond");
	body_bb = append_block(ctx, "dynfors.body");
	exit_bb = append_block(ctx, "dynfors.exit");
	dynfors_return_bb = append_block(ctx, "dynfors.return");
	cont_bb = append_block(ctx, "dynfors.cont");

	ctx->builder->CreateBr(cond_bb);

	/* --- Condition: fetch next row --- */
	ctx->builder->SetInsertPoint(cond_bb);
	{
		llvm::Value *args[] = {
			estate_ref,
			portal_val,
			llvm_const_int32(ctx, target_dno)
		};
		has_row = call_fn(RT_FETCH_CURSOR_ROW, args);
	}
	ctx->builder->CreateCondBr(has_row, body_bb, exit_bb);

	/* --- Body --- */
	ctx->builder->SetInsertPoint(body_bb);

	ctx->builder->CreateStore(llvm::ConstantInt::get(ctx->types[UPL_INT1], 1, false),
				   found_ptr);

	saved_return_bb = ctx->return_bb;
	ctx->return_bb = dynfors_return_bb;

	/* As in FORS: close the portal when a jump crosses this loop */
	{
		llvm::Value *cleanup_args[] = { portal_val };

		upl_push_cleanup(ctx, RT_CLOSE_PORTAL, cleanup_args);
	}

	upl_push_loop(ctx, stmt->label, cond_bb, exit_bb);
	compile_stmts(stmt->body);
	upl_pop_loop(ctx);
	upl_pop_cleanup(ctx);

	ctx->return_bb = saved_return_bb;

	ctx->builder->CreateBr(cond_bb);

	/* --- dynfors_return_bb: close portal then jump to real return --- */
	ctx->builder->SetInsertPoint(dynfors_return_bb);
	{
		llvm::Value *args[] = { estate_ref, portal_val };
		call_fn(RT_CLOSE_PORTAL, args);
	}
	ctx->builder->CreateBr(saved_return_bb);

	/* --- Exit: close portal, set FOUND --- */
	ctx->builder->SetInsertPoint(exit_bb);
	{
		llvm::Value *args[] = { estate_ref, portal_val };
		call_fn(RT_CLOSE_PORTAL, args);
	}

	found_val = ctx->builder->CreateLoad(ctx->types[UPL_INT1],
							   found_ptr, "found");
	{
		llvm::Value *args[] = { plstate_ref_, found_val };
		call_exec((void *) exec_set_found,
						   ctx->types[UPL_VOID], args);
	}

	ctx->builder->CreateBr(cont_bb);
	ctx->builder->SetInsertPoint(cont_bb);
}

/*
 * Compile FOREACH array loop — delegate to runtime.
 *
 * Array element/slice iteration involves deep executor internals
 * (array detoasting, element extraction, type coercion). The entire
 * statement is delegated to the interpreter.
 */
void
function_compiler::compile_foreach_a(UPLpgSQL_stmt_foreach_a *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_foreach_a,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile RETURN NEXT — delegate to runtime.
 */
void
function_compiler::compile_return_next(UPLpgSQL_stmt_return_next *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_return_next,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile RETURN QUERY — delegate to runtime.
 */
void
function_compiler::compile_return_query(UPLpgSQL_stmt_return_query *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_return_query,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile CALL — delegate to runtime.
 */
void
function_compiler::compile_call(UPLpgSQL_stmt_call *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_call,
					   ctx->types[UPL_INT32], args);
}

/*
 * Compile GET DIAGNOSTICS — delegate to runtime.
 */
void
function_compiler::compile_getdiag(UPLpgSQL_stmt_getdiag *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_getdiag,
					   ctx->types[UPL_VOID], args);
}

/*
 * Compile COMMIT — delegate to runtime.
 */
void
function_compiler::compile_commit(UPLpgSQL_stmt_commit *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_commit,
					   ctx->types[UPL_VOID], args);
}

/*
 * Compile ROLLBACK — delegate to runtime.
 */
void
function_compiler::compile_rollback(UPLpgSQL_stmt_rollback *stmt)
{
	llvm::Value *args[] = {
		plstate_ref_,
		llvm_const_ptr(ctx, stmt)
	};

	call_exec((void *) exec_stmt_rollback,
					   ctx->types[UPL_VOID], args);
}

/* Loop stack management now lives in core/upl_compile.c:
 * upl_push_loop(), upl_pop_loop(), upl_find_loop()
 */

} /* namespace uplpgsql */
