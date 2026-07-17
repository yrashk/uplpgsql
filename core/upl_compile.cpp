/*-------------------------------------------------------------------------
 *
 * upl_compile.c
 *		Language-agnostic control flow compilation primitives and the
 *		compilation pipeline for the UPL core engine.
 *
 *		All expressions and statement bodies are opaque void* pointers.
 *		Core uses ctx->callbacks to recurse into the driver for expression
 *		evaluation and body compilation.
 *
 *		Functions:
 *		  - Loop stack: push, pop, find
 *		  - Utility calls: upl_emit_rt_call, upl_emit_direct_call
 *		  - Control flow: if, cond_loop, loop, fori, loop_exit, case,
 *		    return, block (with optional exception handling)
 *		  - Pipeline: upl_compile_function
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
#include "upl.h"

#include "cppgres.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "miscadmin.h"
#ifdef __cplusplus
}
#endif

/* See doc/cpp-rewrite.md on the _/gettext dance around LLVM C++ headers. */
#undef _
#undef gettext

#include <llvm/Support/raw_ostream.h>

#ifndef ENABLE_NLS
#define gettext(x) (x)
#endif
#define _(x) gettext(x)

#include <string>

/*
 * Symbol name to emit for sigsetjmp.
 *
 * Compiled functions with an exception handler call sigsetjmp directly — it
 * has to run in the compiled function's own frame, so it cannot be hidden
 * behind a runtime helper — and the JIT resolves it as a process symbol.
 *
 * glibc does not export a "sigsetjmp" symbol at all: <setjmp.h> defines
 * sigsetjmp as a macro over __sigsetjmp(env, savemask).  Emitting a call to
 * "sigsetjmp" therefore fails to materialize on Linux/glibc, and every
 * function containing an EXCEPTION handler dies with a symbol lookup failure.
 * The two take identical arguments, so naming __sigsetjmp is the whole fix.
 *
 * macOS/BSD and musl do export a real sigsetjmp.  __GLIBC__ comes from
 * features.h, which postgres.h pulls in well before this point.
 */
#ifdef __GLIBC__
#define UPL_SIGSETJMP_SYM	"__sigsetjmp"
#else
#define UPL_SIGSETJMP_SYM	"sigsetjmp"
#endif

/* Generation counter for unique LLVM symbol names across recompilations */
static uint64 compile_gen = 0;

/* Internal helper: evaluate a boolean expression */
static llvm::Value *upl_eval_bool(UPL_compile_ctx *ctx, void *cond_expr);


/* ----------------------------------------------------------------
 *		Loop stack management
 * ----------------------------------------------------------------
 */

/*
 * Push a loop onto the tracking stack.
 *
 * continue_bb is the target for CONTINUE/ITERATE statements.
 * exit_bb is the target for EXIT/LEAVE statements.
 */
void
upl_push_loop(UPL_compile_ctx *ctx, const char *label,
			  llvm::BasicBlock *continue_bb,
			  llvm::BasicBlock *exit_bb)
{
	UPL_loop_info info;

	info.label = label;
	info.is_loop = true;
	info.continue_bb = continue_bb;
	info.exit_bb = exit_bb;
	info.cleanup_depth = (int) ctx->cleanup_stack.size();
	ctx->loop_stack.push_back(info);
}

/*
 * Push a labeled BEGIN...END block onto the tracking stack.
 *
 * A labeled block is a target for "EXIT <label>" but never for CONTINUE, so
 * it carries no continue_bb.  exit_bb is where control resumes after the
 * block; for a block with exception handlers it must be the path that still
 * releases the subtransaction, not a jump past it.
 */
void
upl_push_block_label(UPL_compile_ctx *ctx, const char *label,
					 llvm::BasicBlock *exit_bb)
{
	UPL_loop_info info;

	info.label = label;
	info.is_loop = false;
	info.continue_bb = NULL;
	info.exit_bb = exit_bb;
	info.cleanup_depth = (int) ctx->cleanup_stack.size();
	ctx->loop_stack.push_back(info);
}

/*
 * Pop the top loop from the tracking stack.
 */
void
upl_pop_loop(UPL_compile_ctx *ctx)
{
	Assert(!ctx->loop_stack.empty());
	ctx->loop_stack.pop_back();
}

/*
 * Find a loop on the stack by label.
 * NULL label means the innermost loop.
 */
UPL_loop_info *
upl_find_loop(UPL_compile_ctx *ctx, const char *label)
{
	/* Innermost entry is at the back of the vector */
	for (auto it = ctx->loop_stack.rbegin(); it != ctx->loop_stack.rend(); ++it)
	{
		UPL_loop_info *info = &(*it);

		if (label == NULL)
		{
			/*
			 * Unlabeled EXIT/CONTINUE targets the innermost real loop; a
			 * labeled block enclosing it is not a match.
			 */
			if (info->is_loop)
				return info;
			continue;
		}
		if (info->label != NULL && strcmp(info->label, label) == 0)
			return info;
	}

	return NULL;
}


/* ----------------------------------------------------------------
 *		Cleanup stack management
 * ----------------------------------------------------------------
 */

/*
 * Push an enclosing cleanup onto the tracking stack.
 *
 * The driver calls this before compiling statements that can jump across a
 * held resource, passing the rt_funcs[] index of the helper that releases it
 * on that path (try-exit for a try body, handler-done for a handler body,
 * portal-close for a row loop) and the helper's operands after estate_ref
 * (the frame or portal pointer).  An EXIT/CONTINUE compiled while the entry
 * is on the stack and targeting a loop or block outside it will emit that
 * call before branching.
 */
void
upl_push_cleanup(UPL_compile_ctx *ctx, int unwind_rt_fn,
				 llvm::Value **args, int nargs)
{
	UPL_cleanup_info cleanup = {};
	int			i;

	Assert(nargs >= 0 && nargs <= UPL_CLEANUP_MAX_ARGS);

	cleanup.unwind_rt_fn = unwind_rt_fn;
	cleanup.nargs = nargs;
	for (i = 0; i < nargs; i++)
		cleanup.args[i] = args[i];
	ctx->cleanup_stack.push_back(cleanup);
}

/*
 * Pop the top cleanup from the tracking stack.
 */
void
upl_pop_cleanup(UPL_compile_ctx *ctx)
{
	Assert(!ctx->cleanup_stack.empty());
	ctx->cleanup_stack.pop_back();
}

/*
 * Emit the release calls for every cleanup between the current position and
 * target_depth, innermost first.
 *
 * Innermost first matters: a release commits its own subtransaction, pops
 * its own stmt_mcontext, or closes its own portal, and those must come off
 * in the reverse of the order they were entered, exactly as the constructs'
 * own exit paths would have run had control left them one at a time.
 */
static void
upl_emit_cleanup_unwind(UPL_compile_ctx *ctx, int target_depth)
{
	int			depth;

	for (depth = (int) ctx->cleanup_stack.size(); depth > target_depth; depth--)
	{
		UPL_cleanup_info *cleanup = &ctx->cleanup_stack[depth - 1];
		llvm::Value	   *args[1 + UPL_CLEANUP_MAX_ARGS];
		int				i;

		args[0] = ctx->estate_ref;
		for (i = 0; i < cleanup->nargs; i++)
			args[1 + i] = cleanup->args[i];

		upl_emit_rt_call(ctx, cleanup->unwind_rt_fn, args, 1 + cleanup->nargs);
	}
}


/* ----------------------------------------------------------------
 *		Utility call helpers
 * ----------------------------------------------------------------
 */

/*
 * Emit a call to a registered runtime function.
 *
 * rt_func_idx indexes into ctx->rt_funcs[] and ctx->rt_fntypes[].
 */
llvm::Value *
upl_emit_rt_call(UPL_compile_ctx *ctx, int rt_func_idx,
				 llvm::Value **args, unsigned count)
{
	Assert(rt_func_idx >= 0 &&
		   (size_t) rt_func_idx < ctx->rt_funcs.size());

	return ctx->builder->CreateCall(ctx->rt_funcs[rt_func_idx],
									llvm::ArrayRef<llvm::Value *>(args, count),
									"");
}

/*
 * Emit a direct call to a C function via embedded pointer address.
 *
 * Embeds fn_addr as an LLVM integer constant, casts to a function pointer,
 * and calls it.  This bypasses the runtime wrapper layer for thin wrappers
 * and allows LLVM's optimizer to reason about the call more effectively.
 *
 * The function type is constructed from the argument types and ret_type.
 */
llvm::Value *
upl_emit_direct_call(UPL_compile_ctx *ctx, void *fn_addr,
					 llvm::Type *ret_type,
					 llvm::Value **args, unsigned count)
{
	llvm::Type		   *param_types[8];
	llvm::FunctionType *fn_type;
	llvm::Value		   *fn_ptr;
	unsigned			i;
	const char		   *call_name;

	Assert(count <= 8);

	for (i = 0; i < count; i++)
		param_types[i] = args[i]->getType();

	fn_type = llvm::FunctionType::get(ret_type,
									  llvm::ArrayRef<llvm::Type *>(param_types,
																   count),
									  false);
	fn_ptr = upl_const_ptr(ctx, fn_addr);

	/* Use empty name for void returns, "ret" for value-producing calls */
	call_name = (ret_type == ctx->types[UPL_VOID]) ? "" : "ret";

	return ctx->builder->CreateCall(llvm::FunctionCallee(fn_type, fn_ptr),
									llvm::ArrayRef<llvm::Value *>(args, count),
									call_name);
}


/* ----------------------------------------------------------------
 *		Internal: evaluate a boolean expression
 * ----------------------------------------------------------------
 */

/*
 * Try to compile a boolean expression natively via the driver callback.
 * On failure, fall back to the runtime helper at callbacks.rt_eval_bool.
 *
 * Returns an llvm::Value* of type i1.
 */
static llvm::Value *
upl_eval_bool(UPL_compile_ctx *ctx, void *cond_expr)
{
	llvm::Value *result;

	/* Try native (Tier 1/2) compilation first */
	if (ctx->callbacks.try_compile_bool != NULL &&
		ctx->callbacks.try_compile_bool(ctx, cond_expr, &result))
		return result;

	/* Fall back to runtime helper */
	Assert(ctx->callbacks.rt_eval_bool >= 0);
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_ptr(ctx, cond_expr)
		};

		return upl_emit_rt_call(ctx, ctx->callbacks.rt_eval_bool, args, 2);
	}
}


/* ----------------------------------------------------------------
 *		Control flow primitives
 * ----------------------------------------------------------------
 */

/*
 * Compile an IF / ELSIF / ELSE statement.
 *
 * cond_expr:     the IF condition (opaque)
 * then_stmts:    body for the IF branch
 * num_elsifs:    number of ELSIF clauses
 * elsif_conds:   array of ELSIF condition expressions
 * elsif_bodies:  array of ELSIF statement bodies
 * else_stmts:    ELSE body (NULL if no ELSE)
 */
void
upl_emit_if(UPL_compile_ctx *ctx,
			void *cond_expr,
			void *then_stmts,
			int num_elsifs,
			void **elsif_conds,
			void **elsif_bodies,
			void *else_stmts)
{
	llvm::BasicBlock   *then_bb;
	llvm::BasicBlock   *else_bb;
	llvm::BasicBlock   *merge_bb;
	llvm::Value		   *cond;
	int					i;

	merge_bb = upl_append_block(ctx, "if.merge");

	/* Evaluate the IF condition */
	cond = upl_eval_bool(ctx, cond_expr);

	then_bb = upl_append_block(ctx, "if.then");
	else_bb = upl_append_block(ctx, "if.else");

	ctx->builder->CreateCondBr(cond, then_bb, else_bb);

	/* Then block */
	ctx->builder->SetInsertPoint(then_bb);
	ctx->callbacks.compile_stmts(ctx, then_stmts);
	ctx->builder->CreateBr(merge_bb);

	/* Else/elsif chain */
	ctx->builder->SetInsertPoint(else_bb);

	for (i = 0; i < num_elsifs; i++)
	{
		llvm::BasicBlock   *elsif_then;
		llvm::BasicBlock   *elsif_else;
		llvm::Value		   *elsif_cond;

		elsif_cond = upl_eval_bool(ctx, elsif_conds[i]);

		elsif_then = upl_append_block(ctx, "elsif.then");
		elsif_else = upl_append_block(ctx, "elsif.else");

		ctx->builder->CreateCondBr(elsif_cond, elsif_then, elsif_else);

		ctx->builder->SetInsertPoint(elsif_then);
		ctx->callbacks.compile_stmts(ctx, elsif_bodies[i]);
		ctx->builder->CreateBr(merge_bb);

		ctx->builder->SetInsertPoint(elsif_else);
	}

	/* ELSE body or fall through */
	if (else_stmts != NULL)
		ctx->callbacks.compile_stmts(ctx, else_stmts);

	ctx->builder->CreateBr(merge_bb);

	/* Continue after merge */
	ctx->builder->SetInsertPoint(merge_bb);
}

/*
 * Compile a conditional loop.
 *
 * test_at_top = true:  WHILE loop (condition checked before each iteration)
 * test_at_top = false: REPEAT UNTIL (condition checked after each iteration)
 */
void
upl_emit_cond_loop(UPL_compile_ctx *ctx, const char *label,
				   void *cond_expr, bool test_at_top,
				   void *body_stmts)
{
	llvm::BasicBlock   *cond_bb;
	llvm::BasicBlock   *body_bb;
	llvm::BasicBlock   *exit_bb;
	llvm::Value		   *cond;

	cond_bb = upl_append_block(ctx, test_at_top ? "while.cond" : "repeat.cond");
	body_bb = upl_append_block(ctx, test_at_top ? "while.body" : "repeat.body");
	exit_bb = upl_append_block(ctx, test_at_top ? "while.exit" : "repeat.exit");

	if (test_at_top)
	{
		/* WHILE: branch to condition first */
		ctx->builder->CreateBr(cond_bb);

		/* Condition block */
		ctx->builder->SetInsertPoint(cond_bb);
		cond = upl_eval_bool(ctx, cond_expr);
		ctx->builder->CreateCondBr(cond, body_bb, exit_bb);

		/* Body block */
		ctx->builder->SetInsertPoint(body_bb);

		upl_push_loop(ctx, label, cond_bb, exit_bb);
		ctx->callbacks.compile_stmts(ctx, body_stmts);
		upl_pop_loop(ctx);

		/* Loop back to condition */
		ctx->builder->CreateBr(cond_bb);
	}
	else
	{
		/* REPEAT UNTIL: branch to body first */
		ctx->builder->CreateBr(body_bb);

		/* Body block */
		ctx->builder->SetInsertPoint(body_bb);

		upl_push_loop(ctx, label, cond_bb, exit_bb);
		ctx->callbacks.compile_stmts(ctx, body_stmts);
		upl_pop_loop(ctx);

		/* Fall through to condition */
		ctx->builder->CreateBr(cond_bb);

		/* Condition block: exit if condition is true (UNTIL semantics) */
		ctx->builder->SetInsertPoint(cond_bb);
		cond = upl_eval_bool(ctx, cond_expr);
		ctx->builder->CreateCondBr(cond, exit_bb, body_bb);
	}

	/* Continue after loop */
	ctx->builder->SetInsertPoint(exit_bb);
}

/*
 * Compile an unconditional LOOP ... END LOOP.
 *
 * The loop body executes indefinitely until an EXIT statement branches
 * to exit_bb.
 */
void
upl_emit_loop(UPL_compile_ctx *ctx, const char *label,
			  void *body_stmts)
{
	llvm::BasicBlock   *body_bb;
	llvm::BasicBlock   *exit_bb;

	body_bb = upl_append_block(ctx, "loop.body");
	exit_bb = upl_append_block(ctx, "loop.exit");

	/* Branch to body */
	ctx->builder->CreateBr(body_bb);

	/* Body block */
	ctx->builder->SetInsertPoint(body_bb);

	upl_push_loop(ctx, label, body_bb, exit_bb);
	ctx->callbacks.compile_stmts(ctx, body_stmts);
	upl_pop_loop(ctx);

	/* Loop back */
	ctx->builder->CreateBr(body_bb);

	/* Continue after loop */
	ctx->builder->SetInsertPoint(exit_bb);
}

/*
 * Compile an integer FOR loop.
 *
 * Evaluates lower/upper/step bounds once via runtime helpers, then the
 * actual loop (compare, branch, increment) is pure LLVM IR.
 *
 * Structure:
 *   fori.setup:  evaluate bounds, initialize loop_val alloca
 *   fori.cond:   compare loop_val vs upper -> branch
 *   fori.body:   set found, assign loop var, compile body
 *   fori.step:   increment/decrement with overflow check -> branch to cond
 *   fori.exit:   set FOUND, continue
 */
void
upl_emit_fori(UPL_compile_ctx *ctx, const char *label,
			  int var_dno, void *lower_expr, void *upper_expr,
			  void *step_expr, bool reverse,
			  void *body_stmts)
{
	llvm::Value		   *lower, *upper, *step;
	llvm::Value		   *loop_val_ptr, *found_ptr;
	llvm::Value		   *cur_val, *next_val, *done, *overflow, *found_val;
	llvm::BasicBlock   *cond_bb, *body_bb, *step_bb, *exit_bb, *store_bb;

	/* Evaluate lower bound */
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_ptr(ctx, lower_expr)
		};
		lower = upl_emit_rt_call(ctx, ctx->callbacks.rt_eval_int, args, 2);
	}

	/* Evaluate upper bound */
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_ptr(ctx, upper_expr)
		};
		upper = upl_emit_rt_call(ctx, ctx->callbacks.rt_eval_int, args, 2);
	}

	/* Evaluate step (default 1) */
	if (step_expr)
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_ptr(ctx, step_expr)
		};
		step = upl_emit_rt_call(ctx, ctx->callbacks.rt_eval_int, args, 2);
	}
	else
		step = upl_const_int32(ctx, 1);

	/*
	 * Allocate loop value and found flag in the entry block.
	 *
	 * An alloca emitted at the current position lands inside whatever block we
	 * happen to be in — for a FOR nested in a high-iteration loop that is a
	 * per-iteration block, and the stack is not reclaimed until the function
	 * returns.  mem2reg promotes these scalars away at the -O3 we run, so it
	 * is latent today, but the entry block is where they belong (PG's own
	 * llvmjit does the same).
	 */
	{
		llvm::BasicBlock   *saved_bb = ctx->builder->GetInsertBlock();

		/* Position before the entry block's first instruction (or at end) */
		ctx->builder->SetInsertPoint(ctx->entry_bb, ctx->entry_bb->begin());

		loop_val_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT32],
												  nullptr, "loop_val");
		found_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT1],
											   nullptr, "found");

		ctx->builder->SetInsertPoint(saved_bb);
	}

	/* Initialise them where the loop actually starts */
	ctx->builder->CreateStore(lower, loop_val_ptr);
	ctx->builder->CreateStore(
		llvm::ConstantInt::get(ctx->types[UPL_INT1], 0, false), found_ptr);

	cond_bb  = upl_append_block(ctx, "fori.cond");
	body_bb  = upl_append_block(ctx, "fori.body");
	step_bb  = upl_append_block(ctx, "fori.step");
	exit_bb  = upl_append_block(ctx, "fori.exit");

	ctx->builder->CreateBr(cond_bb);

	/* --- Condition block: check loop_val vs upper --- */
	ctx->builder->SetInsertPoint(cond_bb);
	cur_val = ctx->builder->CreateLoad(ctx->types[UPL_INT32],
									   loop_val_ptr, "cur");

	if (reverse)
		done = ctx->builder->CreateICmpSLT(cur_val, upper, "done");
	else
		done = ctx->builder->CreateICmpSGT(cur_val, upper, "done");

	ctx->builder->CreateCondBr(done, exit_bb, body_bb);

	/* --- Body block --- */
	ctx->builder->SetInsertPoint(body_bb);

	/* Set found = true */
	ctx->builder->CreateStore(
		llvm::ConstantInt::get(ctx->types[UPL_INT1], 1, false), found_ptr);

	/* Assign current value to loop variable via runtime helper */
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_int32(ctx, var_dno),
			cur_val
		};
		upl_emit_rt_call(ctx, ctx->callbacks.rt_assign_int, args, 3);
	}

	/* Push loop: CONTINUE -> step_bb, EXIT -> exit_bb */
	upl_push_loop(ctx, label, step_bb, exit_bb);

	/* Compile loop body */
	ctx->callbacks.compile_stmts(ctx, body_stmts);

	upl_pop_loop(ctx);

	/* Fall through to step */
	ctx->builder->CreateBr(step_bb);

	/* --- Step block: increment/decrement with overflow check --- */
	ctx->builder->SetInsertPoint(step_bb);
	cur_val = ctx->builder->CreateLoad(ctx->types[UPL_INT32],
									   loop_val_ptr, "cur2");

	store_bb = upl_append_block(ctx, "fori.store");

	if (reverse)
	{
		next_val = ctx->builder->CreateSub(cur_val, step, "next");
		/* Overflow if next > cur (underflow in reverse) */
		overflow = ctx->builder->CreateICmpSGT(next_val, cur_val, "overflow");
	}
	else
	{
		next_val = ctx->builder->CreateAdd(cur_val, step, "next");
		/* Overflow if next < cur */
		overflow = ctx->builder->CreateICmpSLT(next_val, cur_val, "overflow");
	}

	ctx->builder->CreateCondBr(overflow, exit_bb, store_bb);

	/* Store block: save next value, loop back */
	ctx->builder->SetInsertPoint(store_bb);
	ctx->builder->CreateStore(next_val, loop_val_ptr);
	ctx->builder->CreateBr(cond_bb);

	/* --- Exit block: set FOUND and continue --- */
	ctx->builder->SetInsertPoint(exit_bb);
	found_val = ctx->builder->CreateLoad(ctx->types[UPL_INT1],
										 found_ptr, "found_val");
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			found_val
		};
		upl_emit_rt_call(ctx, ctx->callbacks.rt_set_found, args, 2);
	}
}

/*
 * Compile an EXIT/CONTINUE (or LEAVE/ITERATE) statement.
 *
 * is_exit = true:  EXIT (branch to exit_bb)
 * is_exit = false: CONTINUE (branch to continue_bb)
 *
 * If cond_expr is non-NULL, the branch is conditional.
 */
void
upl_emit_loop_exit(UPL_compile_ctx *ctx, const char *label,
				   bool is_exit, void *cond_expr)
{
	UPL_loop_info	   *loop;
	llvm::BasicBlock   *target_bb;

	/* Find the target loop */
	loop = upl_find_loop(ctx, label);
	if (loop == NULL)
		elog(ERROR, "upl: EXIT/CONTINUE outside of a loop");

	/*
	 * The parser already rejects CONTINUE naming a block label, so reaching
	 * here with a non-loop target means the two got out of step.
	 */
	if (!is_exit && !loop->is_loop)
		elog(ERROR, "upl: block label \"%s\" cannot be used in CONTINUE",
			 label ? label : "");

	target_bb = is_exit ? loop->exit_bb : loop->continue_bb;

	/*
	 * If the jump crosses one or more held resources — the statement sits
	 * inside a BEGIN...EXCEPTION block or a FOR-over-rows loop (or several)
	 * that the target loop or block encloses — the branch must not skip
	 * their cleanup.  A crossed exception frame holds an open subtransaction
	 * and has PG_exception_stack pointing at itself; branching past its
	 * release leaves both in place, and the next BEGIN then reports "there
	 * is already a transaction in progress" while a later error longjmps
	 * into a block the function has left.  A crossed row loop holds a pinned
	 * portal, and the surrounding transaction cannot commit until it is
	 * closed ("cannot commit while a portal is pinned").
	 *
	 * Emit the release helper for every cleanup above the target's recorded
	 * depth before taking the branch.  A target inside the same constructs
	 * as the statement (loop->cleanup_depth == cleanup_depth) unwinds
	 * nothing, and a construct's *own* entry needs nothing here either: an
	 * exception block's label and a row loop's loop entry are pushed inside
	 * their cleanup, and their exit_bb is the construct's cleanup path,
	 * which releases the resource itself.
	 */

	if (cond_expr)
	{
		/* Conditional EXIT/CONTINUE */
		llvm::BasicBlock   *skip_bb;
		llvm::Value		   *cond;

		skip_bb = upl_append_block(ctx, is_exit ? "exit.skip" : "continue.skip");

		cond = upl_eval_bool(ctx, cond_expr);

		if (loop->cleanup_depth < (int) ctx->cleanup_stack.size())
		{
			/*
			 * The cleanups must only run when the branch is taken, so the
			 * release calls need a block of their own on the taken edge.
			 */
			llvm::BasicBlock *unwind_bb;

			unwind_bb = upl_append_block(ctx,
										 is_exit ? "exit.unwind"
												 : "continue.unwind");
			ctx->builder->CreateCondBr(cond, unwind_bb, skip_bb);

			ctx->builder->SetInsertPoint(unwind_bb);
			upl_emit_cleanup_unwind(ctx, loop->cleanup_depth);
			ctx->builder->CreateBr(target_bb);
		}
		else
			ctx->builder->CreateCondBr(cond, target_bb, skip_bb);

		/* Continue compilation after the skip */
		ctx->builder->SetInsertPoint(skip_bb);
	}
	else
	{
		/* Unconditional EXIT/CONTINUE */
		upl_emit_cleanup_unwind(ctx, loop->cleanup_depth);
		ctx->builder->CreateBr(target_bb);

		/* Dead block for any statements after unconditional EXIT/CONTINUE */
		{
			llvm::BasicBlock *dead_bb = upl_append_block(ctx, "exit.dead");

			ctx->builder->SetInsertPoint(dead_bb);
		}
	}
}

/*
 * Compile a CASE statement (searched or simple).
 *
 * has_test_expr:     true for simple CASE (test expression assigned to temp var)
 * test_varno:        datum number of the temporary variable for simple CASE
 * test_assign_expr:  the test expression to assign (opaque)
 * num_whens:         number of WHEN clauses
 * when_conds:        array of WHEN condition expressions
 * when_bodies:       array of WHEN statement bodies
 * has_else:          true if there is an ELSE clause
 * else_body:         ELSE body statements (opaque)
 * lineno:            line number for error reporting (CASE without ELSE + no match)
 */
void
upl_emit_case(UPL_compile_ctx *ctx,
			  bool has_test_expr, int test_varno,
			  void *test_assign_expr,
			  int num_whens,
			  void **when_conds,
			  void **when_bodies,
			  bool has_else, void *else_body,
			  int lineno)
{
	llvm::BasicBlock   *merge_bb;
	int					i;

	merge_bb = upl_append_block(ctx, "case.merge");

	/*
	 * If simple CASE, evaluate the test expression and assign to the
	 * temporary variable.  The WHEN conditions reference this variable.
	 */
	if (has_test_expr)
		ctx->callbacks.assign_expr(ctx, test_varno, test_assign_expr);

	/* Evaluate each WHEN clause as a conditional branch chain */
	for (i = 0; i < num_whens; i++)
	{
		llvm::BasicBlock   *when_then_bb;
		llvm::BasicBlock   *when_else_bb;
		llvm::Value		   *cond;

		/*
		 * A simple CASE's WHEN conditions read the test temporary, whose type
		 * the driver settles at run time — so they must not be planned now.
		 * Scoped to the condition: the WHEN bodies below are ordinary
		 * statements and compile normally.
		 */
		ctx->defer_cond_plan = has_test_expr;
		cond = upl_eval_bool(ctx, when_conds[i]);
		ctx->defer_cond_plan = false;

		when_then_bb = upl_append_block(ctx, "case.when.then");
		when_else_bb = upl_append_block(ctx, "case.when.else");

		ctx->builder->CreateCondBr(cond, when_then_bb, when_else_bb);

		/* WHEN body */
		ctx->builder->SetInsertPoint(when_then_bb);

		/* Clear temp variable before executing body */
		if (has_test_expr)
		{
			llvm::Value *args[] = {
				ctx->estate_ref,
				upl_const_int32(ctx, test_varno)
			};
			upl_emit_rt_call(ctx, ctx->callbacks.rt_assign_null, args, 2);
		}

		ctx->callbacks.compile_stmts(ctx, when_bodies[i]);
		ctx->builder->CreateBr(merge_bb);

		/* Continue to next WHEN */
		ctx->builder->SetInsertPoint(when_else_bb);
	}

	/* Clear temp variable in the fallthrough path too */
	if (has_test_expr)
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_int32(ctx, test_varno)
		};
		upl_emit_rt_call(ctx, ctx->callbacks.rt_assign_null, args, 2);
	}

	/* ELSE clause or error */
	if (has_else)
	{
		ctx->callbacks.compile_stmts(ctx, else_body);
	}
	else
	{
		/* SQL2003: CASE without ELSE and no match is an error */
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_int32(ctx, lineno)
		};
		upl_emit_rt_call(ctx, ctx->callbacks.rt_case_error, args, 2);
	}

	ctx->builder->CreateBr(merge_bb);

	/* Continue after CASE */
	ctx->builder->SetInsertPoint(merge_bb);
}

/*
 * Compile a RETURN statement.
 *
 * Calls the runtime function at rt_exec_return with (estate, stmt).
 * Stores the result in rc_ptr, branches to return_bb, and creates a
 * dead block for any subsequent statements.
 */
void
upl_emit_return(UPL_compile_ctx *ctx, int rt_exec_return, void *stmt)
{
	llvm::Value *rc;

	/* Call rt_exec_return(estate, stmt) */
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_ptr(ctx, stmt)
		};

		rc = upl_emit_rt_call(ctx, rt_exec_return, args, 2);
	}

	/* Store the return code */
	ctx->builder->CreateStore(rc, ctx->rc_ptr);

	/* Branch to return block */
	ctx->builder->CreateBr(ctx->return_bb);

	/*
	 * Create a dead block for any statements after RETURN.
	 * LLVM requires the builder to be positioned at a valid block.
	 */
	{
		llvm::BasicBlock *dead_bb = upl_append_block(ctx, "dead");

		ctx->builder->SetInsertPoint(dead_bb);
	}
}

/*
 * Compile a BLOCK statement with variable initialization and optional
 * exception handling.
 *
 * If has_exceptions is true, emits the full sigsetjmp pattern:
 *   - Push exception frame (allocates frame, begins subtransaction)
 *   - sigsetjmp(frame, 0) — branch try/catch
 *   - Arm exception frame (set PG_exception_stack)
 *   - Init vars, compile body (with return_bb redirected)
 *   - Try exit on normal completion
 *   - Catch: dispatch to handler via compile_exceptions callback
 *   - Rethrow for unmatched exceptions
 *
 * The compile_exceptions callback is driver-specific because exception
 * list structures vary by language.
 *
 * The rt_* indices used here come from ctx->callbacks:
 *   rt_init_var: void fn(ptr estate, i32 dno)
 *
 * Exception-related RT functions are driver-specific and accessed through
 * the compile_exceptions callback.  The core block primitive only handles
 * the sigsetjmp/branch structure and delegates handler dispatch to the
 * driver.
 *
 * If has_exceptions is false, simply initializes variables and compiles
 * the body.
 */
void
upl_emit_block(UPL_compile_ctx *ctx,
			   int n_initvars, int *initvarnos,
			   void *body_stmts,
			   bool has_exceptions, void *exception_data,
			   void (*compile_exceptions)(UPL_compile_ctx *ctx,
										  void *exception_data))
{
	int i;

	if (has_exceptions)
	{
		ctx->has_exceptions = true;

		/*
		 * Delegate the entire exception block to the driver callback.
		 * The driver handles the complete sigsetjmp pattern, variable
		 * init, body compilation, handler dispatch, and return_bb
		 * redirection.  Exception list structures and subtransaction
		 * management vary by language.
		 */
		compile_exceptions(ctx, exception_data);
		return;
	}

	/* No exceptions: initialize variables and compile body */
	for (i = 0; i < n_initvars; i++)
	{
		llvm::Value *args[] = {
			ctx->estate_ref,
			upl_const_int32(ctx, initvarnos[i])
		};
		upl_emit_rt_call(ctx, ctx->callbacks.rt_init_var, args, 2);
	}

	/* Compile body statements */
	ctx->callbacks.compile_stmts(ctx, body_stmts);
}


/* ----------------------------------------------------------------
 *		Compilation pipeline
 * ----------------------------------------------------------------
 */

/*
 * upl_compile_function - Orchestrate the full compilation pipeline.
 *
 * This is the main entry point called by language drivers.  It:
 *   1. Creates LLVM context/module/builder
 *   2. Registers types
 *   3. Calls hooks->register_rt_funcs() for runtime function declarations
 *   4. Creates the LLVM function with UPL_FUNC_TYPE signature
 *   5. Registers sigsetjmp (returns_twice attribute)
 *   6. Creates entry/return blocks, rc alloca, estate param
 *   7. Calls hooks->setup_entry() for driver-specific setup
 *   8. Calls hooks->compile_body() for AST compilation
 *   9. Falls through to return block
 *  10. Adds nounwind if no exceptions
 *  11. Verifies, optimizes (O3), JIT compiles via OrcJIT
 *
 * Returns the native function pointer (void*).
 *
 * Steps that can raise Postgres errors are ffi_guard'ed, so a compilation
 * failure unwinds as a C++ exception and the caller's compile context
 * releases the LLVM resources through its destructors.
 */
void *
upl_compile_function(UPL_compile_ctx *ctx, UPL_compile_hooks *hooks)
{
	char			func_name[NAMEDATALEN + 32];
	llvm::Value	   *estate_ref;
	llvm::Value	   *rc_val;
	void		   *fn_ptr = NULL;

	/*
	 * Generate a unique function name for LLVM.  Include a generation
	 * counter so that recompiled versions (after CREATE OR REPLACE) don't
	 * collide with the old symbol still present in OrcJIT.
	 */
	snprintf(func_name, sizeof(func_name), "%s_%u_g" UINT64_FORMAT,
			 hooks->func_name_prefix, hooks->fn_oid, compile_gen++);

	/* 1. Create LLVM context, module, builder */
	ctx->context = std::make_unique<llvm::LLVMContext>();
	ctx->module = std::make_unique<llvm::Module>(func_name, *ctx->context);
	ctx->builder = std::make_unique<llvm::IRBuilder<>>(*ctx->context);

	/*
	 * Steps that can raise a Postgres error run through cppgres::ffi_guard,
	 * so a failure surfaces as a C++ exception here and the caller's
	 * compile context unwinds normally, releasing the LLVM resources —
	 * no PG_TRY bookkeeping (see doc/cpp-rewrite.md).
	 */

	/* 2. Register types (core engine) */
	upl_register_types(ctx);

	/* 3. Register runtime function declarations (driver) */
	hooks->register_rt_funcs(ctx);

	/* 4. Create the LLVM function: int32 func(ptr estate) */
	ctx->function = llvm::Function::Create(
		llvm::cast<llvm::FunctionType>(ctx->types[UPL_FUNC_TYPE]),
		llvm::Function::ExternalLinkage, func_name, ctx->module.get());

	/*
	 * 5. Register sigsetjmp as an external function with returns_twice
	 * attribute.  Signature: int sigsetjmp(ptr jmpbuf, int savesigs)
	 */
	{
		llvm::Type *sjparams[] = { ctx->types[UPL_PTR],
								   ctx->types[UPL_INT32] };
		llvm::FunctionType *sjft =
			llvm::FunctionType::get(ctx->types[UPL_INT32],
									sjparams, false);
		llvm::Function *sjfn;

		ctx->sigsetjmp_fntype = sjft;
		sjfn = llvm::Function::Create(sjft,
									  llvm::Function::ExternalLinkage,
									  UPL_SIGSETJMP_SYM,
									  ctx->module.get());
		ctx->sigsetjmp_fn = sjfn;

		/* Mark as returns_twice -- critical for correct codegen */
		sjfn->addFnAttr(llvm::Attribute::ReturnsTwice);
	}

	/* 6. Create entry and return blocks */
	ctx->entry_bb = upl_append_block(ctx, "entry");
	ctx->return_bb = upl_append_block(ctx, "return");

	ctx->builder->SetInsertPoint(ctx->entry_bb);

	/* 7. Allocate return code storage, store default */
	ctx->rc_ptr = ctx->builder->CreateAlloca(ctx->types[UPL_INT32],
											 nullptr, "rc");
	ctx->builder->CreateStore(upl_const_int32(ctx, hooks->default_rc),
							  ctx->rc_ptr);

	/* 8. Get estate parameter (first arg) */
	estate_ref = ctx->function->getArg(0);
	estate_ref->setName("estate");
	ctx->estate_ref = estate_ref;

	/*
	 * 9./10. Driver-specific entry setup, then the function body.  These
	 * recurse into the driver's statement and expression compilers, which
	 * raise Postgres errors for unsupported constructs — guard them so the
	 * error unwinds as a C++ exception.
	 */
	cppgres::ffi_guard{hooks->setup_entry}(ctx);
	cppgres::ffi_guard{hooks->compile_body}(ctx);

	/* 11. Fall through to return block */
	ctx->builder->CreateBr(ctx->return_bb);

	/* 12. Return block: load rc and return */
	ctx->builder->SetInsertPoint(ctx->return_bb);
	rc_val = ctx->builder->CreateLoad(ctx->types[UPL_INT32],
									  ctx->rc_ptr, "rc_val");
	ctx->builder->CreateRet(rc_val);

	/*
	 * 13. Add nounwind only if no exception blocks were compiled.
	 * Functions that call sigsetjmp (returns_twice) must NOT be
	 * nounwind, or LLVM may misoptimize around the setjmp point.
	 */
	if (!ctx->has_exceptions)
		ctx->function->addFnAttr(llvm::Attribute::NoUnwind);

	/* 14. Verify the module (guarded: verification failure is elog(ERROR)) */
	cppgres::ffi_guard{upl_verify_module}(*ctx->module);

	/* 14a. Optionally dump the IR (uplpgsql.dump_ir). */
	if (hooks->dump_ir)
	{
		std::string irstr;
		llvm::raw_string_ostream os(irstr);

		ctx->module->print(os, nullptr);
		elog(LOG, "upl: IR for %s:\n%s", func_name,
			 pstrdup(os.str().c_str()));
	}

	/* 15. Optimize */
	upl_optimize_module(*ctx->module, 3);

	/*
	 * 16. Compile via OrcJIT (guarded: OrcJIT failures are elog(ERROR)).
	 *
	 * upl_jit_compile() takes ownership of the module and context by
	 * move.  The builder references the context, so destroy it first.
	 */
	ctx->builder.reset();
	fn_ptr = cppgres::ffi_guard{upl_jit_compile}(std::move(ctx->module),
												 std::move(ctx->context),
												 func_name);

	return fn_ptr;
}
