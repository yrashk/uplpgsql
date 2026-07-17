/*-------------------------------------------------------------------------
 *
 * upl_compiler.hpp
 *		The PL/pgSQL JIT compiler class.
 *
 *		uplpgsql::function_compiler compiles one PL/pgSQL function to
 *		native code.  One instance exists for the duration of a single
 *		compilation; it owns the core UPL_compile_ctx and the driver's
 *		per-compilation state (native array metadata, cached plstate ref).
 *
 *		The class spans two translation units:
 *		  - statement compilation lives in upl_compile_stmts.cpp
 *		  - expression compilation (the three-tier compiler) lives in
 *		    upl_compile_expr.cpp
 *
 *		The core engine (core/upl_compile.cpp) drives compilation through
 *		C function pointers (UPL_callbacks / UPL_compile_hooks).  Those
 *		slots are filled with static trampolines (cb_*) that recover the
 *		compiler instance from ctx->lang_data — which is set to `this` in
 *		the constructor — and delegate to the corresponding method.
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
#ifndef UPL_COMPILER_HPP
#define UPL_COMPILER_HPP

#include "upl_common.h"

namespace uplpgsql
{

/*
 * Max bytes per native array to allocate on stack via LLVM alloca.
 * Arrays larger than this threshold use palloc0 (heap) via runtime helper.
 * 4096 bytes = 512 float8s or 1024 int4s — fits comfortably in stack frame.
 */
#define NATIVE_ARRAY_STACK_THRESHOLD	4096

/*
 * Metadata for a local array variable lowered to flat native memory
 * by Phase 7 escape analysis.
 *
 * Instead of going through array_get_element/array_set_element per access
 * (~50-100ns each), subscript reads/writes are compiled as direct LLVM
 * GEP+load/store (~1ns).
 *
 * Created by function_compiler::analyze_native_arrays().  The
 * llvm_elemtype, data_ptr, and len_ptr fields are populated during entry
 * block alloca setup (function_compiler::setup_entry()).  The actual
 * memory allocation happens when array_fill() is intercepted during
 * expression compilation (upl_compile_expr.cpp).
 */
typedef struct UPLpgSQL_native_array
{
	int				dno;			/* datum number of the array variable */
	Oid				elemtype;		/* INT4OID, INT8OID, or FLOAT8OID */
	int				elem_size;		/* sizeof(element): 4 or 8 bytes */
	llvm::Type *		llvm_elemtype;	/* i32, i64, or double */
	llvm::Value *	data_ptr;		/* entry-block alloca holding ptr to flat memory */
	llvm::Value *	len_ptr;		/* entry-block alloca: element count, or -1 when
									 * the value is not a 1-D array and so cannot be
									 * held natively (see uplpgsql_rt_native_array_
									 * from_datum) */
	llvm::Value *	nulls_ptr;		/* entry-block alloca: ptr to a per-element
									 * bool array, or NULL when no element is
									 * NULL.  PostgreSQL fills the gap with NULLs
									 * when an assignment extends an array past
									 * its end, so the native form has to be able
									 * to say which elements are null. */
	llvm::Value *	lb_ptr;			/* entry-block alloca: the array's lower bound.
									 * PostgreSQL arrays need not start at 1
									 * ('[2:3]={9,10}'), so subscripts are relative
									 * to this, not to 1. */
	llvm::Value *	cap_ptr;		/* entry-block alloca: allocated element slots
									 * in data.  An append (a write at exactly
									 * lb+len) bumps len up to this without any
									 * reallocation; past it the buffers grow
									 * through uplpgsql_rt_native_array_reserve,
									 * which doubles, so filling an array element
									 * by element is amortized O(1) per write. */
	llvm::Value *	is_heap_ptr;	/* entry-block alloca (i8): 1 when data was
									 * palloc'd and may be repalloc'd/pfree'd, 0
									 * for the array_fill stack buffer, which can
									 * only be copied out of. */
} UPLpgSQL_native_array;

/*
 * Type classification for the expression compiler.
 *
 * Used by classify_expr() to recursively determine whether an entire
 * expression tree can be compiled to Tier 1 native instructions.  Only
 * expressions where ALL nodes classify to a known type (not UNKNOWN) are
 * eligible for Tier 1 compilation.  UNKNOWN falls through to Tier 2 or
 * Tier 3.
 */
typedef enum ExprTypeClass
{
	EXPR_TYPE_INT4,
	EXPR_TYPE_INT8,
	EXPR_TYPE_FLOAT8,
	EXPR_TYPE_BOOL,
	EXPR_TYPE_UNKNOWN		/* not natively inlineable */
} ExprTypeClass;

/*
 * Array element type info resolved at compile time as LLVM constants.
 * Avoids repeated get_element_type()/get_typlenbyvalalign() at runtime.
 */
typedef struct ArrayTypeInfo
{
	llvm::Value	*typlen_val;		/* i32: array type length (-1 for varlena) */
	llvm::Value	*elemtype_val;	/* i32: element type OID */
	llvm::Value	*elmlen_val;		/* i32: element length (widened, see resolve_array_type_info) */
	llvm::Value	*elmbyval_val;	/* i1: element pass-by-value */
	llvm::Value	*elmalign_val;	/* i8: element alignment */
} ArrayTypeInfo;

/*
 * Compiles one PL/pgSQL function to native code.  Statement methods live in
 * upl_compile_stmts.cpp, expression methods in upl_compile_expr.cpp.
 */
class function_compiler
{
public:
	explicit function_compiler(UPLpgSQL_function *func);
	UPLpgSQL_func *compile();		/* runs the core pipeline */

private:
	UPL_compile_ctx		ctx_;

	/*
	 * The methods below were free functions threading a UPL_compile_ctx *ctx
	 * parameter; their bodies are unchanged and keep referring to `ctx`,
	 * which now resolves to this pointer at the owned context.
	 */
	UPL_compile_ctx *const ctx = &ctx_;

	/* The PL/pgSQL function being compiled */
	UPLpgSQL_function  *func_;

	/* Cached plstate pointer (estate->uplpgsql_estate), loaded at entry */
	llvm::Value		   *plstate_ref_ = nullptr;

	/* Native local arrays identified by escape analysis (Phase 7) */
	std::vector<UPLpgSQL_native_array> native_arrays_;

	/*
	 * Core callback trampolines (core takes C function pointers) — one per
	 * UPL_callbacks/UPL_compile_hooks slot the driver fills.  Each recovers
	 * `this` from ctx->lang_data and delegates to the method of the same
	 * name.
	 */
	static function_compiler *self(UPL_compile_ctx *ctx)
	{
		return static_cast<function_compiler *>(ctx->lang_data);
	}

	static void cb_compile_stmts(UPL_compile_ctx *ctx, void *stmts);
	static bool cb_try_compile_bool(UPL_compile_ctx *ctx, void *expr,
									llvm::Value **result_out);
	static void cb_assign_expr(UPL_compile_ctx *ctx, int varno, void *expr);
	static void cb_register_rt_funcs(UPL_compile_ctx *ctx);
	static void cb_setup_entry(UPL_compile_ctx *ctx);
	static void cb_compile_body(UPL_compile_ctx *ctx);
	static void cb_compile_block_exceptions(UPL_compile_ctx *ctx,
											void *exception_data);

	/* ---- statement compilation (upl_compile_stmts.cpp) ---- */

	/* pipeline hooks */
	void		register_runtime_funcs();
	void		setup_entry();
	void		compile_body();

	/* IR emission helpers over the runtime function table */
	llvm::Value *call_fn(UPLpgSQL_rt_func which,
						 llvm::ArrayRef<llvm::Value *> args);
	llvm::Value *call_exec_nosync(void *fn_addr, llvm::Type *ret_type,
								  llvm::ArrayRef<llvm::Value *> args);
	llvm::Value *call_exec(void *fn_addr, llvm::Type *ret_type,
						   llvm::ArrayRef<llvm::Value *> args);

	/* statement dispatch */
	void		compile_stmts(List *stmts);
	void		compile_stmt(UPLpgSQL_stmt *stmt);

	/* per-statement compilers */
	void		compile_block(UPLpgSQL_stmt_block *stmt);
	void		compile_block_exceptions(void *exception_data);
	void		compile_return(UPLpgSQL_stmt_return *stmt);
	void		compile_assign(UPLpgSQL_stmt_assign *stmt);
	void		compile_if(UPLpgSQL_stmt_if *stmt);
	void		compile_while(UPLpgSQL_stmt_while *stmt);
	void		compile_loop(UPLpgSQL_stmt_loop *stmt);
	void		compile_fori(UPLpgSQL_stmt_fori *stmt);
	void		compile_exit(UPLpgSQL_stmt_exit *stmt);
	void		compile_perform(UPLpgSQL_stmt_perform *stmt);
	void		compile_execsql(UPLpgSQL_stmt_execsql *stmt);
	void		compile_raise(UPLpgSQL_stmt_raise *stmt);
	void		compile_case(UPLpgSQL_stmt_case *stmt);
	void		compile_assert(UPLpgSQL_stmt_assert *stmt);
	void		compile_open(UPLpgSQL_stmt_open *stmt);
	void		compile_fetch(UPLpgSQL_stmt_fetch *stmt);
	void		compile_close(UPLpgSQL_stmt_close *stmt);
	void		compile_fors(UPLpgSQL_stmt_fors *stmt);
	void		compile_forc(UPLpgSQL_stmt_forc *stmt);
	void		compile_dynexecute(UPLpgSQL_stmt_dynexecute *stmt);
	void		compile_dynfors(UPLpgSQL_stmt_dynfors *stmt);
	void		compile_foreach_a(UPLpgSQL_stmt_foreach_a *stmt);
	void		compile_return_next(UPLpgSQL_stmt_return_next *stmt);
	void		compile_return_query(UPLpgSQL_stmt_return_query *stmt);
	void		compile_call(UPLpgSQL_stmt_call *stmt);
	void		compile_getdiag(UPLpgSQL_stmt_getdiag *stmt);
	void		compile_commit(UPLpgSQL_stmt_commit *stmt);
	void		compile_rollback(UPLpgSQL_stmt_rollback *stmt);

	/* block variable initialization */
	void		emit_init_vars(llvm::ArrayRef<int> initvarnos);

	/* CASE test expression assignment (see cb_assign_expr) */
	void		assign_expr(int varno, void *expr);

	/* native local array analysis / marshalling */
	void		analyze_native_arrays(UPLpgSQL_function *func);
	/* returned pointers stay valid: never resized after analyze_native_arrays */
	UPLpgSQL_native_array *find_native_array(int dno);
	void		emit_sync_native_array(UPLpgSQL_native_array *na);
	void		sync_native_arrays();
	void		sync_native_arrays_for_expr(UPLpgSQL_expr *expr);
	void		emit_refresh_native_array(UPLpgSQL_native_array *na);

	/* ---- expression compilation (upl_compile_expr.cpp) ---- */

	/* three-tier entry points */
	bool		try_compile_assign(UPLpgSQL_stmt_assign *stmt);
	llvm::Value *try_compile_bool(UPLpgSQL_expr *expr_node);

	/* compile-time SPI prepare + simple expression extraction */
	SPIPlanPtr	prepare_plan_compile_time(UPLpgSQL_expr *expr);
	Expr	   *prepare_and_get_expr(UPLpgSQL_expr *expr);

	/* Tier 1: native LLVM instructions */
	llvm::Value *compile_expr_datum(Expr *expr, llvm::Value *estate_ref,
									ExprTypeClass *result_type);
	llvm::Value *compile_expr_bool(Expr *expr, llvm::Value *estate_ref);
	llvm::Value *tier1_expr_any_null(Expr *expr, llvm::Value *estate_ref);
	llvm::Value *tier1_compile_value_guarded(Expr *val_expr,
											 llvm::Value *estate_ref,
											 ExprTypeClass *val_class,
											 llvm::Value **isnull_out);

	/* Tier 2: fmgr bypass */
	llvm::Value *compile_expr_fmgr(Expr *expr, llvm::Value *estate_ref);
	llvm::Value *compile_expr_fmgr_full(Expr *expr, llvm::Value *estate_ref,
										llvm::Value **isnull_out);
	bool		fmgr_expr_reads_native_array(Expr *expr);
	bool		fmgr_expr_wants_alloc_scope(Expr *expr);
	llvm::Value *fmgr_load_arg_datum(Expr *expr, llvm::Value *estate_ref);
	llvm::Value *fmgr_load_arg_isnull(Expr *expr, llvm::Value *estate_ref);

	/* variable access via GEP */
	llvm::Value *emit_load_var_datum(llvm::Value *estate_ref, int dno);
	llvm::Value *emit_load_param_datum(llvm::Value *estate_ref, int dno);
	llvm::Value *emit_load_param_isnull(llvm::Value *estate_ref, int dno);
	void		emit_store_var_datum(llvm::Value *estate_ref, int dno,
									 llvm::Value *datum_val);
	void		emit_store_var_datum_isnull(llvm::Value *estate_ref, int dno,
											llvm::Value *datum_val,
											llvm::Value *isnull_val);
	void		emit_store_var_null(llvm::Value *estate_ref, int dno);

	/* array subscript helpers */
	ArrayTypeInfo resolve_array_type_info(int array_dno);
	void		emit_set_subscript_null_check(Expr *idx_expr,
											  llvm::Value *estate_ref);
};

} /* namespace uplpgsql */

#endif							/* UPL_COMPILER_HPP */
