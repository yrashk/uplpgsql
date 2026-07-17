/*-------------------------------------------------------------------------
 *
 * uplpgsql_handler.cpp
 *		PostgreSQL language handler entry points for uplpgsql.
 *
 *		This file is the main entry point for all uplpgsql execution.
 *		It contains:
 *
 *		_PG_init()              — Module initialization: sets up LLVM, GUCs,
 *		                          and transaction callbacks.
 *
 *		uplpgsql_call_handler() — Main execution entry for functions and triggers.
 *		                          Compiles the function (get AST via forked parser),
 *		                          checks the JIT cache, optionally JIT-compiles,
 *		                          then executes via native code or interpreter.
 *
 *		uplpgsql_inline_handler() — Executes DO blocks (always interpreted since
 *		                            they run once and JIT overhead would dominate).
 *
 *		uplpgsql_validator()    — CREATE FUNCTION validation via test-compilation.
 *
 *		The call_handler implements the three-state cache flow:
 *		  1. Cache lookup: hit → use JIT'd code, skip → use interpreter
 *		  2. Cache miss → evaluate heuristic (uplpgsql_should_jit)
 *		  3. If should JIT → compile, cache result; else cache skip marker
 *		  4. JIT compilation failures are caught and fall back to interpreter
 *
 *		Error handling is exception-based throughout: every call into the
 *		engine runs under cppgres::ffi_guard, so Postgres errors surface as
 *		C++ exceptions and cleanup is plain RAII.  At each entry point's
 *		boundary a pg_exception is re-thrown to Postgres with full fidelity
 *		(SQLSTATE, detail, context — cppgres::pg_exception::rethrow), and
 *		any other C++ exception is reported as a Postgres error.
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
#include "upl_common.h"

extern "C" {

#include "access/xact.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "commands/event_trigger.h"
#include "commands/trigger.h"
#include "funcapi.h"
#include "executor/executor.h"
#include "nodes/parsenodes.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/plancache.h"
#include "utils/resowner.h"
#include "utils/syscache.h"
#include "utils/varlena.h"

}

#include "cppgres.hpp"

#include "upl_cache.hpp"

/* Extension version, as reported by PG_MODULE_MAGIC_EXT and \dx */
#define UPLPGSQL_VERSION	"1.0"

extern "C" {
PG_MODULE_MAGIC_EXT(
					.name = "uplpgsql",
					.version = UPLPGSQL_VERSION
);
}

/*
 * Forked PL/pgSQL global variables (originally in pl_handler.c).
 *
 * These mirror the original PL/pgSQL GUCs but under the "uplpgsql." prefix.
 * They are referenced by the forked parser/executor code (upl_comp.cpp,
 * upl_exec.cpp, etc.) which was mechanically renamed from plpgsql_* to
 * uplpgsql_*.
 */
static const struct config_enum_entry variable_conflict_options[] = {
	{"error", UPLPGSQL_RESOLVE_ERROR, false},
	{"use_variable", UPLPGSQL_RESOLVE_VARIABLE, false},
	{"use_column", UPLPGSQL_RESOLVE_COLUMN, false},
	{NULL, 0, false}
};

int			uplpgsql_variable_conflict = UPLPGSQL_RESOLVE_ERROR;
bool		uplpgsql_print_strict_params = false;
bool		uplpgsql_check_asserts = true;
static char *uplpgsql_extra_warnings_string = NULL;
static char *uplpgsql_extra_errors_string = NULL;
int			uplpgsql_extra_warnings;
int			uplpgsql_extra_errors;

/* Plugin hook pointer */
UPLpgSQL_plugin **uplpgsql_plugin_ptr = NULL;

/*
 * uplpgsql-specific GUCs (not in standard PL/pgSQL).
 *
 * enable_jit_heuristic: When true, use cost/benefit heuristic to skip JIT for
 *                       functions unlikely to benefit.  When false (default),
 *                       JIT-compile every function.
 * log_compilation:      Log a message when a function is JIT compiled
 * dump_ir:              Dump the generated LLVM IR to the server log (debugging)
 */
static bool uplpgsql_log_compilation = false;
bool		uplpgsql_dump_ir = false;
bool		uplpgsql_enable_jit_heuristic = false;

/* Forward declarations for GUC hooks */
static bool uplpgsql_extra_checks_check_hook(char **newvalue, void **extra,
											 GucSource source);
static void uplpgsql_extra_warnings_assign_hook(const char *newvalue, void *extra);
static void uplpgsql_extra_errors_assign_hook(const char *newvalue, void *extra);

/* Function declarations */
extern "C" {
PG_FUNCTION_INFO_V1(uplpgsql_call_handler);
PG_FUNCTION_INFO_V1(uplpgsql_inline_handler);
PG_FUNCTION_INFO_V1(uplpgsql_validator);
extern void _PG_init(void);
}

/*
 * _PG_init - module load callback
 */
void
_PG_init(void)
{
	static bool inited = false;

	if (inited)
		return;

	/* Initialize LLVM (core engine) */
	upl_llvm_init();

	/*
	 * Register forked PL/pgSQL GUCs under "uplpgsql." prefix
	 */
	DefineCustomEnumVariable("uplpgsql.variable_conflict",
							 "Sets handling of conflicts between UPL/pgSQL variable names and table column names.",
							 NULL,
							 &uplpgsql_variable_conflict,
							 UPLPGSQL_RESOLVE_ERROR,
							 variable_conflict_options,
							 PGC_SUSET, 0,
							 NULL, NULL, NULL);

	DefineCustomBoolVariable("uplpgsql.print_strict_params",
							 "Print information about parameters in the DETAIL part of the error messages generated on INTO ... STRICT failures.",
							 NULL,
							 &uplpgsql_print_strict_params,
							 false,
							 PGC_USERSET, 0,
							 NULL, NULL, NULL);

	DefineCustomBoolVariable("uplpgsql.check_asserts",
							 "Perform checks given in ASSERT statements.",
							 NULL,
							 &uplpgsql_check_asserts,
							 true,
							 PGC_USERSET, 0,
							 NULL, NULL, NULL);

	DefineCustomStringVariable("uplpgsql.extra_warnings",
							   "List of programming constructs that should produce a warning.",
							   NULL,
							   &uplpgsql_extra_warnings_string,
							   "none",
							   PGC_USERSET, GUC_LIST_INPUT,
							   uplpgsql_extra_checks_check_hook,
							   uplpgsql_extra_warnings_assign_hook,
							   NULL);

	DefineCustomStringVariable("uplpgsql.extra_errors",
							   "List of programming constructs that should produce an error.",
							   NULL,
							   &uplpgsql_extra_errors_string,
							   "none",
							   PGC_USERSET, GUC_LIST_INPUT,
							   uplpgsql_extra_checks_check_hook,
							   uplpgsql_extra_errors_assign_hook,
							   NULL);

	/* uplpgsql-specific GUCs */
	DefineCustomBoolVariable("uplpgsql.log_compilation",
							 "Log when functions are JIT compiled.",
							 NULL,
							 &uplpgsql_log_compilation,
							 false,
							 PGC_USERSET, 0,
							 NULL, NULL, NULL);

	DefineCustomBoolVariable("uplpgsql.dump_ir",
							 "Dump LLVM IR to server log.",
							 NULL,
							 &uplpgsql_dump_ir,
							 false,
							 PGC_USERSET, 0,
							 NULL, NULL, NULL);

	DefineCustomBoolVariable("uplpgsql.enable_jit_heuristic",
							 "Use cost/benefit heuristic to skip JIT for functions unlikely to benefit. "
							 "When off (default), all functions are JIT compiled.",
							 NULL,
							 &uplpgsql_enable_jit_heuristic,
							 false,
							 PGC_USERSET, 0,
							 NULL, NULL, NULL);

	/*
	 * Reserve the prefix only after every uplpgsql.* GUC is defined.  Doing it
	 * earlier discards the placeholders for the ones defined below, so a value
	 * SET before the module was first loaded in a session was thrown away with
	 * "invalid configuration parameter name".
	 */
	MarkGUCPrefixReserved("uplpgsql");

	/* Register transaction callbacks for cleanup */
	RegisterXactCallback(uplpgsql_xact_cb, NULL);
	RegisterSubXactCallback(uplpgsql_subxact_cb, NULL);

	/* Set up rendezvous point with optional instrumentation plugin */
	uplpgsql_plugin_ptr = (UPLpgSQL_plugin **)
		find_rendezvous_variable("UPLpgSQL_plugin");

	inited = true;
}

/*
 * GUC check/assign hooks for extra_warnings/extra_errors
 */
static bool
uplpgsql_extra_checks_check_hook(char **newvalue, void **extra, GucSource source)
{
	char	   *rawstring;
	List	   *elemlist;
	ListCell   *l;
	int			extrachecks = 0;
	int		   *myextra;

	if (pg_strcasecmp(*newvalue, "all") == 0)
		extrachecks = UPLPGSQL_XCHECK_ALL;
	else if (pg_strcasecmp(*newvalue, "none") == 0)
		extrachecks = UPLPGSQL_XCHECK_NONE;
	else
	{
		rawstring = pstrdup(*newvalue);
		if (!SplitIdentifierString(rawstring, ',', &elemlist))
		{
			GUC_check_errdetail("List syntax is invalid.");
			pfree(rawstring);
			list_free(elemlist);
			return false;
		}

		foreach(l, elemlist)
		{
			char	   *tok = (char *) lfirst(l);

			if (pg_strcasecmp(tok, "shadowed_variables") == 0)
				extrachecks |= UPLPGSQL_XCHECK_SHADOWVAR;
			else if (pg_strcasecmp(tok, "too_many_rows") == 0)
				extrachecks |= UPLPGSQL_XCHECK_TOOMANYROWS;
			else if (pg_strcasecmp(tok, "strict_multi_assignment") == 0)
				extrachecks |= UPLPGSQL_XCHECK_STRICTMULTIASSIGNMENT;
			else if (pg_strcasecmp(tok, "all") == 0 || pg_strcasecmp(tok, "none") == 0)
			{
				GUC_check_errdetail("Key word \"%s\" cannot be combined with other key words.", tok);
				pfree(rawstring);
				list_free(elemlist);
				return false;
			}
			else
			{
				GUC_check_errdetail("Unrecognized key word: \"%s\".", tok);
				pfree(rawstring);
				list_free(elemlist);
				return false;
			}
		}

		pfree(rawstring);
		list_free(elemlist);
	}

	myextra = (int *) guc_malloc(LOG, sizeof(int));
	if (!myextra)
		return false;
	*myextra = extrachecks;
	*extra = myextra;

	return true;
}

static void
uplpgsql_extra_warnings_assign_hook(const char *newvalue, void *extra)
{
	uplpgsql_extra_warnings = *((int *) extra);
}

static void
uplpgsql_extra_errors_assign_hook(const char *newvalue, void *extra)
{
	uplpgsql_extra_errors = *((int *) extra);
}

namespace uplpgsql {

/*
 * SPI session for a handler invocation.
 *
 * cppgres::spi_executor's flag-taking constructor is protected (the public
 * types decide atomicity from cppgres function-call context, which a raw
 * language handler does not have), so expose it here: the handler computes
 * the flags from fcinfo itself.
 */
struct spi_session : cppgres::spi_executor
{
	explicit spi_session(int flags) : cppgres::spi_executor(flags) {}
};

/*
 * Marks a compiled function busy for the duration of a call, saving and
 * restoring cur_estate — the RAII form of the old PG_FINALLY bookkeeping.
 */
struct function_busy_guard
{
	explicit function_busy_guard(UPLpgSQL_function *func)
		: func_(func), save_cur_estate_(func->cur_estate)
	{
		func_->cfunc.use_count++;
	}

	~function_busy_guard()
	{
		func_->cfunc.use_count--;
		func_->cur_estate = save_cur_estate_;
	}

	function_busy_guard(const function_busy_guard &) = delete;
	function_busy_guard &operator=(const function_busy_guard &) = delete;

private:
	UPLpgSQL_function *func_;
	UPLpgSQL_execstate *save_cur_estate_;
};

/*
 * Procedure-lifespan resource owner for CALL/DO statements, released on
 * every exit path.  Since this resowner is not tied to any parent, failing
 * to free it would result in process-lifespan leaks — which is exactly why
 * it lives in a guard.
 */
struct procedure_resowner_guard
{
	explicit procedure_resowner_guard(bool needed)
		: resowner_(needed
					? cppgres::ffi_guard{::ResourceOwnerCreate}(
						  nullptr, "UPL/pgSQL procedure resources")
					: nullptr)
	{
	}

	~procedure_resowner_guard()
	{
		if (resowner_ == nullptr)
			return;

		try
		{
			cppgres::ffi_guard{::ReleaseAllPlanCacheRefsInOwner}(resowner_);
			cppgres::ffi_guard{::ResourceOwnerDelete}(resowner_);
		}
		catch (...)
		{
			elog(WARNING, "uplpgsql: releasing procedure resource owner failed");
		}
	}

	procedure_resowner_guard(const procedure_resowner_guard &) = delete;
	procedure_resowner_guard &operator=(const procedure_resowner_guard &) = delete;

	operator ResourceOwner() const { return resowner_; }

private:
	ResourceOwner resowner_;
};

/*
 * Busy-marks a DO block's compiled function and frees its memory on every
 * exit path — a DO block's AST is single-use.
 */
struct inline_function_guard
{
	explicit inline_function_guard(UPLpgSQL_function *func) : func_(func)
	{
		func_->cfunc.use_count++;
	}

	~inline_function_guard()
	{
		func_->cfunc.use_count--;
		try
		{
			cppgres::ffi_guard{uplpgsql_free_function_memory}(func_);
		}
		catch (...)
		{
			elog(WARNING, "uplpgsql: freeing DO block function memory failed");
		}
	}

	inline_function_guard(const inline_function_guard &) = delete;
	inline_function_guard &operator=(const inline_function_guard &) = delete;

private:
	UPLpgSQL_function *func_;
};

/*
 * Private EState and resource owner for a DO block's simple-expression
 * execution, released on every exit path.
 */
struct simple_eval_resources
{
	simple_eval_resources()
		: estate_(cppgres::ffi_guard{::CreateExecutorState}()),
		  resowner_(cppgres::ffi_guard{::ResourceOwnerCreate}(
			  nullptr, "UPL/pgSQL DO block simple expressions"))
	{
	}

	~simple_eval_resources()
	{
		try
		{
			cppgres::ffi_guard{::FreeExecutorState}(estate_);
			cppgres::ffi_guard{::ReleaseAllPlanCacheRefsInOwner}(resowner_);
			cppgres::ffi_guard{::ResourceOwnerDelete}(resowner_);
		}
		catch (...)
		{
			elog(WARNING, "uplpgsql: releasing DO block resources failed");
		}
	}

	simple_eval_resources(const simple_eval_resources &) = delete;
	simple_eval_resources &operator=(const simple_eval_resources &) = delete;

	EState *estate() const { return estate_; }
	ResourceOwner resowner() const { return resowner_; }

private:
	EState	   *estate_;
	ResourceOwner resowner_;
};

/*
 * Scope-fail guard: when the scope is left by an exception, fire the
 * subtransaction-abort callback to tear down partially-set-up execution
 * state (the RAII equivalent of the old PG_CATCH-only cleanup).
 */
struct abort_callback_guard
{
	abort_callback_guard() : entry_exceptions_(std::uncaught_exceptions()) {}

	~abort_callback_guard()
	{
		if (std::uncaught_exceptions() <= entry_exceptions_)
			return;

		try
		{
			cppgres::ffi_guard{[] {
				uplpgsql_subxact_cb(SUBXACT_EVENT_ABORT_SUB,
									GetCurrentSubTransactionId(), 0, NULL);
			}}();
		}
		catch (...)
		{
			elog(WARNING, "uplpgsql: DO block abort cleanup failed");
		}
	}

	abort_callback_guard(const abort_callback_guard &) = delete;
	abort_callback_guard &operator=(const abort_callback_guard &) = delete;

private:
	int			entry_exceptions_;
};

/*
 * Subtransaction guard for the JIT-compile fallback: rolls back unless
 * commit() was called.
 *
 * cppgres::internal_subtransaction fixes its commit-or-rollback choice at
 * construction, but the fallback needs rollback-on-exception with
 * commit-on-success, so this guard makes the choice at scope exit.  It
 * mirrors the memory context and resource owner restoration of the C
 * version's BeginInternalSubTransaction / Release / RollbackAndRelease
 * sequence.
 */
struct compile_subtransaction
{
	compile_subtransaction()
		: oldcontext_(::CurrentMemoryContext),
		  oldowner_(::CurrentResourceOwner)
	{
		cppgres::ffi_guard{::BeginInternalSubTransaction}(nullptr);
		::CurrentMemoryContext = oldcontext_;
	}

	void
	commit()
	{
		cppgres::ffi_guard{::ReleaseCurrentSubTransaction}();
		restore();
		committed_ = true;
	}

	~compile_subtransaction()
	{
		if (committed_)
			return;

		/*
		 * Unwinding after a compilation error: the error state was already
		 * captured and flushed by cppgres::pg_exception, so all that is left
		 * is to roll the subtransaction back.  A destructor must not throw;
		 * rollback failing is a can't-happen, but degrade to a warning
		 * rather than std::terminate.
		 */
		try
		{
			::CurrentMemoryContext = oldcontext_;
			cppgres::ffi_guard{::RollbackAndReleaseCurrentSubTransaction}();
			restore();
		}
		catch (...)
		{
			elog(WARNING, "uplpgsql: rollback of JIT compile subtransaction failed");
		}
	}

	compile_subtransaction(const compile_subtransaction &) = delete;
	compile_subtransaction &operator=(const compile_subtransaction &) = delete;

private:
	void
	restore()
	{
		::CurrentMemoryContext = oldcontext_;
		::CurrentResourceOwner = oldowner_;
	}

	::MemoryContext oldcontext_;
	::ResourceOwner oldowner_;
	bool committed_ = false;
};

/*
 * Resolve the JIT'd code for a compiled function: consult the cache, and on
 * a miss either compile (inside a subtransaction, falling back to the
 * interpreter on failure) or record the heuristic's decision to skip.
 *
 * Returns nullptr when the function should run under the interpreter.
 */
static UPLpgSQL_func *
jit_lookup_or_compile(UPLpgSQL_function *func)
{
	auto [status, cached] = upl::cache().lookup(func->fn_oid,
												func->cfunc.fn_xmin,
												func->cfunc.fn_tid,
												func);

	switch (status)
	{
		case upl::cache_status::hit:
			return (UPLpgSQL_func *) cached;

		case upl::cache_status::skip:
			return nullptr;

		case upl::cache_status::miss:
			break;
	}

	/*
	 * First call for this function version.  By default we JIT everything.
	 * When uplpgsql.enable_jit_heuristic is on, run the cost/benefit scorer
	 * to decide.
	 */
	if (uplpgsql_enable_jit_heuristic && !uplpgsql_should_jit(func))
	{
		/* Heuristic says skip — cache that decision */
		upl::cache().store_skip(func->fn_oid,
								func->cfunc.fn_xmin,
								func->cfunc.fn_tid,
								func);
		if (uplpgsql_log_compilation)
			elog(LOG, "uplpgsql: skipping JIT for %s (heuristic)",
				 func->fn_signature);
		return nullptr;
	}

	/*
	 * Compile inside an internal subtransaction.
	 *
	 * Compilation can raise a Postgres error — an unsupported construct, an
	 * expression whose plan will not prepare, an LLVM verification failure —
	 * and we intend to catch that and fall back to the interpreter.
	 * Catching an arbitrary error and carrying on is only safe when a
	 * subtransaction restores the memory context, resource owner, buffer
	 * pins and syscache references; otherwise the "graceful fallback" leaks
	 * whatever the failed compile was holding.  That is not hypothetical: a
	 * simple CASE over a non-integer value fails to plan at compile time and
	 * leaked syscache references on every such function.
	 *
	 * cppgres::ffi_guard turns the error's longjmp into a pg_exception
	 * (capturing and flushing the error state), the subtransaction guard
	 * rolls back during unwind, and the catch block logs the cause.  We must
	 * NOT report the error to the client: compilation failing is not a query
	 * failure — execution continues under the interpreter — so it goes to
	 * the server log instead, folded into the fallback notice: at LOG when
	 * uplpgsql.log_compilation is on, at DEBUG1 otherwise.
	 */
	try
	{
		compile_subtransaction subtx;
		UPLpgSQL_func *jitfunc;

		jitfunc = cppgres::ffi_guard{uplpgsql_compile_function}(func);
		upl::cache().store(func->fn_oid,
						   func->cfunc.fn_xmin,
						   func->cfunc.fn_tid,
						   func,
						   (UPL_func *) jitfunc);

		subtx.commit();

		if (uplpgsql_log_compilation)
			elog(LOG, "uplpgsql: JIT compiled function %s (oid %u)",
				 func->fn_signature, func->fn_oid);

		return jitfunc;
	}
	catch (const cppgres::pg_exception &e)
	{
		elog(uplpgsql_log_compilation ? LOG : DEBUG1,
			 "uplpgsql: JIT compilation failed for %s (%s), "
			 "using interpreter",
			 func->fn_signature, e.message());
		return nullptr;
	}
}

} // namespace uplpgsql

/*
 * uplpgsql_call_handler - main entry point for function/trigger execution
 *
 * For regular functions and triggers, we attempt JIT compilation on first
 * call and execute the native code. On subsequent calls, the cached JIT'd
 * function is used directly. Falls back to the interpreter for event
 * triggers (JIT support for those is future work).
 */
Datum
uplpgsql_call_handler(PG_FUNCTION_ARGS)
{
	bool		nonatomic;
	bool		isTrigger;
	bool		isEventTrigger;
	Datum		retval = (Datum) 0;

	nonatomic = fcinfo->context &&
		IsA(fcinfo->context, CallContext) &&
		!castNode(CallContext, fcinfo->context)->atomic;

	isTrigger = CALLED_AS_TRIGGER(fcinfo);
	isEventTrigger = CALLED_AS_EVENT_TRIGGER(fcinfo);

	/*
	 * Postgres errors below surface as pg_exception (every call into the
	 * engine runs under cppgres::ffi_guard), so the guards' destructors run
	 * on both exit paths; the catch clauses at the end re-throw to Postgres
	 * with full fidelity.
	 */
	try
	{
		uplpgsql::spi_session spi(nonatomic ? SPI_OPT_NONATOMIC : 0);

		/*
		 * Compile the function (get AST via our forked parser).
		 *
		 * forValidator is false here: we are executing, not validating.  It
		 * must not be confused with the trigger flags — those reach
		 * do_compile() through fcinfo's context, which is already set up by
		 * the caller.  Passing the trigger flag compiled every trigger in
		 * validator mode on every fire, which ran the extra syntax checks at
		 * execution time and, worse, applied uplpgsql.extra_warnings/
		 * extra_errors to a running trigger — so a DML statement could fail
		 * purely because a GUC was set.
		 */
		UPLpgSQL_function *func =
			cppgres::ffi_guard{uplpgsql_compile}(fcinfo, false);

		/* Mark the function as busy, so it can't be deleted from under us */
		uplpgsql::function_busy_guard busy(func);

		/*
		 * If we'll need a procedure-lifespan resowner to execute any CALL or
		 * DO statements, create it now.
		 */
		uplpgsql::procedure_resowner_guard procedure_resowner(
			nonatomic && func->requires_procedure_resowner);

		/*
		 * For regular functions and triggers, try the JIT path.
		 * Event triggers still use the interpreter.
		 */
		UPLpgSQL_func *jitfunc =
			isEventTrigger ? nullptr : uplpgsql::jit_lookup_or_compile(func);

		if (isTrigger)
		{
			auto exec = jitfunc != nullptr
				? cppgres::ffi_guard{uplpgsql_exec_trigger_jit}(
					  func, (TriggerData *) fcinfo->context,
					  jitfunc->jit_func)
				: cppgres::ffi_guard{uplpgsql_exec_trigger}(
					  func, (TriggerData *) fcinfo->context);

			retval = PointerGetDatum(exec);
		}
		else if (isEventTrigger)
		{
			cppgres::ffi_guard{uplpgsql_exec_event_trigger}(
				func, (EventTriggerData *) fcinfo->context);
			/* there's no return value in this case */
		}
		else if (jitfunc != nullptr)
		{
			/* Execute via JIT'd native code */
			retval = cppgres::ffi_guard{uplpgsql_exec_function_jit}(
				func, fcinfo, (EState *) nullptr, (ResourceOwner) nullptr,
				(ResourceOwner) procedure_resowner, !nonatomic,
				jitfunc->jit_func);
		}
		else
		{
			/* Fallback to interpreter */
			retval = cppgres::ffi_guard{uplpgsql_exec_function}(
				func, fcinfo, (EState *) nullptr, (ResourceOwner) nullptr,
				(ResourceOwner) procedure_resowner, !nonatomic);
		}
	}
	catch (cppgres::pg_exception &e)
	{
		e.rethrow();
	}
	catch (const std::exception &e)
	{
		cppgres::report(ERROR, "%s", e.what());
	}

	return retval;
}

/*
 * uplpgsql_inline_handler - DO block execution
 *
 * Executes anonymous code blocks (DO $$ ... $$ LANGUAGE uplpgsql).
 * DO blocks are always interpreted (not JIT'd) since they run once and
 * compilation overhead would exceed any benefit.
 *
 * This mirrors plpgsql_inline_handler() — sets up a private EState and
 * resowner for simple-expression execution that survive COMMIT/ROLLBACK.
 */
Datum
uplpgsql_inline_handler(PG_FUNCTION_ARGS)
{
	LOCAL_FCINFO(fake_fcinfo, 0);
	InlineCodeBlock *codeblock = castNode(InlineCodeBlock,
										  DatumGetPointer(PG_GETARG_DATUM(0)));
	FmgrInfo	flinfo;
	Datum		retval = (Datum) 0;

	try
	{
		uplpgsql::spi_session spi(codeblock->atomic ? 0 : SPI_OPT_NONATOMIC);

		/* Compile the anonymous code block */
		UPLpgSQL_function *func = cppgres::ffi_guard{uplpgsql_compile_inline}(
			codeblock->source_text);

		/*
		 * Mark the function as busy (just pro forma) and arrange for its
		 * memory to be freed on every exit path — a DO block's AST is
		 * single-use.
		 */
		uplpgsql::inline_function_guard func_guard(func);

		/*
		 * Set up a fake fcinfo with just enough info to satisfy
		 * uplpgsql_exec_function().
		 */
		MemSet(fake_fcinfo, 0, SizeForFunctionCallInfo(0));
		MemSet(&flinfo, 0, sizeof(flinfo));
		fake_fcinfo->flinfo = &flinfo;
		flinfo.fn_oid = InvalidOid;
		flinfo.fn_mcxt = CurrentMemoryContext;

		/*
		 * Create a private EState and resowner for simple-expression
		 * execution.  These must survive any COMMIT/ROLLBACK the DO block
		 * executes, and are released on every exit path.
		 */
		uplpgsql::simple_eval_resources simple_eval;

		/*
		 * On failure only: fire the subtransaction-abort callback so
		 * partially-set-up execution state is torn down before the
		 * resources above are released (guards destruct innermost-first).
		 */
		uplpgsql::abort_callback_guard on_error;

		retval = cppgres::ffi_guard{uplpgsql_exec_function}(
			func, fake_fcinfo, simple_eval.estate(), simple_eval.resowner(),
			simple_eval.resowner(), codeblock->atomic);
	}
	catch (cppgres::pg_exception &e)
	{
		e.rethrow();
	}
	catch (const std::exception &e)
	{
		cppgres::report(ERROR, "%s", e.what());
	}

	return retval;
}

/*
 * uplpgsql_validator - CREATE FUNCTION validation
 *
 * Validates by test-compiling the function with PL/pgSQL's parser.
 */
Datum
uplpgsql_validator(PG_FUNCTION_ARGS)
{
	Oid			funcoid = PG_GETARG_OID(0);
	bool		is_trigger = false;
	bool		is_event_trigger = false;

	if (!CheckFunctionValidatorAccess(fcinfo->flinfo->fn_oid, funcoid))
		PG_RETURN_VOID();

	try
	{
		/* Get the function's return type from its pg_proc entry */
		{
			cppgres::syscache<Form_pg_proc, cppgres::oid> proc(funcoid);

			if ((*proc).prorettype == TRIGGEROID)
				is_trigger = true;
			else if ((*proc).prorettype == EVENT_TRIGGEROID)
				is_event_trigger = true;
		}

		/* Postpone body checks if !check_function_bodies */
		if (check_function_bodies)
		{
			LOCAL_FCINFO(fake_fcinfo, 0);
			FmgrInfo	flinfo;
			TriggerData trigdata;
			EventTriggerData etrigdata;

			uplpgsql::spi_session spi(0);

			MemSet(fake_fcinfo, 0, SizeForFunctionCallInfo(0));
			MemSet(&flinfo, 0, sizeof(flinfo));
			fake_fcinfo->flinfo = &flinfo;
			flinfo.fn_oid = funcoid;
			flinfo.fn_mcxt = CurrentMemoryContext;

			if (is_trigger)
			{
				MemSet(&trigdata, 0, sizeof(trigdata));
				trigdata.type = T_TriggerData;
				fake_fcinfo->context = (Node *) &trigdata;
			}
			else if (is_event_trigger)
			{
				MemSet(&etrigdata, 0, sizeof(etrigdata));
				etrigdata.type = T_EventTriggerData;
				fake_fcinfo->context = (Node *) &etrigdata;
			}

			/*
			 * Test-compile the function.
			 *
			 * forValidator must be true — we are the validator.  It is what
			 * enables the extra syntax checks, the extra_warnings/
			 * extra_errors checks, polymorphic return-type resolution, and
			 * source-position reporting in compile errors; do_compile()
			 * zeroes all of that when it is false.  The trigger flags are
			 * conveyed via fake_fcinfo's context, set above, not through
			 * this argument.
			 */
			cppgres::ffi_guard{uplpgsql_compile}(fake_fcinfo, true);
		}
	}
	catch (cppgres::pg_exception &e)
	{
		e.rethrow();
	}
	catch (const std::exception &e)
	{
		cppgres::report(ERROR, "%s", e.what());
	}

	PG_RETURN_VOID();
}
