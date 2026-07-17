# C++20 / cppgres rewrite notes

This tree is being rewritten from C to coherent C++20 on top of
[cppgres](https://github.com/cppgres/cppgres) (vendored as the amalgamated
single header `cppgres.hpp`, generated with `cpp-amalgamate src/cppgres.hpp`),
and from the LLVM C API to the LLVM C++ API.

## Result

The rewrite is complete: no `PG_TRY` outside PostgreSQL itself, no raw
`SPI_*` call sites (everything runs under `cppgres::ffi_guard` or the
cppgres SPI executor/plan API; the only exception is
`SPI_result_code_string`/`SPI_plan_get_plan_sources`, static lookups that
cannot error), no `foreach` (ranges over `cppgres::list<T>`), and the
compilers/handler/interpreter boundaries are classes
(`uplpgsql::function_compiler`, `uplpgsql::call`, `uplpgsql::do_block`).

One hard-won lesson is recorded here for future conversions: plpgsql's C
code sometimes reads a `foreach` loop's cell variable *after* the loop
("did any handler match" in `exec_stmt_block`, first-iteration tests in
dump code).  A range-for conversion that leaves the old `ListCell`
declaration behind turns those reads into uninitialized-variable UB that
only detonates at `-O2`.  Convert such loops with explicit flags.

## Stages

1. **Transitional (done first):** every translation unit compiles as C++20
   with minimal, semantics-preserving edits. Regression suite green.
2. **Idiomatic:** module-by-module rewrite, regression suite green after each
   module:
   - `core/` — `upl::` namespace. LLVM C++ API (`llvm::IRBuilder<>`,
     `llvm::Module`, `llvm::orc::LLJIT`). The bitcode round-trip in
     `upl_jit_compile` disappears: modules are built in a context that is
     moved into `llvm::orc::ThreadSafeModule`. Cache becomes a class over
     `std::unordered_map`. Compile context becomes a class whose emit
     helpers are methods.
   - `drivers/plpgsql/` handler — cppgres: `exception_guard`/`ffi_guard`,
     GUC registration, memory contexts, subtransaction-guarded compile
     fallback.
   - `common/` compile layer (`upl_compile_stmts.cpp`,
     `upl_compile_expr.cpp`) — LLVM C++ API alongside core.
   - `common/` fork layer (`upl_exec.cpp`, `upl_comp.cpp`, `upl_funcs.cpp`,
     `upl_runtime.cpp`) — although forked from PostgreSQL's
     `pl_exec.c`/`pl_comp.c`/`pl_funcs.c`, upstream diffability is
     explicitly NOT a goal: these are full C++20 citizens like everything
     else (classes over state-threading free functions, nullptr, C++
     casts, value initialization, C++ strings where lifetimes allow).
     The only exemptions are hard ABI: `extern "C"` `uplpgsql_rt_*`
     symbol names for OrcJIT, JIT-GEP'd struct layouts
     (`UPLpgSQL_exec_state`, the datum offsets), and the generated
     parser skeleton's internals.
   - Grammar/scanner stay flex/bison with C skeletons, compiled as C++;
     actions are C++-clean.

   Error handling (phase 2): exception-based throughout.  Calls into the
   engine run under `cppgres::ffi_guard`; entry points convert back at
   their boundary with `pg_exception::rethrow()` (full fidelity) or
   `cppgres::report`.  Cleanup is RAII — including error-path-only cleanup
   (scope-fail guards via `std::uncaught_exceptions`), never try/catch.
   Runtime helpers called from JIT'd code are exception barriers: a C++
   exception must never unwind into a JIT frame.

   API shape: no bare multi-positional engine calls with casted nulls at
   call sites.  Call-scoped state and dispatch live in classes
   (`uplpgsql::call`, `uplpgsql::do_block`, the compiler class), and the
   entry points reduce to constructing them inside an SPI session and
   calling `execute()`.

## Invariants (do not break)

- **RT helper linkage.** JIT'd code resolves `uplpgsql_rt_*` helpers *by
  symbol name* through Orc's process-symbol generator. They must remain
  `extern "C"`, default visibility (`UPL_RT_EXPORT`), with C-compatible
  signatures. Same for `sigsetjmp` and the PG dynloader entry points
  (`_PG_init`, `pg_finfo_*`, handler functions).
- **Longjmp vs destructors.** `elog(ERROR)` longjmps and does NOT run C++
  destructors. Code holding RAII resources must call into PG through
  `cppgres::ffi_guard` (converts the longjmp into a C++ `pg_exception`), and
  the extension boundary converts exceptions back to PG errors
  (`cppgres::exception_guard`). Until a module is converted, PG-style
  error flow (trivially-destructible locals only) is acceptable.
- **PG macro collisions with LLVM C++ headers.** Include PG headers (inside
  `extern "C"`) first, then `#undef _` and `#undef gettext` before including
  any `llvm/*` C++ header. PG's `_` macro otherwise rewrites LLVM's
  `ErrorAsOutParameter _(Err);` into a shadowing declaration and silently
  changes behavior.
- **Exceptions stay inside the extension.** Nothing may throw across a JIT'd
  frame or into PG. The JIT'd code itself uses sigsetjmp-based frames
  (`uplpgsql_rt_exception_*`), not C++ EH; that machinery is ABI, not style.
- **`-fexceptions` is required**; LLVM's `--cxxflags` suggests
  `-fno-exceptions` and must not be taken wholesale.

## Verification loop

```
make install PG_CONFIG=<pg19>/bin/pg_config
make installcheck PG_CONFIG=... PGPORT=... PGHOST=... PGUSER=...
```
All 15 `plpgsql_*` regression tests must pass (baseline established on
PostgreSQL 19beta2).
