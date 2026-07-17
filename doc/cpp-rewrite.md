# C++20 / cppgres rewrite notes

This tree is being rewritten from C to coherent C++20 on top of
[cppgres](https://github.com/cppgres/cppgres) (vendored as the amalgamated
single header `cppgres.hpp`, generated with `cpp-amalgamate src/cppgres.hpp`),
and from the LLVM C API to the LLVM C++ API.

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
     `upl_runtime.cpp`) — these are deliberate forks of PostgreSQL's
     `pl_exec.c`/`pl_comp.c`/`pl_funcs.c` and their maintainability comes
     from staying diffable against upstream.  They compile as C++20 and get
     a light idiomatic pass (nullptr, no writable-string conversions, C++
     types at the compiler seam), but keep PG structure and error flow by
     design.  Their entry points keep C linkage: the bison parser and the
     OrcJIT symbol resolution depend on it.
   - Grammar/scanner stay flex/bison with C skeletons, compiled as C++;
     actions are C++-clean.

   Error-handling boundaries (implemented in the handler): code that
   recovers from errors itself (the JIT-compile fallback) converts longjmp
   to exceptions via `cppgres::ffi_guard` and uses RAII; execution paths
   whose errors belong to the client stay Postgres-native (`PG_TRY`/
   `PG_FINALLY`) to preserve error fidelity.  The compile pipeline keeps a
   `PG_TRY` that resets the context's LLVM `unique_ptr`s on longjmp before
   re-throwing.

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
