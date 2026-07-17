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
   - `common/` — `uplpgsql::` namespace; RAII where lifetimes allow, C++
     casts, `std::string_view`/`std::span` at internal seams. The exec
     engine keeps its PG-derived structure (it is a fork of `pl_exec.c` and
     must track PG semantics) but reads as C++.
   - Grammar/scanner stay flex/bison with C skeletons, compiled as C++;
     actions are C++-clean.

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
