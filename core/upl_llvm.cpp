/*-------------------------------------------------------------------------
 *
 * upl_llvm.cpp
 *		LLVM infrastructure for the UPL Core Engine.
 *
 *		This file manages the LLVM machinery that underlies the JIT
 *		compiler.  It is completely language-agnostic — it knows nothing
 *		about statements, expressions, or any specific procedural language.
 *
 *		Key components:
 *
 *		upl_llvm_init()       — One-time initialization per backend:
 *		  initializes LLVM native target, creates a target machine for
 *		  the host platform, and creates a single LLJIT instance that is
 *		  shared by all compiled functions in the backend.  Registers a
 *		  process symbol search generator so that runtime helpers (and any
 *		  other process symbols) are automatically resolvable from JIT'd
 *		  code without explicit symbol registration.
 *
 *		upl_register_types()  — Populates ctx->types[] with LLVM type
 *		  references (void, i1, i8, i16, i32, i64, double, ptr, Datum,
 *		  function type).  Called once per compilation.
 *
 *		upl_verify_module()   — Runs LLVM's IR verification pass.
 *		  Catches malformed IR before it reaches the optimizer or code gen.
 *
 *		upl_optimize_module() — Runs the LLVM new pass manager with
 *		  a specified optimization level (O0-O3).  Uses the target machine
 *		  for target-specific optimizations.
 *
 *		upl_jit_compile()     — The final step: moves the module and its
 *		  context into an orc::ThreadSafeModule (no serialization — with
 *		  the C++ API a module changes hands by move; the old C-API
 *		  implementation had to round-trip through bitcode), adds it to
 *		  LLJIT, and looks up the function symbol to return a native
 *		  function pointer.
 *
 *		Design notes:
 *		  - One LLJIT per backend (not per function) to amortize setup cost
 *		  - Old compiled functions are intentionally leaked because LLJIT
 *		    doesn't support cheap per-function removal
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

/* See doc/cpp-rewrite.md on the _/gettext dance around LLVM C++ headers. */
#undef _
#undef gettext

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#ifndef ENABLE_NLS
#define gettext(x) (x)
#endif
#define _(x) gettext(x)

#include <string>

namespace {

/*
 * Per-backend LLVM state.
 *
 * jit_instance: the single LLJIT instance shared by all compiled functions
 *   in this backend.  Created once in upl_llvm_init().
 *
 * target_machine: used for optimization passes (the new pass manager needs
 *   a target machine to apply target-specific transformations).
 *
 * llvm_initialized: prevents double initialization.
 */
std::unique_ptr<llvm::orc::LLJIT> jit_instance;
std::unique_ptr<llvm::TargetMachine> target_machine;
bool llvm_initialized = false;

/*
 * Render an llvm::Error into a palloc'd string, consuming the error.
 */
char *
error_to_pstr(llvm::Error err)
{
	std::string s = llvm::toString(std::move(err));

	return pstrdup(s.c_str());
}

/*
 * The host CPU's feature string, in the "+feat1,-feat2,..." form
 * createTargetMachine() expects.
 */
std::string
host_cpu_features()
{
	std::string features;

	for (const auto &[name, enabled] : llvm::sys::getHostCPUFeatures())
	{
		features += (enabled ? '+' : '-');
		features += name.str();
		features += ',';
	}
	if (!features.empty())
		features.pop_back();

	return features;
}

} // namespace

/*
 * upl_llvm_init - Initialize LLVM (once per process)
 */
void
upl_llvm_init(void)
{
	if (llvm_initialized)
		return;

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	llvm::InitializeNativeTargetAsmParser();

	/*
	 * Create target machine (used for optimization passes).
	 *
	 * Guarded so that a retry — reached when a step below elog(ERROR)s and
	 * longjmps out before llvm_initialized is set — reuses the machine
	 * built by the failed attempt instead of leaking it and building another.
	 */
	if (target_machine == nullptr)
	{
		llvm::Triple triple(llvm::sys::getDefaultTargetTriple());
		std::string error;
		const llvm::Target *target =
			llvm::TargetRegistry::lookupTarget(triple, error);

		if (target == nullptr)
			elog(ERROR, "upl: failed to get LLVM target: %s",
				 pstrdup(error.c_str()));

		target_machine.reset(target->createTargetMachine(
			triple,
			llvm::sys::getHostCPUName(),
			host_cpu_features(),
			llvm::TargetOptions(),
			llvm::Reloc::PIC_,
			/* code model: JIT default */ std::nullopt,
			llvm::CodeGenOptLevel::Aggressive));

		/*
		 * createTargetMachine reports failure by returning null; it has no
		 * error-message out-param, so name the triple we failed on.
		 */
		if (target_machine == nullptr)
			elog(ERROR, "upl: failed to create LLVM target machine for %s",
				 pstrdup(triple.str().c_str()));
	}

	/*
	 * Create the OrcJIT instance, with a process-symbol search generator so
	 * runtime helpers are automatically resolvable from JIT'd code.
	 */
	{
		auto jit = llvm::orc::LLJITBuilder().create();

		if (!jit)
			elog(ERROR, "upl: failed to create OrcJIT: %s",
				 error_to_pstr(jit.takeError()));

		auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
			(*jit)->getDataLayout().getGlobalPrefix());

		if (!gen)
			elog(ERROR, "upl: failed to create symbol generator: %s",
				 error_to_pstr(gen.takeError()));

		(*jit)->getMainJITDylib().addGenerator(std::move(*gen));
		jit_instance = std::move(*jit);
	}

	/*
	 * Set last, once the target machine and JIT instance both exist.  The
	 * steps above elog(ERROR) on failure, which longjmps out; setting the
	 * flag any earlier would make a retry short-circuit above and leave
	 * jit_instance null for the life of the backend.
	 */
	llvm_initialized = true;

	elog(DEBUG1, "upl: LLVM %d.%d initialized, OrcJIT ready",
		 LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR);
}

/*
 * upl_llvm_shutdown - Clean up LLVM resources
 */
void
upl_llvm_shutdown(void)
{
	jit_instance.reset();
	target_machine.reset();
}

/*
 * upl_register_types - Register LLVM types in a compilation context
 */
void
upl_register_types(UPL_compile_ctx *ctx)
{
	/*
	 * UPL_INTPTR is Int64 and upl_const_ptr() builds pointers out of UPL_INT64,
	 * so a 32-bit host would silently truncate every embedded pointer.  All
	 * supported builds are 64-bit; fail at compile time rather than at run
	 * time if that ever changes.
	 */
	static_assert(sizeof(void *) == 8, "UPL assumes 64-bit pointers");

	llvm::LLVMContext &c = *ctx->context;

	ctx->types[UPL_VOID]   = llvm::Type::getVoidTy(c);
	ctx->types[UPL_INT1]   = llvm::Type::getInt1Ty(c);
	ctx->types[UPL_INT8]   = llvm::Type::getInt8Ty(c);
	ctx->types[UPL_INT16]  = llvm::Type::getInt16Ty(c);
	ctx->types[UPL_INT32]  = llvm::Type::getInt32Ty(c);
	ctx->types[UPL_INT64]  = llvm::Type::getInt64Ty(c);
	ctx->types[UPL_DOUBLE] = llvm::Type::getDoubleTy(c);
	ctx->types[UPL_PTR]    = llvm::PointerType::getUnqual(c);
	ctx->types[UPL_INTPTR] = llvm::Type::getInt64Ty(c);	/* 64-bit platforms */
	ctx->types[UPL_DATUM]  = ctx->types[UPL_INT64];

	/* Function type: int32 func(ptr estate) */
	ctx->types[UPL_FUNC_TYPE] = llvm::FunctionType::get(
		ctx->types[UPL_INT32], {ctx->types[UPL_PTR]}, false);
}

/*
 * upl_verify_module - Verify LLVM module IR is well-formed
 */
void
upl_verify_module(llvm::Module &module)
{
	std::string error;
	llvm::raw_string_ostream os(error);

	if (llvm::verifyModule(module, &os))
		elog(ERROR, "upl: LLVM module verification failed: %s",
			 pstrdup(os.str().c_str()));
}

/*
 * upl_optimize_module - Run LLVM optimization passes
 */
void
upl_optimize_module(llvm::Module &module, int level)
{
	static const llvm::OptimizationLevel levels[] = {
		llvm::OptimizationLevel::O0,
		llvm::OptimizationLevel::O1,
		llvm::OptimizationLevel::O2,
		llvm::OptimizationLevel::O3,
	};

	if (level < 0 || level > 3)
		level = 2;

	llvm::LoopAnalysisManager lam;
	llvm::FunctionAnalysisManager fam;
	llvm::CGSCCAnalysisManager cgam;
	llvm::ModuleAnalysisManager mam;
	llvm::PassBuilder pb(target_machine.get());

	pb.registerModuleAnalyses(mam);
	pb.registerCGSCCAnalyses(cgam);
	pb.registerFunctionAnalyses(fam);
	pb.registerLoopAnalyses(lam);
	pb.crossRegisterProxies(lam, fam, cgam, mam);

	llvm::ModulePassManager mpm =
		(level == 0) ? pb.buildO0DefaultPipeline(levels[level])
					 : pb.buildPerModuleDefaultPipeline(levels[level]);

	mpm.run(module, mam);
}

/*
 * upl_jit_compile - Add module to OrcJIT and look up a function.
 *
 * Takes ownership of the module and its context (they move into OrcJIT's
 * ThreadSafeModule).  Returns the native function pointer.
 */
void *
upl_jit_compile(std::unique_ptr<llvm::Module> module,
				std::unique_ptr<llvm::LLVMContext> context,
				const char *func_name)
{
	llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));

	if (llvm::Error err = jit_instance->addIRModule(std::move(tsm)))
		elog(ERROR, "upl: failed to add module to OrcJIT: %s",
			 error_to_pstr(std::move(err)));

	/* Look up the compiled function */
	auto sym = jit_instance->lookup(func_name);

	if (!sym)
		elog(ERROR, "upl: symbol lookup failed for %s: %s",
			 func_name, error_to_pstr(sym.takeError()));

	void	   *addr = sym->toPtr<void *>();

	elog(DEBUG1, "upl: JIT compiled %s at %p", func_name, addr);

	return addr;
}
