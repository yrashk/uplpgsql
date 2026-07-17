/*-------------------------------------------------------------------------
 *
 * upl_cache.cpp
 *		Compiled function cache — see upl_cache.hpp for the contract.
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
#include "upl_cache.hpp"

namespace upl {

function_cache &
cache()
{
	/*
	 * Heap-allocated per backend, alive for the backend's lifetime — the
	 * C++ analogue of the old TopMemoryContext hash table.
	 */
	static function_cache instance;

	return instance;
}

function_cache::result
function_cache::lookup(Oid fn_oid, TransactionId fn_xmin,
					   ItemPointerData fn_tid, const void *current_lang_func)
{
	auto it = entries_.find(fn_oid);

	if (it == entries_.end())
		return {};

	entry &e = it->second;

	/*
	 * Invalidate when the function was replaced (CREATE OR REPLACE changes
	 * fn_xmin/fn_tid) or when the language compiler recreated its function
	 * struct: the JIT'd code has AST node pointers from the original struct
	 * embedded as LLVM constants, and those would now dangle.
	 */
	if (e.fn_xmin != fn_xmin ||
		!ItemPointerEquals(&e.fn_tid, &fn_tid) ||
		e.lang_func != current_lang_func)
	{
		elog(DEBUG1, "upl: invalidating cache for function %u "
			 "(xmin %u -> %u, lang_func %p -> %p)",
			 fn_oid,
			 e.fn_xmin, fn_xmin,
			 e.lang_func, current_lang_func);

		/*
		 * Erase so the heuristic is re-evaluated.  Old JIT'd code (if any)
		 * is intentionally leaked.
		 */
		entries_.erase(it);
		return {};
	}

	if (e.skip)
		return {cache_status::skip, nullptr};

	return {cache_status::hit, e.func};
}

void
function_cache::store(Oid fn_oid, TransactionId fn_xmin,
					  ItemPointerData fn_tid, const void *lang_func,
					  UPL_func *func)
{
	entries_[fn_oid] = {func, false, fn_xmin, fn_tid, lang_func};
}

void
function_cache::store_skip(Oid fn_oid, TransactionId fn_xmin,
						   ItemPointerData fn_tid, const void *lang_func)
{
	entries_[fn_oid] = {nullptr, true, fn_xmin, fn_tid, lang_func};
}

} // namespace upl
