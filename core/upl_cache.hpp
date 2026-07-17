/*-------------------------------------------------------------------------
 *
 * upl_cache.hpp
 *		Compiled function cache — per-backend map keyed by function OID.
 *
 *		The cache is language-agnostic.  It stores compiled function
 *		pointers with invalidation based on fn_xmin/fn_tid (pg_proc row
 *		version) and on the identity of the language-specific function
 *		struct (the JIT'd code embeds AST node pointers from that struct
 *		as constants, so a recompiled struct makes the code stale even
 *		when the pg_proc row is unchanged).
 *
 *		Three states per function:
 *		  - miss: not yet seen → caller evaluates the JIT heuristic,
 *		    then either compiles and store()s, or store_skip()s
 *		  - skip: the heuristic decided JIT wouldn't help → interpreter
 *		    on all future calls
 *		  - hit:  JIT'd code is available
 *
 *		Old JIT'd code is intentionally leaked when entries are
 *		invalidated: LLJIT does not support cheap per-function removal.
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
#pragma once

#include "upl.h"

#include <unordered_map>

namespace upl {

enum class cache_status
{
	miss,						/* not cached: evaluate heuristic, compile */
	skip,						/* heuristic said no: use the interpreter */
	hit,						/* JIT'd code available */
};

struct function_cache
{
	struct result
	{
		cache_status status = cache_status::miss;
		UPL_func   *func = nullptr; /* set on hit */
	};

	/*
	 * Look up a function by OID, validating against the current pg_proc row
	 * version (fn_xmin/fn_tid) and the current language-specific function
	 * struct pointer.  A stale entry is erased and reported as a miss so the
	 * caller re-evaluates the heuristic.
	 */
	result lookup(Oid fn_oid, TransactionId fn_xmin, ItemPointerData fn_tid,
				  const void *current_lang_func);

	void store(Oid fn_oid, TransactionId fn_xmin, ItemPointerData fn_tid,
			   const void *lang_func, UPL_func *func);

	void store_skip(Oid fn_oid, TransactionId fn_xmin, ItemPointerData fn_tid,
					const void *lang_func);

private:
	struct entry
	{
		UPL_func	   *func;	/* nullptr when skip is set */
		bool			skip;
		TransactionId	fn_xmin;
		ItemPointerData	fn_tid;
		const void	   *lang_func;
	};

	std::unordered_map<Oid, entry> entries_;
};

/* The per-backend cache instance. */
function_cache &cache();

} // namespace upl
