// Copyright 2026 Memgraph Ltd.
//
// Use of this software is governed by the Business Source License
// included in the file licenses/BSL.txt; by using this file, you agree to be bound by the terms of the Business Source
// License, and you may not use this file except in compliance with the Business Source License.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0, included in the file
// licenses/APL.txt.

#pragma once

#include <initializer_list>

#include "query/plan_v2/bind_semantics.hpp"

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2::bind {

/// Construct a SymbolSet from a brace-initialiser list of raw uint32_t IDs.
/// Shared across plan_v2 unit tests that need lightweight SymbolSet fixtures.
inline auto MakeSet(std::initializer_list<uint32_t> ids) -> SymbolSet {
  SymbolSet s;
  for (auto id : ids) s.insert(planner::core::EClassId{id});
  return s;
}

}  // namespace memgraph::query::plan::v2::bind
