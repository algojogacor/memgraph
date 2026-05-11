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

#include <cstdint>
#include <span>

#include "query/plan_v2/private_analysis.hpp"
#include "query/plan_v2/private_symbol.hpp"

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2 {

using EGraph = planner::core::EGraph<symbol, analysis>;

/// Pluggable cardinality estimator.
///
/// The cost model calls this once per e-node that needs a cardinality estimate
/// (currently: Function).  The enode, its argument e-classes, and the full
/// e-graph are passed so implementations can walk the graph for constant
/// deduction.  Non-Function callers receive kDefaultRowEstimate by default.
///
/// Concrete implementations:
///   - BuiltinEstimator  - constant deduction for builtins (e.g. range(0,5)
///                         → 6); falls back to kDefaultRowEstimate otherwise.
///                         Production default.
struct CardinalityEstimator {
  virtual ~CardinalityEstimator() = default;

  virtual auto Estimate(planner::core::ENode<symbol> const &enode,
                        std::span<planner::core::EClassId const> arg_eclasses, EGraph const &eg) const -> double = 0;
};

}  // namespace memgraph::query::plan::v2
