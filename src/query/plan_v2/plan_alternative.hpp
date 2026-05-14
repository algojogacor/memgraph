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

#include <compare>

#include "planner/extract/pareto_frontier.hpp"
#include "query/plan_v2/bind_semantics.hpp"

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2 {

/// One candidate plan reachable at an e-class.
///
/// `cardinality` is the number of values flowing through this point in the
/// plan: 1 for scalar expressions, list length for list-producing
/// expressions, output rows for row-pipe operators (Output, future Unwind /
/// scans).  Composition at row-pipe operators multiplies child cardinalities.
///
/// `introduces` / `required` participate in the kind dichotomy described in
/// `src/query/plan_v2/CONTEXT.md`: operator Alts populate `introduces`;
/// expression Alts populate `required`.  Whether a Bind is "alive" or "dead"
/// is derived at read time from `sym ∈ chosen.introduces` where
/// `sym = enode.children()[1]`.
struct Alternative {
  double cost;
  /// Number of values flowing through this point in the plan.  Defaults to 1
  /// (scalar): leaves, identifiers, and per-evaluation expression operators
  /// all sit at cardinality 1; row-pipe operators inherit / multiply.
  double cardinality = 1.0;
  /// Symbols that MUST be bound by ancestors.  Operator Alts: always empty
  /// (residual demand is absorbed at the operator's construction-time
  /// validation against `input.introduces`).  Expression Alts: free `Identifier`
  /// references this expression demands from scope.
  bind::SymbolSet required;
  /// Symbols this subtree's row pipe makes AVAILABLE TO OPERATORS THAT PULL
  /// ROWS FROM IT.  Operator Alts: `input.introduces ∪ own_syms` where
  /// own_syms is the operator's own bindings (Bind/Unwind sym, Output's
  /// NamedOutput syms, Subquery's exposed_syms).  Expression Alts: always
  /// empty.
  ///
  /// Scope barriers strip introductions: see symbol::Subquery's cost case
  /// for the canonical pattern - inner.introduces is dropped at the
  /// boundary, only the explicit exposed_syms from the e-node's children
  /// cross into the outer scope.
  bind::SymbolSet introduces;
  planner::core::ENodeId enode_id;  ///< Which enode achieves this alternative
};

/// Pareto dimensions for Alternative — four axes:
///   - cost        : lower is better
///   - cardinality : lower is better (parents multiply per-row cost
///                   contributions by it, so a smaller row pipe wins
///                   unconditionally when cost and required tie)
///   - required    : smaller subset is better (needs less from ancestors)
///   - introduces  : larger subset is better (helps a sibling NamedOutput
///                   absorb its required-set demand; without this axis a
///                   cheap dead-Bind alt would dominate the alive-Bind
///                   alt that downstream Outputs actually depend on).
using AlternativeDim_Cost = planner::core::extract::Dim<&Alternative::cost, planner::core::extract::LowerIsBetter>;
using AlternativeDim_Cardinality =
    planner::core::extract::Dim<&Alternative::cardinality, planner::core::extract::LowerIsBetter>;
using AlternativeDim_Required =
    planner::core::extract::Dim<&Alternative::required, planner::core::extract::SmallerSubsetIsBetter>;
using AlternativeDim_Introduces =
    planner::core::extract::Dim<&Alternative::introduces, planner::core::extract::LargerSubsetIsBetter>;

}  // namespace memgraph::query::plan::v2
