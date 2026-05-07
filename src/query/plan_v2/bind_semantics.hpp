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

// ============================================================================
// Bind semantics: "let sym = expr in input".
// ============================================================================
//
// A Bind introduces a variable.  Some part of the plan uses that variable
// via Identifier; some Bind has to introduce it; otherwise the plan refers
// to a name that doesn't exist.
//
// We track this with two sets, both stored as SymbolSet:
//
//   `required`  - "what variables does this part of the plan still NEED
//                  someone to introduce?"  Built bottom-up by the cost
//                  model.  Identifier(x) starts {x}.  Expressions union
//                  what their children need.  An alive Bind for `x`
//                  takes `x` off the list because it introduces `x`.
//
//   `provided`  - "what variables ARE currently in scope?"  Tracked
//                  top-down by the resolver as it walks the plan.  Each
//                  alive Bind adds its variable when going into the
//                  input below it.
//
// At every node, the resolver checks: are all variables this node needs
// already in scope?  i.e. `required` ⊆ `provided`.  If yes, use it.  If
// no, the plan is broken at that node.
//
// A Bind is "alive" if the input below it actually uses the variable.
// "Dead" if the input doesn't use it - the Bind is just dead weight, and
// the resolver/builder skip it entirely.  Whether a Bind is alive is
// decided once, by the cost model, when it computes `required` and sees
// whether `sym` shows up in the input's needs.

#include <algorithm>
#include <ranges>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2::bind {

/// Set of EClassIds - used for "demand" sets (`required`) and "provided"
/// contexts in the resolver.  flat_set + small_vector keeps small sets
/// (typical Cypher demand depth ≤ 8) heap-free.
using SymbolSet = boost::container::flat_set<planner::core::EClassId, std::less<>,
                                             boost::container::small_vector<planner::core::EClassId, 8>>;

/// Cost of a Symbol leaf alternative.
///
/// Invariant: every Symbol eclass has exactly one alternative
/// `{cost = kSymbolCost, required = ∅}`.  Cost-model and resolver collapse
/// the sym child to a scalar on this assumption; the resolver asserts the
/// leaf shape on entry as the canary if the invariant ever weakens.
inline constexpr double kSymbolCost = 1.0;

/// "Are all the variables this node needs already in scope?"
/// Used by the resolver going down the plan.
inline auto IsCompatible(SymbolSet const &required, SymbolSet const &provided) -> bool {
  return std::ranges::includes(provided, required);
}

/// "Does the input below this Bind actually use `sym`?"
/// Used by the cost model going up the plan.  If yes, this Bind has work
/// to do (alive: evaluate expr, introduce sym).  If no, the Bind is dead
/// weight and we'll skip it.
inline auto IsAlive(SymbolSet const &input_required, planner::core::EClassId sym) -> bool {
  return input_required.contains(sym);
}

/// Cost of the alive branch.  Pay for input, sym evaluation, and expr.
inline auto AliveCost(double input_cost, double sym_cost, double expr_cost) -> double {
  return input_cost + sym_cost + expr_cost;
}

/// Cost of the dead branch.  Only the input runs; sym and expr are skipped.
inline auto DeadCost(double input_cost) -> double { return input_cost; }

/// What variables does an alive Bind still need from above?
/// Take what the input still needed, drop `sym` (this Bind introduces
/// it), then add whatever `expr` references - because we're about to
/// evaluate `expr`, so its needs become this Bind's needs.
inline auto AliveRequired(SymbolSet const &input_required, planner::core::EClassId sym, SymbolSet const &expr_required)
    -> SymbolSet {
  boost::container::small_vector<planner::core::EClassId, 16> buf;
  buf.reserve(input_required.size() + expr_required.size());
  auto input_minus_sym = input_required | std::views::filter([sym](planner::core::EClassId id) { return id != sym; });
  std::ranges::set_union(input_minus_sym, expr_required, std::back_inserter(buf));
  return SymbolSet(boost::container::ordered_unique_range, buf.begin(), buf.end());
}

}  // namespace memgraph::query::plan::v2::bind
