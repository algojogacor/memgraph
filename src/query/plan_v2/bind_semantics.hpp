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
// Bind semantics: shared algebra for "compute expr, bind to sym, run input".
// ============================================================================
//
// Bind alive vs dead is a *resolution* decision (does input demand sym?); the
// cost model and resolver both consume this algebra.  Build-time consumers
// observe the resolver's outcome rather than re-deciding.

#include <algorithm>
#include <ranges>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2::bind {

/// Set of EClassIds — used for "demand" sets (`required`) and "provided"
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

/// `required ⊆ provided`: every symbol this alternative demands has been
/// bound by some ancestor in the resolver's `provided` context.
inline auto IsCompatible(SymbolSet const &required, SymbolSet const &provided) -> bool {
  return std::ranges::includes(provided, required);
}

/// Predicate — is the Bind alive given the input alt's demand set?
///
/// A Bind is *alive* when the input demands the symbol it would bind: the
/// expr must run, sym must be evaluated, the binding has work to do.
/// A Bind is *dead* when the input doesn't demand the symbol: expr is unused,
/// sym is unused, the Bind is a no-op pass-through over input.
inline auto IsAlive(SymbolSet const &input_required, planner::core::EClassId sym) -> bool {
  return input_required.contains(sym);
}

/// Cost of the alive branch.  Pay for input, sym evaluation, and expr.
inline auto AliveCost(double input_cost, double sym_cost, double expr_cost) -> double {
  return input_cost + sym_cost + expr_cost;
}

/// Cost of the dead branch.  Only the input runs; sym and expr are skipped.
inline auto DeadCost(double input_cost) -> double { return input_cost; }

/// Required-set algebra for the alive branch — `(input.required \ {sym}) ∪ expr.required`.
///
/// Removing `sym` reflects that the Bind itself supplies that symbol; whatever
/// `expr` demands flows up because the binding does not satisfy expr's needs.
inline auto AliveRequired(SymbolSet const &input_required, planner::core::EClassId sym, SymbolSet const &expr_required)
    -> SymbolSet {
  boost::container::small_vector<planner::core::EClassId, 16> buf;
  buf.reserve(input_required.size() + expr_required.size());
  auto input_minus_sym = input_required | std::views::filter([sym](planner::core::EClassId id) { return id != sym; });
  std::ranges::set_union(input_minus_sym, expr_required, std::back_inserter(buf));
  return SymbolSet(boost::container::ordered_unique_range, buf.begin(), buf.end());
}

}  // namespace memgraph::query::plan::v2::bind
