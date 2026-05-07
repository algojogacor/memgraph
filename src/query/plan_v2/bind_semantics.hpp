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
//
// Two opposing flows of symbol-eclass sets, both stored as SymbolSet:
//
//   Demand (`required`) - propagates BOTTOM-UP in the cost model.
//     Identifier(sym) seeds {sym}.  Expression operators union demands
//     from their children.  An *alive* Bind removes its `sym` from the
//     up-propagating demand because the Bind itself supplies that sym.
//     A *dead* Bind passes its input's demand through unchanged (expr
//     wasn't evaluated, so expr's demand doesn't matter).
//
//   Provided - propagates TOP-DOWN in the resolver.
//     Starts empty at the root.  Each alive Bind adds its `sym` to
//     `provided` when descending into the input child.  At every chosen
//     alt, the resolver checks `required ⊆ provided`: every symbol the
//     alt demands has been supplied by a Bind on the resolver's current
//     path from root.  If not, no compatible alt exists -> extraction
//     fails (or the resolver tries another alt with a smaller `required`).
//
// The cost model and resolver meet at the alt: the cost model produces
// alts with `required` set; the resolver picks an alt whose `required`
// is a subset of the `provided` it has accumulated.

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

/// Top-down compatibility check used by the resolver.
/// `required ⊆ provided`: every symbol this alternative demands (bottom-up)
/// has been supplied by some Bind ancestor in the resolver's accumulated
/// `provided` context (top-down).
inline auto IsCompatible(SymbolSet const &required, SymbolSet const &provided) -> bool {
  return std::ranges::includes(provided, required);
}

/// Bottom-up alive predicate used by the cost model.
///
/// A Bind is *alive* when the input's demand set contains `sym`: the input
/// subtree (in this structural form) needs `sym`, so this Bind must run -
/// expr is evaluated, sym takes its value.
/// A Bind is *dead* when the input does not demand `sym`: expr is unused,
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

/// Demand-set algebra for the alive branch - `(input.required \ {sym}) ∪ expr.required`.
///
/// Bottom-up flow at the alive Bind:
///   - input demanded `sym`; Bind supplies it, so `sym` drops out of the
///     up-propagating demand.
///   - expr is evaluated, so its own demands now ride upward (some Bind
///     above must supply whatever expr referenced via Identifier).
inline auto AliveRequired(SymbolSet const &input_required, planner::core::EClassId sym, SymbolSet const &expr_required)
    -> SymbolSet {
  boost::container::small_vector<planner::core::EClassId, 16> buf;
  buf.reserve(input_required.size() + expr_required.size());
  auto input_minus_sym = input_required | std::views::filter([sym](planner::core::EClassId id) { return id != sym; });
  std::ranges::set_union(input_minus_sym, expr_required, std::back_inserter(buf));
  return SymbolSet(boost::container::ordered_unique_range, buf.begin(), buf.end());
}

}  // namespace memgraph::query::plan::v2::bind
