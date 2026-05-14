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
// Full vocabulary (operator vs expression Alt kinds, `introduces` /
// `required` / `in_scope` / `must_introduce`, the construction-time
// absorption rule and the resolver's threading rule) lives in
// `src/query/plan_v2/CONTEXT.md`.  This header keeps only the data types
// (`SymbolSet`) and the small cost helpers (`AliveCost`, `DeadCost`,
// `kSymbolCost`) used by `egraph_converter.cpp`.

#include <algorithm>
#include <ranges>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2::bind {

/// Set of EClassIds - used for "demand" sets (`required`) and "provided"
/// contexts in the resolver.  flat_set + small_vector keeps small sets
/// (typical Cypher demand depth ≤ 8) heap-free.
///
/// We store EClassIds, but every id in here is the e-class of a Symbol
/// e-node, and each Symbol e-class corresponds to exactly one variable in
/// the query.  Two reasons that holds:
///   1. MakeSymbol(position, name) is hashconsed, so the same variable
///      always lands in the same e-class, and different variables land
///      in different ones.
///   2. No rewrite rule merges two Symbol e-classes together.
/// So treating an EClassId in this set as "a variable" is safe today.  If
/// someone later adds a rewrite that merges Symbol e-classes, that breaks
/// and IsAlive/IsCompatible will silently mix up different variables.
///
/// All operations (set algebra, compatibility checks, range construction)
/// are members rather than free functions — call sites read as
/// straightforward expressions on the set object.
class SymbolSet {
 public:
  using set_type = boost::container::flat_set<planner::core::EClassId, std::less<>,
                                              boost::container::small_vector<planner::core::EClassId, 8>>;

  SymbolSet() = default;

  SymbolSet(std::initializer_list<planner::core::EClassId> il) : set_(il) {}

  // Construct from an arbitrary range of EClassIds (sorts and deduplicates).
  template <std::ranges::input_range R>
    requires std::same_as<std::ranges::range_value_t<R>, planner::core::EClassId>
  explicit SymbolSet(R &&rng) {
    auto seq = set_.extract_sequence();
    for (auto id : rng) seq.push_back(id);
    std::ranges::sort(seq);
    seq.erase(std::ranges::unique(seq).begin(), seq.end());
    set_.adopt_sequence(boost::container::ordered_unique_range, std::move(seq));
  }

  // --- Access ---
  auto begin() const { return set_.begin(); }

  auto end() const { return set_.end(); }

  auto size() const { return set_.size(); }

  auto empty() const { return set_.empty(); }

  auto contains(planner::core::EClassId x) const { return set_.contains(x); }

  // --- Queries ---
  [[nodiscard]] bool is_alive(planner::core::EClassId sym) const { return contains(sym); }

  [[nodiscard]] bool is_compatible(SymbolSet const &provided) const { return std::ranges::includes(provided, *this); }

  // --- Mutation ---
  void insert(planner::core::EClassId x) { set_.insert(x); }

  template <typename Iter>
  void insert(Iter first, Iter last) {
    set_.insert(first, last);
  }

  void erase(planner::core::EClassId x) { set_.erase(x); }

  // --- Set algebra (return new SymbolSet, *this unchanged) ---
  auto set_union(SymbolSet const &other) const -> SymbolSet {
    SymbolSet out;
    auto seq = out.set_.extract_sequence();
    seq.reserve(set_.size() + other.set_.size());
    std::ranges::set_union(set_, other.set_, std::back_inserter(seq));
    out.set_.adopt_sequence(boost::container::ordered_unique_range, std::move(seq));
    return out;
  }

  auto difference(SymbolSet const &other) const -> SymbolSet {
    SymbolSet out;
    auto seq = out.set_.extract_sequence();
    seq.reserve(set_.size());
    std::ranges::set_difference(set_, other.set_, std::back_inserter(seq));
    out.set_.adopt_sequence(boost::container::ordered_unique_range, std::move(seq));
    return out;
  }

  auto difference_one(planner::core::EClassId x) const -> SymbolSet {
    SymbolSet out = *this;
    out.set_.erase(x);
    return out;
  }

  auto alive_required(planner::core::EClassId sym, SymbolSet const &expr_required) const -> SymbolSet {
    auto out = difference_one(sym);
    out.insert(expr_required.begin(), expr_required.end());
    return out;
  }

  // --- Raw sequence (for bulk construction) ---
  auto extract_sequence() { return set_.extract_sequence(); }

  void adopt_sequence(boost::container::ordered_unique_range_t, set_type::sequence_type &&seq) {
    set_.adopt_sequence(boost::container::ordered_unique_range, std::move(seq));
  }

  // --- Equality ---
  auto operator==(SymbolSet const &other) const -> bool = default;

 private:
  set_type set_;
};

/// Cost of a Symbol leaf alternative.
///
/// Invariant: every Symbol eclass has exactly one alternative
/// `{cost = kSymbolCost, required = ∅}`.  Cost-model and resolver collapse
/// the sym child to a scalar on this assumption; the resolver asserts the
/// leaf shape on entry as the canary if the invariant ever weakens.
inline constexpr double kSymbolCost = 1.0;

/// Cost of the alive branch.  Bind preserves input cardinality but evaluates
/// `expr` once per input row (Produce semantics in v1), so `expr_cost` is
/// scaled by `input_cardinality`.  For standalone Binds above Once
/// (cardinality 1) this collapses to `expr_cost`; for Binds inside a row
/// pipeline (above Unwind, scans) it correctly amortises per-row.
[[nodiscard]] inline auto AliveCost(double input_cost, double sym_cost, double expr_cost, double input_cardinality)
    -> double {
  return input_cost + sym_cost + input_cardinality * expr_cost;
}

/// Cost of the dead branch.  Only the input runs; sym and expr are skipped.
[[nodiscard]] inline auto DeadCost(double input_cost) -> double { return input_cost; }

}  // namespace memgraph::query::plan::v2::bind
