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

#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <span>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

namespace memgraph::planner::core::extract {

/// A dominance relation over Alt: a default-constructible binary callable
/// (Alt const&, Alt const&) -> std::partial_ordering with the convention:
///   less       - lhs is dominated by rhs (rhs is at least as good in all dims)
///   greater    - lhs dominates rhs
///   equivalent - both dominate each other (Pareto-equal: same cost, same demand)
///   unordered  - incomparable (neither dominates the other)
/// The dominance relation MUST be transitive - if a >= b and b >= c then
/// a >= c (with `>=` here meaning "dominates or is equivalent to"). prune()'s
/// early break relies on this property.
///
/// Returning a 3-way ordering (rather than two separate bool calls for the two
/// directions) lets each dominance functor compare the two alternatives in a
/// single pass over their per-alt state - typically a sorted required-set,
/// where the two-bool form would walk both sets twice.
template <typename Fn, typename Alt>
concept DominanceRelation =
    std::invocable<Fn, Alt const &, Alt const &> &&
    std::convertible_to<std::invoke_result_t<Fn, Alt const &, Alt const &>, std::partial_ordering> &&
    std::default_initializable<Fn>;

// ============================================================================
// Compositional Pareto comparison
// ============================================================================
// A dominance functor over Alt can be expressed as the Pareto-fold of one or
// more per-dimension comparators.  Each comparator answers: "for this single
// dimension, does a dominate b, get dominated by b, tie, or is the dim itself
// incomparable?"  pareto_fold combines them: agreement on direction (with at
// least one strict) means dominance; disagreement means incomparable.
//
// Convention (matches DominanceRelation):
//   less        - lhs is dominated
//   greater     - lhs dominates
//   equivalent  - tied
//   unordered   - incomparable

/// "Lower is better" comparator for any three-way-comparable type.  The natural
/// `<=>` says a < b ⇒ less, but in dominance terms "smaller" means "better"
/// means a dominates b ⇒ greater.  Swapping operands inverts the ordering
/// without an enum dance, and strong_ordering implicitly converts to
/// partial_ordering on return.
inline constexpr auto lower_is_better = [](auto const &a, auto const &b) -> std::partial_ordering { return b <=> a; };

/// "Higher is better" - direct `<=>`, no swap.
inline constexpr auto higher_is_better = [](auto const &a, auto const &b) -> std::partial_ordering { return a <=> b; };

/// "Smaller-by-inclusion is better" comparator for two sorted ranges.
/// Single forward merge over both ranges to determine the subset relations,
/// with mid-pass early-exit once both subset flags are false (incomparable).
inline constexpr auto smaller_subset_is_better = []<std::ranges::input_range R>(R const &a,
                                                                                R const &b) -> std::partial_ordering {
  auto it_a = std::ranges::begin(a);
  auto const end_a = std::ranges::end(a);
  auto it_b = std::ranges::begin(b);
  auto const end_b = std::ranges::end(b);
  bool a_subset_b = true;
  bool b_subset_a = true;
  while (it_a != end_a && it_b != end_b) {
    auto const cmp = *it_a <=> *it_b;
    if (std::is_lt(cmp)) {
      a_subset_b = false;  // *it_a is in a but not in b
      ++it_a;
    } else if (std::is_gt(cmp)) {
      b_subset_a = false;  // *it_b is in b but not in a
      ++it_b;
    } else {
      ++it_a;
      ++it_b;
    }
    if (!a_subset_b && !b_subset_a) break;
  }
  if (it_a != end_a) a_subset_b = false;
  if (it_b != end_b) b_subset_a = false;
  // a ⊆ b means a "needs less" → a is better → a dominates b → greater.
  if (a_subset_b && b_subset_a) return std::partial_ordering::equivalent;
  if (a_subset_b) return std::partial_ordering::greater;
  if (b_subset_a) return std::partial_ordering::less;
  return std::partial_ordering::unordered;
};

/// Lift a member-pointer + per-value comparator into a per-Alt dim function.
/// Usage: `dim<&Alt::cost>(lower_is_better)` yields a callable
/// `(Alt const&, Alt const&) -> std::partial_ordering`.
template <auto MemPtr, typename Cmp>
[[nodiscard]] constexpr auto dim(Cmp cmp) {
  return [cmp = std::move(cmp)](auto const &a, auto const &b) -> std::partial_ordering {
    return cmp(a.*MemPtr, b.*MemPtr);
  };
}

/// Fold per-dimension orderings into a single Pareto ordering.
///   - Any unordered dim → result is unordered.
///   - Disagreement (one less, another greater) → unordered.
///   - All equivalent → equivalent.
///   - Otherwise the agreed direction wins.
template <typename Alt, typename... Dims>
[[nodiscard]] auto pareto_fold(Alt const &a, Alt const &b, Dims const &...dims) -> std::partial_ordering {
  using std::partial_ordering;
  auto acc = partial_ordering::equivalent;
  auto step = [&](partial_ordering next) -> bool {
    if (next == partial_ordering::unordered) {  // dim itself incomparable
      acc = partial_ordering::unordered;
      return false;  // stop the && fold
    }
    if (next == partial_ordering::equivalent) return true;  // tied dim doesn't change direction
    if (acc == partial_ordering::equivalent) {              // first directional dim
      acc = next;
      return true;
    }
    if (acc != next) {  // direction conflict
      acc = partial_ordering::unordered;
      return false;
    }
    return true;
  };
  // Short-circuit fold: once step returns false, remaining dims aren't called.
  // Saves the set-merge in smaller_subset_is_better when an earlier scalar dim
  // already conflicts.
  (void)(step(dims(a, b)) && ...);
  return acc;
}

/// A combine function for ParetoFrontier::combine: produces a new Alt from a
/// pair of input Alts (one cartesian-product element).  Callable repeatedly
/// over the n × m product, so the operator must be const-callable on whatever
/// state the functor carries.
template <typename Fn, typename Alt>
concept Combiner = std::invocable<Fn const &, Alt const &, Alt const &> &&
                   std::convertible_to<std::invoke_result_t<Fn const &, Alt const &, Alt const &>, Alt>;

/// Generic Pareto frontier over alternatives of type Alt.
///
/// DominanceFn must satisfy `DominanceRelation<DominanceFn, Alt>` - see the
/// concept above for the transitivity contract.
///
/// CombineFn (used by combine/flat_map, not part of the type): (Alt, Alt) -> Alt
///   Produces a new alternative from two parent alternatives (Cartesian product element).
///
/// Example:
///   struct MyDominance {
///     static auto operator()(MyAlt const &a, MyAlt const &b) -> bool {
///       return b.cost <= a.cost && b.req.contains_all_of(a.req);
///     }
///   };
///   using Frontier = ParetoFrontier<MyAlt, MyDominance>;
template <typename Alt, typename DominanceFn>
  requires std::copyable<Alt> && DominanceRelation<DominanceFn, Alt>
struct ParetoFrontier {
  ParetoFrontier() = default;

  /// Construct from an unpruned list of alternatives.  Prunes on construction
  /// so the resulting frontier satisfies the Pareto invariant.  This is the
  /// only way to seed a frontier from raw data; flat_map / combine /
  /// merge_in_place are the compositional alternatives.
  explicit ParetoFrontier(std::vector<Alt> alts) : alts_(std::move(alts)) { prune(); }

  /// Read-only view over the (Pareto-pruned) alternatives.  Returning span keeps
  /// the storage choice (currently std::vector) out of the public contract.
  [[nodiscard]] auto alts() const noexcept -> std::span<Alt const> { return alts_; }

  /// In-place mutation that the caller promises preserves the Pareto invariant.
  /// Calls fn(alt) on each surviving alt; no re-prune is performed.
  ///
  /// Contract: fn must not change relative ordering under DominanceFn - i.e.,
  /// for any two alts A and B in the frontier, the truth value of
  /// DominanceFn{}(A, B) must be the same after fn(A) and fn(B) as before.
  /// Adding a uniform constant to a `cost` field, or rewriting a field that
  /// dominance does not read (e.g., enode_id), both satisfy this contract.
  /// Mutations that could flip dominance must go through flat_map / merge /
  /// combine instead, which re-prune.
  template <typename Fn>
    requires std::invocable<Fn, Alt &>
  void mutate_pruning_invariant_preserving(Fn &&fn) {
    for (auto &alt : alts_) fn(alt);
  }

  /// Flat-map: for each alternative, produce zero or more new alternatives via a callback,
  /// collect into a new frontier, then prune. This is the general pattern for
  /// per-alt conditional emission.
  /// @param fn  (Alt const&, auto emit) -> void - calls emit(Alt&&) to produce output alternatives.
  template <typename Fn>
  [[nodiscard]] static auto flat_map(ParetoFrontier const &input, Fn &&fn) -> ParetoFrontier {
    auto out = std::vector<Alt>{};
    out.reserve(input.alts_.size());  // heuristic: at least one output per input
    for (auto const &alt : input.alts_) {
      fn(alt, [&](Alt &&v) { out.push_back(std::move(v)); });
    }
    return ParetoFrontier{std::move(out)};
  }

  /// Union another frontier into this one and re-prune.  Both `*this` and
  /// `other` are already Pareto-pruned; only cross-pairs and within-suffix
  /// pairs need checking (see prune_with_pruned_prefix).  `other`'s alts
  /// are moved-from on return.
  void merge_in_place(ParetoFrontier &&other) {
    auto const pruned_prefix = alts_.size();
    alts_.reserve(pruned_prefix + other.alts_.size());
    alts_.insert(alts_.end(), std::make_move_iterator(other.alts_.begin()), std::make_move_iterator(other.alts_.end()));
    prune_with_pruned_prefix(pruned_prefix);
  }

  /// Cartesian product of two frontiers. For each (l, r) pair, calls combine_fn(l, r)
  /// to produce a new alternative, then prunes the result.
  template <typename CombineFn>
    requires Combiner<CombineFn, Alt>
  [[nodiscard]] static auto combine(ParetoFrontier const &lhs, ParetoFrontier const &rhs, CombineFn &&combine_fn)
      -> ParetoFrontier {
    auto out = std::vector<Alt>{};
    out.reserve(lhs.alts_.size() * rhs.alts_.size());
    for (auto const &l : lhs.alts_) {
      for (auto const &r : rhs.alts_) {
        out.push_back(combine_fn(l, r));
      }
    }
    return ParetoFrontier{std::move(out)};
  }

 private:
  // std::vector chosen so ParetoFrontier moves stay pointer-swap; small_vector's
  // element-wise move dominates here because Alt is large with a non-trivial move.
  std::vector<Alt> alts_;

  /// Remove alternatives dominated by any other alternative in the frontier.
  /// Two-pass: first mark dominated indices, then erase. This avoids reading
  /// moved-from elements (std::erase_if/remove_if moves elements during its pass).
  /// Private - there is no path to seed an unpruned frontier from outside, so
  /// external prune() calls would always be no-ops.
  void prune() { prune_with_pruned_prefix(0); }

  /// Prune assuming `alts_[0..pruned_prefix)` is already Pareto-pruned: skip
  /// pair checks that lie entirely within the pruned prefix.  Used by
  /// `merge_in_place` where two pre-pruned sets are concatenated; pruned_prefix
  /// is 0 for a from-scratch prune.
  ///
  /// For each ordered pair (i, j) we need to check, j ranges over:
  ///   - i in prefix [0, pruned_prefix):  j in suffix [pruned_prefix, n)   (cross-pairs)
  ///   - i in suffix [pruned_prefix, n):  j in (i, n)                      (within-suffix)
  /// That is `j_start(i) = max(pruned_prefix, i + 1)`, which collapses both
  /// passes into one loop.
  void prune_with_pruned_prefix(size_t pruned_prefix) {
    auto const n = alts_.size();
    if (n <= pruned_prefix) return;  // nothing newly added; no new pairs
    if (n <= 1) return;              // 0 or 1 alt total: no pairs at all

    // SBO buffer for the dominated-flag array.  Frontiers rarely exceed 64
    // alternatives after pruning; larger ones fall back to heap.
    boost::container::small_vector<bool, 64> dominated(n, false);

    for (size_t i = 0; i < n; ++i) {
      if (dominated[i]) continue;
      auto const j_start = std::max(pruned_prefix, i + 1);
      for (size_t j = j_start; j < n; ++j) {
        if (dominated[j]) continue;
        auto const cmp = DominanceFn{}(alts_[i], alts_[j]);
        if (cmp == std::partial_ordering::less || cmp == std::partial_ordering::equivalent) {
          // i dominated by j (or Pareto-equal: drop one, keep j by convention).
          dominated[i] = true;
          // Transitivity break - see prune() / DominanceRelation concept.
          break;
        }
        if (cmp == std::partial_ordering::greater) {
          dominated[j] = true;
        }
        // unordered: keep both, continue scanning.
      }
    }

    // Compact survivors in place.
    size_t write = 0;
    for (size_t read = 0; read < n; ++read) {
      if (dominated[read]) continue;
      if (write != read) alts_[write] = std::move(alts_[read]);
      ++write;
    }
    alts_.resize(write);
  }
};

/// Concept for alternatives usable with CostResultBase.  Beyond what
/// `cost` must be a non-static data member (not a property/function);
/// resolve projects via `&Alt::cost`.
template <typename Alt>
concept ParetoAlt = std::copyable<Alt> && requires(Alt const &a) {
  { a.cost } -> std::totally_ordered;
  { a.enode_id };
};

/// Base for ParetoFrontier types that use min-cost as their resolve/min_cost
/// strategy.  Derived types get resolve(), min_cost(), and convenience
/// constructors for free.  The Self type for the merge return / *this is
/// deduced via C++23 explicit object parameters; derived classes do not pass
/// themselves through a CRTP template parameter.  Alt must have `.cost`
/// (double-compatible) and `.enode_id` fields.
template <typename Alt, typename DominanceFn>
  requires ParetoAlt<Alt>
struct CostResultBase : ParetoFrontier<Alt, DominanceFn> {
  using Base = ParetoFrontier<Alt, DominanceFn>;
  using Base::Base;
  using cost_t = decltype(Alt::cost);

  CostResultBase() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  CostResultBase(Base base) : Base(std::move(base)) {}

  /// Initializer-list construction prunes on construction.
  CostResultBase(std::initializer_list<Alt> init) : Base(std::vector<Alt>(init)) {}

  /// Vector construction prunes on construction.  Sibling of the
  /// initializer-list ctor for callers that build alts at runtime.
  // NOLINTNEXTLINE(google-explicit-constructor)
  CostResultBase(std::vector<Alt> alts) : Base(std::move(alts)) {}

  template <typename Self>
  auto resolve(this Self const &self) -> std::pair<decltype(Alt::enode_id), cost_t const &> {
    auto const alts = self.alts();
    auto it = std::ranges::min_element(alts, {}, &Alt::cost);
    assert(it != alts.end() && "resolve called on empty frontier");
    return {it->enode_id, it->cost};
  }
};

}  // namespace memgraph::planner::core::extract
