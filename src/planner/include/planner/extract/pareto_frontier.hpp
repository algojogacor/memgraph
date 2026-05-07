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
#include <concepts>
#include <span>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

namespace memgraph::planner::core::extract {

/// A dominance relation over Alt: a default-constructible binary callable
/// (Alt const&, Alt const&) -> convertible-to-bool returning true when the
/// first argument is dominated by the second. The relation MUST be transitive
/// - if dominates(a, b) and dominates(b, c) then dominates(a, c). prune()'s
/// early break relies on this property and on nothing else; reflexivity is
/// permitted (a may dominate itself), and antisymmetry is not required.
/// Reflexive duplicates resolve to "the later index in iteration order survives."
///
/// We use std::invocable (not std::predicate): std::predicate requires
/// regular_invocable which implies semantic equality preservation, but a
/// dominance relation is reflexive and not equality-preserving in that sense.
template <typename Fn, typename Alt>
concept DominanceRelation =
    std::invocable<Fn, Alt const &, Alt const &> &&
    std::convertible_to<std::invoke_result_t<Fn, Alt const &, Alt const &>, bool> && std::default_initializable<Fn>;

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

  /// Read-only view over the (Pareto-pruned) alternatives.  Returning span keeps
  /// the storage choice (currently std::vector) out of the public contract.
  [[nodiscard]] auto alts() const noexcept -> std::span<Alt const> { return alts_; }

  /// Construct from an unpruned list of alternatives.  Prunes on construction
  /// so the resulting frontier satisfies the Pareto invariant.  This is the
  /// only public way to seed a frontier from raw data; the static factories
  /// below (flat_map / combine) and merge_in_place handle compositional
  /// construction.
  [[nodiscard]] static auto from_unpruned(std::vector<Alt> alts) -> ParetoFrontier {
    auto result = ParetoFrontier{};
    result.alts_ = std::move(alts);
    result.prune();
    return result;
  }

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
    auto result = ParetoFrontier{};
    result.alts_.reserve(input.alts_.size());  // heuristic: at least one output per input
    for (auto const &alt : input.alts_) {
      fn(alt, [&](Alt &&out) { result.alts_.push_back(std::move(out)); });
    }
    result.prune();
    return result;
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
    auto result = ParetoFrontier{};
    result.alts_.reserve(lhs.alts_.size() * rhs.alts_.size());
    for (auto const &l : lhs.alts_) {
      for (auto const &r : rhs.alts_) {
        result.alts_.push_back(combine_fn(l, r));
      }
    }
    result.prune();
    return result;
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

  /// Prune assuming `alts_[0..pruned_prefix)` is already Pareto-pruned: avoids
  /// re-checking pairs within the already-pruned prefix.  Used by
  /// `merge_in_place` where two pre-pruned sets are concatenated - only
  /// cross-pairs and within-suffix pairs need checking.
  void prune_with_pruned_prefix(size_t pruned_prefix) {
    auto const n = alts_.size();
    if (n - pruned_prefix == 0) return;                        // nothing newly added
    if (pruned_prefix + 1 >= n && pruned_prefix == 0) return;  // 0 or 1 total alt
    // SBO buffer for the dominated-flag array. Frontiers rarely exceed 64
    // alternatives after pruning; larger ones fall back to heap.
    boost::container::small_vector<bool, 64> dominated(n, false);
    // Cross-checks: prefix × new (i in [0, pruned_prefix), j in [pruned_prefix, n))
    for (size_t i = 0; i < pruned_prefix; ++i) {
      if (dominated[i]) continue;
      for (size_t j = pruned_prefix; j < n; ++j) {
        if (dominated[j]) continue;
        if (DominanceFn{}(alts_[i], alts_[j])) {
          dominated[i] = true;
          break;
        }
        if (DominanceFn{}(alts_[j], alts_[i])) {
          dominated[j] = true;
        }
      }
    }
    // Within-new checks: (i, j) both >= pruned_prefix.  When pruned_prefix==0
    // this is the full O(N²) pass; when pruned_prefix==M it's the K² portion.
    for (size_t i = pruned_prefix; i < n; ++i) {
      if (dominated[i]) continue;
      for (size_t j = i + 1; j < n; ++j) {
        if (dominated[j]) continue;
        if (DominanceFn{}(alts_[i], alts_[j])) {
          dominated[i] = true;
          // Transitivity break - see prune() / DominanceRelation concept.
          break;
        }
        if (DominanceFn{}(alts_[j], alts_[i])) {
          dominated[j] = true;
        }
      }
    }
    size_t write = 0;
    for (size_t read = 0; read < n; ++read) {
      if (!dominated[read]) {
        if (write != read) alts_[write] = std::move(alts_[read]);
        ++write;
      }
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

  /// Initializer-list construction prunes on construction - no path to a
  /// non-pruned frontier from outside the class hierarchy.
  CostResultBase(std::initializer_list<Alt> init) : Base(Base::from_unpruned(std::vector<Alt>(init))) {}

  /// Vector construction prunes on construction.  Sibling of the
  /// initializer-list ctor for callers that build alts at runtime; replaces
  /// an explicit `from_unpruned` factory so callers don't have to repeat the
  /// derived type as a template argument.
  // NOLINTNEXTLINE(google-explicit-constructor)
  CostResultBase(std::vector<Alt> alts) : Base(Base::from_unpruned(std::move(alts))) {}

  template <typename Self>
  auto resolve(this Self const &self) -> std::pair<decltype(Alt::enode_id), cost_t const &> {
    auto const alts = self.alts();
    auto it = std::ranges::min_element(alts, {}, &Alt::cost);
    assert(it != alts.end() && "resolve called on empty frontier");
    return {it->enode_id, it->cost};
  }
};

}  // namespace memgraph::planner::core::extract
