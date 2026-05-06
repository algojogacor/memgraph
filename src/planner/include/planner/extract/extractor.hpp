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

#include <concepts>
#include <deque>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <cassert>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "planner/extract/pareto_frontier.hpp"

import memgraph.planner.core.egraph;

namespace memgraph::planner::core::extract {

// ============================================================================
// CostResult contract
// ============================================================================
//
// Every cost model defines:
//   using CostResult = ...;
//   operator()(ENode const &, ENodeId, span<CostResult>) -> CostResult
//
// `children` is a mutable span: cost models may move-from individual entries
// to consume child frontiers in place.  The extractor does not reuse the span
// after the cost-model call returns, so consumed entries are safely discarded.
//
// CostResult must satisfy CostResultType (defined below).  ParetoFrontier-based
// cost models derive from CostResultBase (planner/extract/pareto_frontier.hpp).
// DefaultCostResult<T> is the scalar reference adapter.

/// CostResult contract — enforced at compile time.
/// Every CostResult type must provide:
///   cost_t                 — the scalar cost type (must be totally_ordered)
///   a.merge(b)             — combine frontiers
///   a.resolve()            — pick the best enode
///   a.min_cost()           — extract the comparable cost (asserts non-empty for frontiers)
///   a.resolve_with_cost()  — paired (enode_id, cost) so callers needing both
///                            avoid scanning the frontier twice
template <typename CR>
concept CostResultType = std::copyable<CR> && requires(CR const &a, CR const &b) {
  typename CR::cost_t;
  requires std::totally_ordered<typename CR::cost_t>;
  { a.merge(b) } -> std::same_as<CR>;
  { a.resolve() } noexcept -> std::same_as<ENodeId>;
  { a.min_cost() } noexcept -> std::same_as<typename CR::cost_t>;
  { a.resolve_with_cost() } -> std::same_as<std::pair<ENodeId, typename CR::cost_t>>;
};

/// Default scalar CostResult — wraps a cost value with enode metadata.
/// Reference adapter for cost models that don't need Pareto frontiers.
template <std::totally_ordered T>
struct DefaultCostResult {
  using cost_t = T;

  T cost;
  ENodeId enode_id;

  [[nodiscard]] auto merge(DefaultCostResult const &other) const -> DefaultCostResult {
    return cost <= other.cost ? *this : other;
  }

  [[nodiscard]] auto resolve_with_cost() const -> std::pair<ENodeId, cost_t> { return {enode_id, cost}; }

  [[nodiscard]] auto resolve() const noexcept -> ENodeId { return enode_id; }

  [[nodiscard]] auto min_cost() const noexcept -> cost_t { return cost; }
};

// ============================================================================
// Extraction pipeline types
// ============================================================================

/// Per-eclass frontier during cost propagation.
/// nullopt means "in progress" (cycle detection).
template <typename CostResult>
using EClassFrontier = std::optional<CostResult>;

/// Map from EClassId to its computed frontier.  Part of the Resolver contract:
/// resolvers receive `FrontierMap<CR> const &` after ComputeFrontiers populates it.
template <typename CostResult>
using FrontierMap = boost::unordered_flat_map<EClassId, EClassFrontier<CostResult>>;

/// Selection: one enode chosen per eclass, with its cost.
template <typename CostType>
struct Selection {
  ENodeId enode_id;
  CostType cost;
};

/// Map from EClassId to the enode chosen by a Resolver, with its cost.
template <typename CostType>
using SelectionMap = boost::unordered_flat_map<EClassId, Selection<CostType>>;

// ============================================================================
// Resolver contract
// ============================================================================
//
// A Resolver is a stateless functor `r(egraph, frontier_map, root, out)` that
// fills `out` with a SelectionMap.  It chooses one enode per eclass and decides
// which of that enode's children are part of the extracted tree.
//
// Caller-clears: Extract() calls ctx.clear() before invoking the resolver, so
// `out` is empty on entry.  Direct callers (e.g. unit tests) must do the same.
//
// Contract on the populated SelectionMap:
//   - root is in the map.
//   - For each (id, sel) in the map, sel.enode_id is a valid enode in eclass id.
//   - For each (id, sel) in the map, every child of sel.enode_id that the
//     resolver wishes to be part of the extracted tree is also in the map.
//   - Children absent from the map are deliberately excluded ("dead").
//
// Downstream stages (CollectDependencies, TopologicalSort) skip absent children
// — that is how the contract surfaces in the rest of the pipeline.
//
// Two production adapters:
//   * DefaultResolver: walks all children of the chosen enode.
//   * A context-aware variant downstream that honours alive/dead semantics
//     and re-resolves shared eclasses on incompatible re-visits.

template <typename R, typename Symbol, typename Analysis, typename CostResult>
concept Resolver =
    CostResultType<CostResult> && std::invocable<R, EGraph<Symbol, Analysis> const &, FrontierMap<CostResult> const &,
                                                 EClassId, SelectionMap<typename CostResult::cost_t> &>;

/// Generic Resolver that selects each eclass via CostResult::resolve_with_cost
/// and walks every child of the chosen enode.  Safe for any cost model whose
/// children are unconditionally part of the extracted tree.
///
/// NOT safe for cost models where a chosen alt may exclude some of its
/// enode's children from the extracted tree.  Those need a context-aware
/// resolver.
struct DefaultResolver {
  template <typename Symbol, typename Analysis, CostResultType CostResult>
  void operator()(EGraph<Symbol, Analysis> const &egraph, FrontierMap<CostResult> const &frontier_map, EClassId root,
                  SelectionMap<typename CostResult::cost_t> &out) const {
    auto to_visit = std::vector{root};
    auto visited = boost::unordered_flat_set{root};

    while (!to_visit.empty()) {
      auto current = to_visit.back();
      to_visit.pop_back();

      auto it = frontier_map.find(current);
      assert(it != frontier_map.end() && it->second.has_value());

      auto const &frontier = *it->second;
      auto [enode_id, cost] = frontier.resolve_with_cost();
      out.try_emplace(current, enode_id, cost);

      auto const &enode = egraph.get_enode(enode_id);
      for (auto child : enode.children()) {
        if (visited.insert(child).second) {
          to_visit.push_back(child);
        }
      }
    }
  }
};

// ============================================================================
// Extraction stages
// ============================================================================
// Convenience callers should use Extract().  The four stages below are also
// public: per-stage tests and callers that interleave their own work between
// ComputeFrontiers and resolve compose them directly (see
// ConvertToLogicalOperator in plan_v2, which validates root satisfiability
// between ComputeFrontiers and the resolver).

/// In-degree map for topological sorting.
using InDegreeMap = boost::unordered_flat_map<EClassId, int>;

/// Bottom-up cost propagation. Calls cost_model(enode, enode_id, children) for each enode,
/// merges results via CostResult::merge across enodes in the same eclass.
template <typename Symbol, typename Analysis, typename CostModel>
  requires CostResultType<typename CostModel::CostResult>
[[nodiscard]] auto ComputeFrontiers(EGraph<Symbol, Analysis> const &egraph, CostModel const &cost_model,
                                    EClassId eclass_id, FrontierMap<typename CostModel::CostResult> &frontier_map)
    -> std::optional<typename CostModel::CostResult> {
  using CostResult = CostModel::CostResult;

  assert(!egraph.needs_rebuild() && "egraph must be rebuilt before extraction");

  if (auto const it = frontier_map.find(eclass_id); it != frontier_map.end()) {
    return it->second;
  }

  auto const &eclass = egraph.eclass(eclass_id);

  // Mark this e-class as "in progress" with nullopt frontier to detect cycles
  frontier_map.emplace(eclass_id, std::nullopt);

  auto merged_frontier = std::optional<CostResult>{};

  auto children_frontiers = std::vector<CostResult>{};
  for (auto const &enode_id : eclass.nodes()) {
    auto const &enode = egraph.get_enode(enode_id);
    children_frontiers.clear();
    auto has_cyclic_child = false;
    for (auto child : enode.children()) {
      auto frontier = ComputeFrontiers(egraph, cost_model, child, frontier_map);
      if (!frontier) {
        has_cyclic_child = true;
        // N.B. intentionally no break — continue processing remaining children
        // so their costs are computed and cached for other extraction paths
      } else {
        children_frontiers.emplace_back(std::move(*frontier));
      }
    }
    if (has_cyclic_child) continue;

    auto enode_frontier = cost_model(enode, enode_id, children_frontiers);

    if (!merged_frontier) {
      merged_frontier = std::move(enode_frontier);
    } else {
      merged_frontier = std::move(*merged_frontier).merge(std::move(enode_frontier));
    }
  }

  if (merged_frontier) {
    frontier_map[eclass_id] = *merged_frontier;
    return merged_frontier;
  }

  // All enodes cyclic — remove sentinel
  frontier_map.erase(eclass_id);
  return std::nullopt;
}

/// Scratch buffer set used by the BFS in CollectDependencies. Owned by
/// ExtractionContext so that warm Extract() calls don't reallocate.
struct DependencyScratch {
  std::vector<EClassId> bfs;
  boost::unordered_flat_set<EClassId> visited;

  void clear() {
    bfs.clear();
    visited.clear();
  }
};

template <typename Symbol, typename Analysis, typename CostResult>
void CollectDependencies(EGraph<Symbol, Analysis> const &egraph, SelectionMap<CostResult> const &enode_selection,
                         EClassId root, InDegreeMap &out, DependencyScratch &scratch) {
  out.emplace(root, 0);
  scratch.bfs.push_back(root);
  scratch.visited.insert(root);
  scratch.bfs.reserve(enode_selection.size());
  scratch.visited.reserve(enode_selection.size());

  // Non-recursive BFS search
  while (!scratch.bfs.empty()) {
    auto curr = scratch.bfs.back();
    scratch.bfs.pop_back();

    auto enode_it = enode_selection.find(curr);
    assert(enode_it != enode_selection.end() && "all reachable EClasses should have selected ENode");

    auto const &enode = egraph.get_enode(enode_it->second.enode_id);
    for (auto child : enode.children()) {
      // Only walk children present in the selection (Resolver contract:
      // absent children are deliberately excluded).
      if (!enode_selection.contains(child)) continue;
      ++out[child];
      if (scratch.visited.insert(child).second) {
        scratch.bfs.emplace_back(child);
      }
    }
  }
}

/// Kahn's topological sort.  `in_degree` is consumed in place — its counts are
/// decremented to zero by the algorithm; on return its contents are unspecified
/// from the caller's perspective.  `out` and `ready` are filled (caller-clears).
template <typename Symbol, typename Analysis, typename CostResult>
void TopologicalSort(EGraph<Symbol, Analysis> const &egraph, SelectionMap<CostResult> const &enode_selection,
                     InDegreeMap &in_degree, std::vector<std::pair<EClassId, ENodeId>> &out,
                     std::deque<EClassId> &ready) {
  auto const expected = in_degree.size();
  out.reserve(expected);

  for (auto const &[eclass, degree] : in_degree)
    if (degree == 0) ready.push_back(eclass);

  while (!ready.empty()) {
    auto current = ready.front();
    ready.pop_front();

    auto it = enode_selection.find(current);
    assert(it != enode_selection.end() && "all reachable EClasses should have selected ENode");

    auto enode_id = it->second.enode_id;
    out.emplace_back(current, enode_id);

    auto const &enode = egraph.get_enode(enode_id);
    for (EClassId child : enode.children()) {
      auto deg_it = in_degree.find(child);
      if (deg_it == in_degree.end()) continue;  // resolver excluded child — see Resolver contract
      if (--deg_it->second == 0) {
        ready.push_back(child);
      }
    }
  }

  // Post-condition: all nodes must have been emitted. If not, the input contained a cycle,
  // which means an upstream stage (ComputeFrontiers or the Resolver) admitted a cyclic
  // dependency into the resolved selection — a bug in that stage.
  assert(out.size() == expected &&
         "TopologicalSort: cycle detected — resolved selection is not a DAG; "
         "check ComputeFrontiers and the Resolver for upstream bug");
}

// ============================================================================
// Extract — single deep entry point
// ============================================================================
//
// The pipeline (frontier-build → resolve → collect-deps → topo-sort) lives
// here as one function so the order, the invariants, and the contract between
// stages are all in one place.  Two customisation points: the `cost_model`
// (CostResultType) and the `resolver` (Resolver).

/// Caller-owned buffer for stage state, reused across Extract() calls.
///
/// All four output buffers (frontier_map, selection, in_degree, order) and the
/// two scratch buffers (deps, ready) are passed by reference into the pipeline
/// stages, which fill them in place.  clear() preserves capacity so that warm
/// Extract() calls allocate only when growing past the high-water mark.
template <CostResultType CostResult>
struct ExtractionContext {
  FrontierMap<CostResult> frontier_map;
  SelectionMap<typename CostResult::cost_t> selection;
  InDegreeMap in_degree;
  std::vector<std::pair<EClassId, ENodeId>> order;
  DependencyScratch deps;
  std::deque<EClassId> ready;

  void clear() {
    frontier_map.clear();
    selection.clear();
    in_degree.clear();
    order.clear();
    deps.clear();
    ready.clear();
  }
};

/// View over ExtractionContext-owned storage.  Valid until the next Extract()
/// call on the same context.
template <CostResultType CostResult>
struct ExtractView {
  std::span<std::pair<EClassId, ENodeId> const> order;
  CostResult::cost_t root_cost;
};

/// Primary entry point.  Caller owns `ctx`; the returned view points into
/// ctx-owned storage and is valid until the next Extract() call on `ctx`.
template <typename Symbol, typename Analysis, typename CostModel, typename ResolverFn>
  requires CostResultType<typename CostModel::CostResult> &&
           Resolver<ResolverFn, Symbol, Analysis, typename CostModel::CostResult>
[[nodiscard]] auto Extract(EGraph<Symbol, Analysis> const &egraph, EClassId root, CostModel const &cost_model,
                           ResolverFn resolver, ExtractionContext<typename CostModel::CostResult> &ctx)
    -> ExtractView<typename CostModel::CostResult> {
  using CostResult = CostModel::CostResult;

  ctx.clear();

  // Stage 1: bottom-up cost propagation.  Reserve up-front so the recursive
  // descent doesn't re-hash as eclasses are inserted.
  ctx.frontier_map.reserve(egraph.num_classes());
  (void)ComputeFrontiers(egraph, cost_model, root, ctx.frontier_map);

  // Stage 2: top-down resolution.  Resolver is responsible for the contract
  // documented above (chosen-coverage selection map).
  resolver(egraph, ctx.frontier_map, root, ctx.selection);

  // Stage 3: count in-degrees over the resolver-chosen child set.
  CollectDependencies(egraph, ctx.selection, root, ctx.in_degree, ctx.deps);

  // Stage 4: topological sort.  Consumes ctx.in_degree in place.
  TopologicalSort(egraph, ctx.selection, ctx.in_degree, ctx.order, ctx.ready);

  auto root_cost = typename CostResult::cost_t{};
  if (auto it = ctx.selection.find(root); it != ctx.selection.end()) {
    root_cost = it->second.cost;
  }

  return ExtractView<CostResult>{ctx.order, root_cost};
}

}  // namespace memgraph::planner::core::extract
