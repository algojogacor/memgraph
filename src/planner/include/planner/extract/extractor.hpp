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
#include <span>
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
//   operator()(ENode const &, ENodeId, span<CostResult const * const>) -> CostResult
//
// `children` is a span of pointers into `frontier_map`; the pointees are
// read-only.  Cost models that need to consume a child frontier (e.g. pass
// it by value into a helper that takes by-value) must copy from the pointee.
// Extracting via pointers avoids per-call copies for read-only cost models.
//
// CostResult must satisfy CostResultType (defined below).  ParetoFrontier-based
// cost models derive from CostResultBase (planner/extract/pareto_frontier.hpp).

/// CostResult contract - enforced at compile time.
/// Every CostResult type must provide:
///   cost_t                  - the scalar cost type (must be totally_ordered)
///   a.merge_in_place(b)     - fold `b` (an rvalue) into `a`, mutating `a`
///   a.resolve()             - paired (enode_id, cost-by-const-ref) of the
///                             chosen alternative; the reference is valid
///                             for the lifetime of the frontier
template <typename CR>
concept CostResultType = std::copyable<CR> && requires(CR &a, CR &&r, CR const &c) {
  typename CR::cost_t;
  requires std::totally_ordered<typename CR::cost_t>;
  { a.merge_in_place(std::move(r)) };
  { c.resolve() } -> std::same_as<std::pair<ENodeId, typename CR::cost_t const &>>;
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
// Precondition: `out` is empty on entry.
//
// Contract on the populated SelectionMap:
//   - root is in the map.
//   - For each (id, sel) in the map, sel.enode_id is a valid enode in eclass id.
//   - For each (id, sel) in the map, every child of sel.enode_id that the
//     resolver wishes to be part of the extracted tree is also in the map.
//   - Children absent from the map are deliberately excluded ("dead").

template <typename R, typename Symbol, typename Analysis, typename CostResult>
concept Resolver =
    CostResultType<CostResult> && std::invocable<R, EGraph<Symbol, Analysis> const &, FrontierMap<CostResult> const &,
                                                 EClassId, SelectionMap<typename CostResult::cost_t> &>;

// ============================================================================
// Extraction stages
// ============================================================================
// Most callers should use Extract().  The individual stages are public for
// callers that need to interleave their own work between them.

/// In-degree map for topological sorting.
using InDegreeMap = boost::unordered_flat_map<EClassId, int>;

/// Pool of children-frontier buffers used by ComputeFrontiers' recursive
/// descent.  Each recursive frame acquires one buffer; nested frames acquire
/// fresh buffers further down the pool, then release them on return.  A single
/// shared buffer would not work because the recursive call to ComputeFrontiers
/// happens inside the per-enode loop, which would clobber the caller frame's
/// buffer.  Capacity of inner vectors persists across queries because the
/// outer pool is owned by ExtractionContext.
template <CostResultType CostResult>
struct FrontierBufferPool {
 private:
  /// Acquire a buffer for the current recursive frame.  Cleared on entry but
  /// keeps its capacity from previous uses at the same depth.
  auto internal_acquire() -> std::vector<CostResult const *> & {
    if (depth == pool.size()) pool.emplace_back();
    auto &buf = pool[depth++];
    buf.clear();
    return buf;
  }

  void internal_release() noexcept { --depth; }

 public:
  struct [[nodiscard]] AcquiredGuard {
    FrontierBufferPool *owner;

    explicit AcquiredGuard(FrontierBufferPool &p) : owner{&p} {}

    ~AcquiredGuard() { owner->internal_release(); }

    AcquiredGuard(AcquiredGuard const &) = delete;
    AcquiredGuard(AcquiredGuard &&) = delete;
    auto operator=(AcquiredGuard const &) -> AcquiredGuard & = delete;
    auto operator=(AcquiredGuard &&) -> AcquiredGuard & = delete;
  };

  auto acquire() -> std::pair<AcquiredGuard, std::vector<CostResult const *> &> {
    return {std::piecewise_construct, std::forward_as_tuple(*this), std::forward_as_tuple(internal_acquire())};
  }

  void clear() noexcept { depth = 0; }

 private:
  // std::deque so references into pool entries remain valid when the pool
  // grows during a deeper recursive frame's acquire().  std::vector would
  // reallocate and dangle the outer caller's children_frontiers reference.
  std::deque<std::vector<CostResult const *>> pool;
  size_t depth = 0;
};

template <typename CostResult>
struct FrontierContext {
  FrontierMap<CostResult> frontier_map;
  FrontierBufferPool<CostResult> frontier_buffer_pool;

  void clear() {
    frontier_map.clear();
    frontier_buffer_pool.clear();
  }
};

/// Bottom-up cost propagation. Calls cost_model(enode, enode_id, children) for each enode,
/// merges results via CostResult::merge across enodes in the same eclass.
///
/// Returns a pointer into `frontier_map`. nullptr means the eclass is cyclic
/// (either fully unreachable or in-progress on the current recursion path).
/// The pointer is valid until the next mutation of `frontier_map` by the
/// caller; ComputeFrontiers itself never invalidates the returned pointer
/// across recursive calls, since each call re-finds before returning.
template <typename Symbol, typename Analysis, typename CostModel>
  requires CostResultType<typename CostModel::CostResult>
[[nodiscard]] auto ComputeFrontiers(EGraph<Symbol, Analysis> const &egraph, CostModel const &cost_model,
                                    EClassId eclass_id, FrontierContext<typename CostModel::CostResult> &ctx)
    -> CostModel::CostResult const * {
  using CostResult = CostModel::CostResult;

  assert(!egraph.needs_rebuild() && "egraph must be rebuilt before extraction");

  auto &out = ctx.frontier_map;

  if (auto const it = out.find(eclass_id); it != out.end()) {
    return it->second ? &*it->second : nullptr;
  }

  auto const &eclass = egraph.eclass(eclass_id);

  // Mark this e-class as "in progress" with nullopt frontier to detect cycles.
  // Iterator from this emplace is not retained: recursive calls below may
  // rehash frontier_map and invalidate it.
  out.emplace(eclass_id, std::nullopt);

  auto merged_frontier = std::optional<CostResult>{};

  // Acquire a per-frame buffer for collecting child-frontier pointers.  Each
  // recursive call further down acquires its own slot in the pool, so this
  // frame's buffer is not clobbered.
  auto [guard, children_frontiers] = ctx.frontier_buffer_pool.acquire();
  for (auto const &enode_id : eclass.nodes()) {
    auto const &enode = egraph.get_enode(enode_id);

    // Phase 1: recurse to populate frontier_map.  We don't keep the returned
    // pointers because subsequent recursive inserts may rehash and invalidate
    // them (boost::unordered_flat_map uses open addressing).
    for (auto child : enode.children()) {
      (void)ComputeFrontiers(egraph, cost_model, child, ctx);
    }

    // Phase 2: look up each child's frontier now that no further inserts will
    // happen in this enode's iteration.  Pointers obtained here remain valid
    // until the next mutation of frontier_map - which only occurs at the end
    // of this function (overwriting the sentinel for `eclass_id`), and that
    // does not trigger a rehash since the key already exists.
    children_frontiers.clear();
    children_frontiers.reserve(enode.children().size());
    auto has_cyclic_child = false;
    for (auto child : enode.children()) {
      auto it = out.find(child);
      if (it == out.end() || !it->second) {
        // Erased (fully cyclic) or in-progress sentinel (self/mutual cycle).
        // Cost model won't be called, no need to gather remaining children.
        has_cyclic_child = true;
        break;
      }
      children_frontiers.push_back(&*it->second);
    }
    if (has_cyclic_child) continue;

    auto enode_frontier = cost_model(enode, enode_id, children_frontiers);

    if (!merged_frontier) {
      merged_frontier = std::move(enode_frontier);
    } else {
      merged_frontier->merge_in_place(std::move(enode_frontier));
    }
  }

  // Re-find: the sentinel iterator from emplace above may have been
  // invalidated by rehashes during recursion.
  auto sentinel_it = out.find(eclass_id);
  assert(sentinel_it != out.end());
  if (merged_frontier) {
    sentinel_it->second = std::move(merged_frontier);
    return &*sentinel_it->second;
  }

  // All enodes cyclic - remove sentinel
  out.erase(sentinel_it);
  return nullptr;
}

/// Scratch buffers used by the dependency traversal in CollectDependencies.
/// Owned by ExtractionContext so that warm Extract() calls don't reallocate.
struct TraversalScratch {
  std::vector<EClassId> worklist;
  boost::unordered_flat_set<EClassId> visited;

  void clear() noexcept {
    worklist.clear();
    visited.clear();
  }
};

/// Contiguous FIFO queue for EClassIds.  Stores all elements in a single
/// vector; a front cursor advances instead of shifting elements.  Retains
/// buffer capacity across clear() calls, so warm TopologicalSort calls are
/// allocation-free after the first call's high-water mark is reached.
struct FifoQueue {
  void push_back(EClassId id) { buf_.push_back(id); }

  [[nodiscard]] auto front() const -> EClassId { return buf_[front_]; }

  void pop_front() { ++front_; }

  [[nodiscard]] auto empty() const -> bool { return front_ == buf_.size(); }

  void clear() noexcept {
    buf_.clear();
    front_ = 0;
  }

 private:
  std::vector<EClassId> buf_;
  std::size_t front_ = 0;
};

template <typename Symbol, typename Analysis, typename CostResult>
void CollectDependencies(EGraph<Symbol, Analysis> const &egraph, SelectionMap<CostResult> const &enode_selection,
                         EClassId root, TraversalScratch &scratch, InDegreeMap &out) {
  out.emplace(root, 0);
  auto const n = enode_selection.size();
  scratch.worklist.reserve(n);
  scratch.visited.reserve(n);
  scratch.worklist.push_back(root);
  scratch.visited.insert(root);

  // Iterative DFS traversal (LIFO worklist).
  while (!scratch.worklist.empty()) {
    auto curr = scratch.worklist.back();
    scratch.worklist.pop_back();

    auto enode_it = enode_selection.find(curr);
    assert(enode_it != enode_selection.end() && "all reachable EClasses should have selected ENode");

    auto const &enode = egraph.get_enode(enode_it->second.enode_id);
    for (auto child : enode.children()) {
      // Only walk children present in the selection (Resolver contract:
      // absent children are deliberately excluded).
      if (!enode_selection.contains(child)) continue;
      ++out[child];
      if (scratch.visited.insert(child).second) {
        scratch.worklist.emplace_back(child);
      }
    }
  }
}

/// Kahn's topological sort.  `in_degree` is consumed in place - its counts are
/// decremented to zero by the algorithm; on return its contents are unspecified
/// from the caller's perspective.  `out` and `ready` are filled (caller-clears).
template <typename Symbol, typename Analysis, typename CostResult>
void TopologicalSort(EGraph<Symbol, Analysis> const &egraph, SelectionMap<CostResult> const &enode_selection,
                     InDegreeMap &in_degree, FifoQueue &ready, std::vector<std::pair<EClassId, ENodeId>> &out) {
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
      if (deg_it == in_degree.end()) continue;  // resolver excluded child - see Resolver contract
      if (--deg_it->second == 0) {
        ready.push_back(child);
      }
    }
  }

  // Post-condition: all nodes must have been emitted. If not, the input contained a cycle,
  // which means an upstream stage (ComputeFrontiers or the Resolver) admitted a cyclic
  // dependency into the resolved selection - a bug in that stage.
  assert(out.size() == expected &&
         "TopologicalSort: cycle detected - resolved selection is not a DAG; "
         "check ComputeFrontiers and the Resolver for upstream bug");
}

// ============================================================================
// Extract - single deep entry point
// ============================================================================
//
// The pipeline (frontier-build → resolve → collect-deps → topo-sort) lives
// here as one function so the order, the invariants, and the contract between
// stages are all in one place.  Two customisation points: the `cost_model`
// (CostResultType) and the `resolver` (Resolver).

/// Caller-owned buffer for stage state, reused across Extract() calls.
///
/// All four output buffers (frontier_ctx, selection, in_degree, order) and the
/// two scratch buffers (deps, ready) are passed by reference into the pipeline
/// stages, which fill them in place.  clear() preserves capacity so that warm
/// Extract() calls allocate only when growing past the high-water mark.
template <CostResultType CostResult>
struct ExtractionContext {
  FrontierContext<CostResult> frontier_ctx;
  SelectionMap<typename CostResult::cost_t> selection;
  InDegreeMap in_degree;
  std::vector<std::pair<EClassId, ENodeId>> order;
  TraversalScratch deps;
  FifoQueue ready;

  void clear() noexcept {
    frontier_ctx.clear();
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
///
/// This entry point is intended for tests and benchmarks. Production code calls
/// ComputeFrontiers and the resolver directly via QueryPlannerContext.
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
  ctx.frontier_ctx.frontier_map.reserve(egraph.num_classes());
  (void)ComputeFrontiers(egraph, cost_model, root, ctx.frontier_ctx);

  // Stage 2: top-down resolution.  Resolver is responsible for the contract
  // documented above (chosen-coverage selection map).
  resolver(egraph, ctx.frontier_ctx.frontier_map, root, ctx.selection);

  // Stage 3: count in-degrees over the resolver-chosen child set.
  CollectDependencies(egraph, ctx.selection, root, ctx.deps, ctx.in_degree);

  // Stage 4: topological sort.  Consumes ctx.in_degree in place.
  TopologicalSort(egraph, ctx.selection, ctx.in_degree, ctx.ready, ctx.order);

  auto root_cost = typename CostResult::cost_t{};
  if (auto it = ctx.selection.find(root); it != ctx.selection.end()) {
    root_cost = it->second.cost;
  }

  return ExtractView<CostResult>{ctx.order, root_cost};
}

}  // namespace memgraph::planner::core::extract
