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
//   operator()(ENode const &, ENodeId, span<CostResult const * const> children) -> CostResult
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

/// Map from EClassId to its computed frontier.  Passed to the Resolver after
/// ComputeFrontiers populates it.
template <typename CostResult>
using FrontierMap = boost::unordered_flat_map<EClassId, EClassFrontier<CostResult>>;

// ============================================================================
// Extraction stages
// ============================================================================

/// Pool of children-frontier buffers used by ComputeFrontiers' recursive
/// descent.  Each recursive frame acquires one buffer; nested frames acquire
/// fresh buffers further down the pool, then release them on return.  A single
/// shared buffer would not work because the recursive call to ComputeFrontiers
/// happens inside the per-enode loop, which would clobber the caller frame's
/// buffer.  Capacity of inner vectors persists across queries because the
/// outer pool is owned by FrontierContext.
template <CostResultType CostResult>
struct FrontierBufferPool {
 private:
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
    auto has_uncomputable_child = false;
    for (auto child : enode.children()) {
      auto it = out.find(child);
      if (it == out.end() || !it->second) {
        // Erased (fully cyclic) or in-progress sentinel (self/mutual cycle).
        // Cost model won't be called, no need to gather remaining children.
        has_uncomputable_child = true;
        break;
      }
      children_frontiers.push_back(&*it->second);
    }
    if (has_uncomputable_child) continue;

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

// ============================================================================
// DfsPostOrder — resolver scaffolding
// ============================================================================
//
// Resolvers fill a `vector<Entry>` in children-before-parents order.  The
// caller supplies a `resolve(key, visit_child) -> Entry` callback that handles
// per-node logic (frontier lookup, alt selection, child-key dispatch); this
// function supplies the recursion, deduplication, and post-order emit.
//
// `seen` is owned by the caller for reuse across queries (clear() between
// calls).  `out` is also caller-owned and must be empty on entry.
//
// resolve signature: (Key key, auto visit_child) -> Entry
//   - Must call visit_child(child_key) for each child the entry depends on.
//   - May call visit_child zero times (leaf node).
//   - Must return the Entry to emit for `key` after all children are visited.

template <typename Key, typename KeyHash, typename Entry, typename ResolveFn>
void DfsPostOrder(Key root, boost::unordered_flat_set<Key, KeyHash> &seen, std::vector<Entry> &out,
                  ResolveFn &&resolve) {
  struct Recurse {
    boost::unordered_flat_set<Key, KeyHash> &seen;
    std::vector<Entry> &out;
    std::remove_reference_t<ResolveFn> &resolve;

    void operator()(Key key) {
      if (!seen.insert(key).second) return;
      auto entry = resolve(key, [this](Key child) { (*this)(std::move(child)); });
      out.push_back(std::move(entry));
    }
  };

  Recurse{seen, out, resolve}(std::move(root));
}

}  // namespace memgraph::planner::core::extract
