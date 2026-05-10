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

#include <cassert>
#include <vector>

#include <boost/unordered/unordered_flat_set.hpp>

#include "planner/extract/extractor.hpp"

namespace memgraph::planner::test_support {

/// Minimal CostResultType for tests that only need a scalar cost.
/// Production cost models use ParetoFrontier-based CostResultBase; use this
/// when a simple ordered cost is sufficient.
template <std::totally_ordered T>
struct DefaultCostResult {
  using cost_t = T;
  T cost;
  core::ENodeId enode_id;

  void merge_in_place(DefaultCostResult &&other) {
    if (other.cost < cost) *this = other;
  }

  [[nodiscard]] auto resolve() const -> std::pair<core::ENodeId, cost_t const &> { return {enode_id, cost}; }
};

static_assert(core::extract::CostResultType<DefaultCostResult<double>>);

/// Generic Resolver for tests and benchmarks: selects each eclass via
/// CostResult::resolve and walks every child of the chosen enode.  Safe
/// for any cost model whose children are unconditionally part of the
/// extracted tree.
///
/// Not safe for cost models where a chosen alt may exclude some of its
/// enode's children (e.g. plan_v2's PlanResolver, which honours alive/dead
/// semantics).  Production cost models with that shape provide their own
/// resolver.
struct DefaultResolver {
  template <typename Symbol, typename Analysis, core::extract::CostResultType CostResult>
  void operator()(core::EGraph<Symbol, Analysis> const &egraph,
                  core::extract::FrontierMap<CostResult> const &frontier_map, core::EClassId root,
                  core::extract::SelectionMap<typename CostResult::cost_t> &out) const {
    assert(out.empty() && "Resolver precondition: out must be empty on entry");
    auto to_visit = std::vector{root};
    auto visited = boost::unordered_flat_set{root};

    while (!to_visit.empty()) {
      auto current = to_visit.back();
      to_visit.pop_back();

      auto it = frontier_map.find(current);
      assert(it != frontier_map.end() && it->second.has_value());

      auto const &frontier = *it->second;
      auto [enode_id, cost] = frontier.resolve();
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

}  // namespace memgraph::planner::test_support
