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

#include "query/plan_v2/builtin_estimator.hpp"

#include <optional>

#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/egraph_internal.hpp"

namespace memgraph::query::plan::v2 {

namespace {

/// Best-effort: if `eclass_id` carries a `Literal` e-node whose stored value
/// is an int, return it.
///
/// TODO: replace with EClass Analysis read once analysis_data is live.
/// (Phase 2 in ADR 0005 - the static_assert in eclass.cppm blocking
/// non-empty Analysis is the gating change.)  Today we walk the e-class
/// for a Literal e-node and reverse-look up the literal-store map.
auto TryReadIntLiteral(EGraph const &eg, egraph const &facade, planner::core::EClassId eclass_id)
    -> std::optional<int64_t> {
  auto const canonical = eg.find(eclass_id);
  auto const &cls = eg.eclass(canonical);
  auto const &lit_store = internal::get_impl(facade).storage<symbol::Literal>().store;

  for (auto enode_id : cls.nodes()) {
    auto const &enode = eg.get_enode(enode_id);
    if (enode.symbol() != symbol::Literal) continue;
    auto const id = enode.disambiguator();
    // Reverse lookup: store is value -> id.  Literal e-classes are tiny
    // (typically one e-node per distinct value) so the linear scan over
    // the value-keyed map is fine for Phase 1.
    for (auto const &[val, stored_id] : lit_store) {
      if (stored_id != id) continue;
      if (val.IsInt()) return val.ValueInt();
      return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

auto BuiltinEstimator::EstimateFunctionCardinality(uint64_t function_id,
                                                   std::span<planner::core::EClassId const> arg_eclasses,
                                                   EGraph const &eg) const -> double {
  auto const *info = facade.FunctionInfoById(function_id);
  if (info == nullptr) return kDefaultRowEstimate;

  switch (info->kind) {
    case BuiltinKind::Range: {
      // range(start, end[, step]).  Phase 1 only handles two-arg int-literal
      // start/end with implicit step=1; non-literal or non-int args fall
      // back to the default.
      if (arg_eclasses.size() < 2) return kDefaultRowEstimate;
      auto const a = TryReadIntLiteral(eg, facade, arg_eclasses[0]);
      auto const b = TryReadIntLiteral(eg, facade, arg_eclasses[1]);
      if (!a || !b) return kDefaultRowEstimate;
      // Cypher range(a, b) is inclusive on both ends -> b - a + 1, clamped at 0.
      auto const span = static_cast<double>(*b) - static_cast<double>(*a) + 1.0;
      return span < 0.0 ? 0.0 : span;
    }
    case BuiltinKind::Unknown:
      return kDefaultRowEstimate;
  }
  return kDefaultRowEstimate;
}

}  // namespace memgraph::query::plan::v2
