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

#include "query/plan_v2/builtin_functions.hpp"
#include "query/plan_v2/cardinality_estimator.hpp"

namespace memgraph::query::plan::v2 {

struct egraph;  // public e-graph facade

/// Estimator for built-in Cypher functions.
///
/// Resolves the function id to a cached BuiltinKind via the e-graph's
/// function-info table (set up by symbol_make_traits<symbol::Function>),
/// then dispatches:
///   - Range:   walks both arg e-classes for `Literal` int e-nodes.  When
///              both ints are concrete, returns `b - a + 1` (clamped at 0).
///              Otherwise returns kDefaultRowEstimate.
///   - Unknown: returns kDefaultRowEstimate (UDF / unrecognised builtin).
///
/// The estimator looks up FunctionInfo through the public e-graph facade
/// (which the cost model passes alongside the EGraph for the e-class walk).
/// The constructor takes that facade by reference so the estimator can find
/// the right interner without round-tripping through the cost model.
struct BuiltinEstimator final : CardinalityEstimator {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  egraph const &facade;

  explicit BuiltinEstimator(egraph const &f) : facade(f) {}

  auto EstimateFunctionCardinality(uint64_t function_id, std::span<planner::core::EClassId const> arg_eclasses,
                                   EGraph const &eg) const -> double override;
};

}  // namespace memgraph::query::plan::v2
