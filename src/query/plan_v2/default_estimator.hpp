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

#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/cardinality_estimator.hpp"

namespace memgraph::query::plan::v2 {

/// Returns kDefaultRowEstimate for every call.  Used when no smarter
/// estimator is wired (default-constructed QueryPlannerContext) and as
/// a safe fallback for unrecognised functions.
struct DefaultEstimator final : CardinalityEstimator {
  auto EstimateFunctionCardinality(uint64_t /*function_id*/, std::span<planner::core::EClassId const> /*arg_eclasses*/,
                                   EGraph const & /*eg*/) const -> double override {
    return kDefaultRowEstimate;
  }
};

}  // namespace memgraph::query::plan::v2
