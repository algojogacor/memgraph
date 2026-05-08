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

// Verifies the CardinalityEstimator plumbing: a mock injected through
// QueryPlannerContext is reachable via the public accessor and gets called
// when the cost model invokes it.  No estimator call sites exist yet in the
// cost model (this is the plumbing-only slice), so this test exercises the
// reachability contract directly by calling through the accessor.

#include <gtest/gtest.h>

#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/cardinality_estimator.hpp"
#include "query/plan_v2/egraph_converter.hpp"

namespace memgraph::query::plan::v2 {
namespace {

struct RecordingEstimator final : CardinalityEstimator {
  mutable int call_count = 0;
  double return_value;

  explicit RecordingEstimator(double v) : return_value(v) {}

  auto EstimateFunctionCardinality(uint64_t /*function_id*/, std::span<planner::core::EClassId const> /*arg_eclasses*/,
                                   EGraph const & /*eg*/) const -> double override {
    ++call_count;
    return return_value;
  }
};

TEST(EstimatorPlumbing, DefaultContextHasNoOverride) {
  QueryPlannerContext ctx;
  // No override -> ConvertToLogicalOperator builds a BuiltinEstimator
  // over the current egraph for the call.
  EXPECT_EQ(ctx.estimator_override(), nullptr);
}

TEST(EstimatorPlumbing, InjectedEstimatorIsReachable) {
  auto recorder = std::make_unique<RecordingEstimator>(42.0);
  auto *raw = recorder.get();
  QueryPlannerContext ctx{std::move(recorder)};

  // The accessor returns the same object we injected.
  EXPECT_EQ(ctx.estimator_override(), raw);
  EXPECT_EQ(raw->call_count, 0);

  // Calling through the accessor reaches the mock.  An empty arg span and a
  // dummy EGraph reference is fine: RecordingEstimator does not dereference
  // the egraph at this layer.
  EGraph dummy_eg;
  auto const result = ctx.estimator_override()->EstimateFunctionCardinality(0, {}, dummy_eg);
  EXPECT_DOUBLE_EQ(result, 42.0);
  EXPECT_EQ(raw->call_count, 1);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
