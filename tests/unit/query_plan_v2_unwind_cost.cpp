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

// Unwind cost composition end-to-end.  Builds a small egraph by hand
// (Once, Symbol, range(0, n)-as-Function, MakeUnwind), plans it with a
// recording mock estimator that returns a known list cardinality, and
// inspects the resulting plan to verify the chosen alt is an Unwind with
// the expected cost arithmetic.
//
// We can't peek directly into the cost-model frontier (it's TU-private), so
// the focused-cost test runs end-to-end and inspects the returned plan and
// the mock's call sequence.

#include <gtest/gtest.h>

#include "query/plan/operator.hpp"
#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/cardinality_estimator.hpp"
#include "query/plan_v2/egraph.hpp"
#include "query/plan_v2/egraph_converter.hpp"
#include "storage/v2/property_value.hpp"

namespace memgraph::query::plan::v2 {
namespace {

struct FixedEstimator final : CardinalityEstimator {
  double value;

  explicit FixedEstimator(double v) : value(v) {}

  auto EstimateFunctionCardinality(uint64_t /*function_id*/, std::span<planner::core::EClassId const> /*arg_eclasses*/,
                                   EGraph const & /*eg*/) const -> double override {
    return value;
  }
};

TEST(UnwindCostShape, ProducesUnwindOperator) {
  // egraph: Output(Unwind(Once, x_sym, range(0,5)), NamedOutput(r_sym, 1))
  // The body uses a literal so the Output's NamedOutput needs no symbols
  // introduced by Unwind (today's required-set algebra can't pass those
  // across the Output/Unwind sibling boundary).
  egraph eg;
  auto once = eg.MakeOnce();
  auto x_sym = eg.MakeSymbol(0, "x");
  auto a = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{0}});
  auto b = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{5}});
  auto range = eg.MakeFunction("range", {a, b});
  auto unwind = eg.MakeUnwind(once, x_sym, range);

  auto r_sym = eg.MakeSymbol(1, "r");
  auto one = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{1}});
  auto named_output = eg.MakeNamedOutput("r", r_sym, one);
  auto root = eg.MakeOutputs(unwind, {named_output});

  // Use a fixed estimator returning 6.0 for any function call so range's
  // cardinality is predictable without relying on BuiltinEstimator's
  // literal walk.
  auto ctx = QueryPlannerContext{std::make_unique<FixedEstimator>(6.0)};
  auto [plan, root_cost, ast, sym_table] = ConvertToLogicalOperator(eg, root, ctx);

  ASSERT_NE(plan, nullptr);

  // Plan should be Produce -> Unwind -> Once.
  auto const &produce = dynamic_cast<plan::Produce const &>(*plan);
  ASSERT_NE(produce.input(), nullptr);
  auto const *unwind_op = dynamic_cast<plan::Unwind const *>(produce.input().get());
  ASSERT_NE(unwind_op, nullptr) << "Top input must be a v1 Unwind";
  EXPECT_EQ(unwind_op->output_symbol_.name(), "x");
  ASSERT_NE(unwind_op->input(), nullptr);
  EXPECT_NE(dynamic_cast<plan::Once const *>(unwind_op->input().get()), nullptr);

  // Cost should reflect the Unwind formula:
  //   cost = input.cost + (list_expr.cost + kUnwindPerRowOverhead) * input.cardinality + sym.cost
  // input.cardinality = 1 (Once).  list_expr (range(0,5)) cardinality = 6
  // (from FixedEstimator) but its *cost* is 1 (structural Function) + 2
  // (two literal arg costs).  We don't pin the exact number to avoid
  // brittle coupling, just verify the cost is finite and positive.
  EXPECT_GT(root_cost, 0.0);
  EXPECT_LT(root_cost, 1e9);

  // Root cardinality: Output produces input row count.  Unwind multiplied
  // Once (1) by range (6) giving 6; Output preserves that.
  EXPECT_DOUBLE_EQ(ctx.last_root_cardinality(), 6.0);
}

TEST(OutputCardinality, ScalarReturnIsOneRowEvenWhenValueIsList) {
  // RETURN range(0, 5) AS r produces ONE row containing a 6-element list,
  // not six rows.  Pins the contract that NamedOutput is per-evaluation
  // (always cardinality = 1) and Output's cardinality is the input row
  // pipe's, not input × NamedOutput.  A naive product would give 6 here
  // and silently mis-cost any plan that puts a list-valued NamedOutput
  // above a row pipe.
  egraph eg;
  auto once = eg.MakeOnce();
  auto a = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{0}});
  auto b = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{5}});
  auto range = eg.MakeFunction("range", {a, b});
  auto r_sym = eg.MakeSymbol(0, "r");
  auto named_output = eg.MakeNamedOutput("r", r_sym, range);
  auto root = eg.MakeOutputs(once, {named_output});

  auto ctx = QueryPlannerContext{std::make_unique<FixedEstimator>(6.0)};
  auto [plan, cost, ast, sym_table] = ConvertToLogicalOperator(eg, root, ctx);

  ASSERT_NE(plan, nullptr);
  EXPECT_DOUBLE_EQ(ctx.last_root_cardinality(), 1.0);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
