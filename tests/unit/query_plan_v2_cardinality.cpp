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

// Cardinality estimation tests for plan_v2.
// BuiltinEstimator: range(int, int) literal deduction, fallback to kDefaultRowEstimate.
// Unwind/Output/Subquery: end-to-end cardinality propagation.

#include <gtest/gtest.h>

#include "query/exceptions.hpp"
#include "query/plan/operator.hpp"
#include "query/plan_v2/builtin_estimator.hpp"
#include "query/plan_v2/builtin_functions.hpp"
#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/cardinality_estimator.hpp"
#include "query/plan_v2/egraph.hpp"
#include "query/plan_v2/egraph_converter.hpp"
#include "query/plan_v2/egraph_internal.hpp"
#include "storage/v2/property_value.hpp"

namespace memgraph::query::plan::v2 {
namespace {

using EClassId = planner::core::EClassId;

auto IntLiteral(egraph &eg, int64_t v) -> eclass { return eg.MakeLiteral(storage::ExternalPropertyValue{v}); }

auto CoreOf(egraph const &eg) -> EGraph const & { return internal::get_impl(eg).egraph_; }

auto AsCoreId(eclass e) -> EClassId { return EClassId{e.value_of()}; }

auto EstimateBuiltin(egraph &eg, eclass fn, std::vector<EClassId> args) -> double {
  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto const fn_eclass = core_eg.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto const &fn_enode = core_eg.get_enode(fn_enode_id);
  return estimator.Estimate(fn_enode, args, core_eg);
}

// ============================================================================
// BuiltinKind classification
// ============================================================================

TEST(BuiltinKindClassifier, RangeIsRecognised) {
  EXPECT_EQ(BuiltinKindFor("range"), BuiltinKind::Range);
  EXPECT_EQ(BuiltinKindFor("RANGE"), BuiltinKind::Range);
  EXPECT_EQ(BuiltinKindFor("Range"), BuiltinKind::Range);
}

TEST(BuiltinKindClassifier, UnknownFallback) {
  EXPECT_EQ(BuiltinKindFor("toString"), BuiltinKind::Unknown);
  EXPECT_EQ(BuiltinKindFor(""), BuiltinKind::Unknown);
  EXPECT_EQ(BuiltinKindFor("rang"), BuiltinKind::Unknown);
}

// ============================================================================
// BuiltinEstimator: Range cardinality
// ============================================================================

TEST(BuiltinEstimator, RangeWithIntLiteralsReturnsCount) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  auto b = IntLiteral(eg, 5);
  EXPECT_DOUBLE_EQ(EstimateBuiltin(eg, eg.MakeFunction("range", {a, b}), {AsCoreId(a), AsCoreId(b)}), 6.0);
}

TEST(BuiltinEstimator, RangeWithReversedBoundsClampsAtZero) {
  egraph eg;
  auto a = IntLiteral(eg, 5);
  auto b = IntLiteral(eg, 0);
  EXPECT_DOUBLE_EQ(EstimateBuiltin(eg, eg.MakeFunction("range", {a, b}), {AsCoreId(a), AsCoreId(b)}), 0.0);
}

TEST(BuiltinEstimator, RangeWithParameterFallsBackToDefault) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  auto b = eg.MakeParameterLookup(0);
  EXPECT_DOUBLE_EQ(EstimateBuiltin(eg, eg.MakeFunction("range", {a, b}), {AsCoreId(a), AsCoreId(b)}),
                   kDefaultRowEstimate);
}

TEST(BuiltinEstimator, UnknownFunctionFallsBackToDefault) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  EXPECT_DOUBLE_EQ(EstimateBuiltin(eg, eg.MakeFunction("unknown_func", {a}), {AsCoreId(a)}), kDefaultRowEstimate);
}

TEST(BuiltinEstimator, UnknownFunctionIdReturnsDefault) {
  egraph eg;
  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto fn = eg.MakeFunction("dummy", {});
  auto const fn_eclass = core_eg.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto fn_enode = core_eg.get_enode(fn_enode_id);
  fn_enode = planner::core::ENode{fn_enode.symbol(), fn_enode.children(), 0};
  EXPECT_DOUBLE_EQ(estimator.Estimate(fn_enode, {}, core_eg), kDefaultRowEstimate);
}

// ============================================================================
// Unwind cost composition end-to-end
// ============================================================================

struct FixedEstimator final : CardinalityEstimator {
  double value;

  explicit FixedEstimator(double v) : value(v) {}

  auto Estimate(planner::core::ENode<symbol> const &, std::span<planner::core::EClassId const>, EGraph const &) const
      -> double override {
    return value;
  }
};

TEST(UnwindCostShape, ProducesUnwindOperator) {
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

  // TODO: this is fake the estimator should be real so we can actully get size from the list
  auto ctx = QueryPlannerContext{std::make_unique<FixedEstimator>(6.0)};
  auto [plan, root_cost, ast, sym_table] = ConvertToLogicalOperator(eg, root, ctx);

  ASSERT_NE(plan, nullptr);

  auto const &produce = dynamic_cast<plan::Produce const &>(*plan);
  ASSERT_NE(produce.input(), nullptr);
  auto const *unwind_op = dynamic_cast<plan::Unwind const *>(produce.input().get());
  ASSERT_NE(unwind_op, nullptr) << "Top input must be a v1 Unwind";
  EXPECT_EQ(unwind_op->output_symbol_.name(), "x");
  ASSERT_NE(unwind_op->input(), nullptr);
  EXPECT_NE(dynamic_cast<plan::Once const *>(unwind_op->input().get()), nullptr);

  EXPECT_GT(root_cost, 0.0);
  EXPECT_LT(root_cost, 1e9);
  EXPECT_DOUBLE_EQ(ctx.last_root_cardinality(), 6.0);
}

// ============================================================================
// Subquery scope barrier
// ============================================================================

TEST(SubqueryBarrier, InnerBindingsStripped) {
  egraph eg;
  auto inner_once = eg.MakeOnce();
  auto x_sym = eg.MakeSymbol(0, "x");
  auto inner_one = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{1}});
  auto inner_bind = eg.MakeBind(inner_once, x_sym, inner_one);
  auto y_sym = eg.MakeSymbol(1, "y");
  auto inner_id_x = eg.MakeIdentifier(x_sym);
  auto inner_named = eg.MakeNamedOutput("y", y_sym, inner_id_x);
  auto inner_root = eg.MakeOutputs(inner_bind, {inner_named});

  auto outer_once = eg.MakeOnce();
  auto subq = eg.MakeSubquery(outer_once, inner_root, {y_sym});

  auto col_y_sym = eg.MakeSymbol(2, "y");
  auto outer_id_y = eg.MakeIdentifier(y_sym);
  auto outer_named_ok = eg.MakeNamedOutput("y", col_y_sym, outer_id_y);
  auto outer_root_ok = eg.MakeOutputs(subq, {outer_named_ok});

  QueryPlannerContext ctx;
  auto [plan_ok, _cost, _ast, _sym] = ConvertToLogicalOperator(eg, outer_root_ok, ctx);
  ASSERT_NE(plan_ok, nullptr);

  auto col_x_sym = eg.MakeSymbol(3, "x");
  auto outer_id_x = eg.MakeIdentifier(x_sym);
  auto outer_named_bad = eg.MakeNamedOutput("x", col_x_sym, outer_id_x);
  auto outer_root_bad = eg.MakeOutputs(subq, {outer_named_bad});

  QueryPlannerContext ctx_bad;
  EXPECT_THROW((void)ConvertToLogicalOperator(eg, outer_root_bad, ctx_bad), QueryException);
}

TEST(SubqueryBarrier, ImportingCallSurfacesNotYetImplemented) {
  egraph eg;
  auto outer_sym = eg.MakeSymbol(0, "x");
  auto outer_once = eg.MakeOnce();
  auto outer_bind = eg.MakeBind(outer_once, outer_sym, eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{1}}));

  auto inner_once = eg.MakeOnce();
  auto inner_id = eg.MakeIdentifier(outer_sym);
  auto y_sym = eg.MakeSymbol(1, "y");
  auto inner_named = eg.MakeNamedOutput("y", y_sym, inner_id);
  auto inner_root = eg.MakeOutputs(inner_once, {inner_named});

  auto subq = eg.MakeSubquery(outer_bind, inner_root, {y_sym});

  auto col_y_sym = eg.MakeSymbol(2, "y");
  auto outer_id_y = eg.MakeIdentifier(y_sym);
  auto outer_named = eg.MakeNamedOutput("y", col_y_sym, outer_id_y);
  auto outer_root = eg.MakeOutputs(subq, {outer_named});

  QueryPlannerContext ctx;
  EXPECT_THROW((void)ConvertToLogicalOperator(eg, outer_root, ctx), NotYetImplemented);
}

// ============================================================================
// Output cardinality contract
// ============================================================================

TEST(OutputCardinality, ScalarReturnIsOneRowEvenWhenValueIsList) {
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

// ============================================================================
// Kind dichotomy: construction-time validation and Output own_syms injection.
// See `src/query/plan_v2/CONTEXT.md` and ADR 0009.
// ============================================================================

// Output's `introduces` must include each NamedOutput's sym (own_syms rule).
// Pre-fix, OutputCombine set `introduces = pipe.introduces`, dropping the
// NamedOutput syms entirely.  Post-fix, the dispatch arm injects them.
TEST(OutputIntroduces, IncludesNamedOutputSyms) {
  egraph eg;
  auto once = eg.MakeOnce();
  auto a_sym = eg.MakeSymbol(0, "a");
  auto one = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{1}});
  auto bind_a = eg.MakeBind(once, a_sym, one);

  auto c_sym = eg.MakeSymbol(1, "c");
  auto id_a = eg.MakeIdentifier(a_sym);
  auto named_c = eg.MakeNamedOutput("c", c_sym, id_a);
  auto root = eg.MakeOutputs(bind_a, {named_c});

  QueryPlannerContext ctx;
  // If Output's introduces missed the NamedOutput sym `c`, parent-frame
  // resolution would still pass at the root (parent demands nothing).  But a
  // round-trip through ConvertToLogicalOperator exercises the resolver's
  // chosen.introduces fully: an Output Alt that doesn't introduce `c` would
  // fail when the resolver computes `chosen.introduces − own_syms` and finds
  // own_syms not contained in chosen.introduces (DMG_ASSERT in debug).
  // Plain check: a valid plan is produced.
  auto [plan, _cost, _ast, _sym] = ConvertToLogicalOperator(eg, root, ctx);
  ASSERT_NE(plan, nullptr);
}

// The chained-Bind regression at the cost-model level: outer Bind's expr
// references inner Bind's sym.  Pre-fix, BindFlatMap propagated expr.required
// upward without absorbing against input.introduces, leaving `a` falsely in
// the outer Bind's required.
TEST(BindAbsorption, ExprRequiredAbsorbedByInputIntroduces) {
  egraph eg;
  auto once = eg.MakeOnce();
  auto a_sym = eg.MakeSymbol(0, "a");
  auto one = eg.MakeLiteral(storage::ExternalPropertyValue{int64_t{1}});
  auto bind_a = eg.MakeBind(once, a_sym, one);

  auto b_sym = eg.MakeSymbol(1, "b");
  auto id_a = eg.MakeIdentifier(a_sym);
  auto bind_b = eg.MakeBind(bind_a, b_sym, id_a);  // b = a, expr references a

  auto c_sym = eg.MakeSymbol(2, "c");
  auto id_b = eg.MakeIdentifier(b_sym);
  auto named_c = eg.MakeNamedOutput("c", c_sym, id_b);
  auto root = eg.MakeOutputs(bind_b, {named_c});

  QueryPlannerContext ctx;
  auto [plan, _cost, _ast, _sym] = ConvertToLogicalOperator(eg, root, ctx);
  ASSERT_NE(plan, nullptr);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
