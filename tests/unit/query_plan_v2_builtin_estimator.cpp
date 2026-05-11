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

// BuiltinEstimator unit tests: range(int, int) literal deduction, fall-back
// to kDefaultRowEstimate for non-literal arguments and unrecognised builtins.

#include <gtest/gtest.h>

#include "query/plan_v2/builtin_estimator.hpp"
#include "query/plan_v2/builtin_functions.hpp"
#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/egraph.hpp"
#include "query/plan_v2/egraph_internal.hpp"
#include "storage/v2/property_value.hpp"

namespace memgraph::query::plan::v2 {
namespace {

using EClassId = planner::core::EClassId;

auto IntLiteral(egraph &eg, int64_t v) -> eclass { return eg.MakeLiteral(storage::ExternalPropertyValue{v}); }

auto CoreOf(egraph const &eg) -> EGraph const & { return internal::get_impl(eg).egraph_; }

auto AsCoreId(eclass e) -> EClassId { return EClassId{e.value_of()}; }

TEST(BuiltinKindClassifier, RangeIsRecognised) {
  EXPECT_EQ(BuiltinKindFor("range"), BuiltinKind::Range);
  EXPECT_EQ(BuiltinKindFor("RANGE"), BuiltinKind::Range);
  EXPECT_EQ(BuiltinKindFor("Range"), BuiltinKind::Range);
}

TEST(BuiltinKindClassifier, UnknownFallback) {
  EXPECT_EQ(BuiltinKindFor("toString"), BuiltinKind::Unknown);
  EXPECT_EQ(BuiltinKindFor(""), BuiltinKind::Unknown);
  EXPECT_EQ(BuiltinKindFor("rang"), BuiltinKind::Unknown);  // close but not equal
}

TEST(BuiltinEstimator, RangeWithIntLiteralsReturnsCount) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  auto b = IntLiteral(eg, 5);
  auto fn = eg.MakeFunction("range", {a, b});

  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto const fn_eclass = internal::get_impl(eg).egraph_.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto const &fn_enode = core_eg.get_enode(fn_enode_id);

  std::array<EClassId, 2> args{AsCoreId(a), AsCoreId(b)};
  EXPECT_DOUBLE_EQ(estimator.Estimate(fn_enode, args, core_eg), 6.0);
}

TEST(BuiltinEstimator, RangeWithReversedBoundsClampsAtZero) {
  egraph eg;
  auto a = IntLiteral(eg, 5);
  auto b = IntLiteral(eg, 0);
  auto fn = eg.MakeFunction("range", {a, b});

  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto const fn_eclass = internal::get_impl(eg).egraph_.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto const &fn_enode = core_eg.get_enode(fn_enode_id);

  std::array<EClassId, 2> args{AsCoreId(a), AsCoreId(b)};
  EXPECT_DOUBLE_EQ(estimator.Estimate(fn_enode, args, core_eg), 0.0);  // 0 - 5 + 1 = -4, clamped to 0
}

TEST(BuiltinEstimator, RangeWithParameterFallsBackToDefault) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  auto b = eg.MakeParameterLookup(0);  // not a literal
  auto fn = eg.MakeFunction("range", {a, b});

  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto const fn_eclass = internal::get_impl(eg).egraph_.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto const &fn_enode = core_eg.get_enode(fn_enode_id);

  std::array<EClassId, 2> args{AsCoreId(a), AsCoreId(b)};
  EXPECT_DOUBLE_EQ(estimator.Estimate(fn_enode, args, core_eg), kDefaultRowEstimate);
}

TEST(BuiltinEstimator, UnknownFunctionFallsBackToDefault) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  auto fn = eg.MakeFunction("unknown_func", {a});

  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto const fn_eclass = internal::get_impl(eg).egraph_.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto const &fn_enode = core_eg.get_enode(fn_enode_id);

  std::array<EClassId, 1> args{AsCoreId(a)};
  EXPECT_DOUBLE_EQ(estimator.Estimate(fn_enode, args, core_eg), kDefaultRowEstimate);
}

TEST(BuiltinEstimator, UnknownFunctionIdReturnsDefault) {
  egraph eg;
  BuiltinEstimator estimator{eg};
  auto const &core_eg = CoreOf(eg);
  auto fn = eg.MakeFunction("dummy", {});
  auto const fn_eclass = internal::get_impl(eg).egraph_.find(EClassId{fn.value_of()});
  auto const fn_enode_id = core_eg.eclass(fn_eclass).nodes()[0];
  auto fn_enode = core_eg.get_enode(fn_enode_id);
  // Override disambiguator to 0 to simulate an unknown function id.
  fn_enode = planner::core::ENode{fn_enode.symbol(), fn_enode.children(), 0};
  EXPECT_DOUBLE_EQ(estimator.Estimate(fn_enode, {}, core_eg), kDefaultRowEstimate);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
