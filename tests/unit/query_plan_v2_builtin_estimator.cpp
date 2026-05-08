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

auto FunctionId(egraph const &eg, std::string_view name) -> uint64_t {
  auto const &store = internal::get_impl(eg).storage<symbol::Function>().store;
  auto it = store.find(std::string{name});
  return it->second;
}

// Helper: get the underlying core EGraph reference and the arg eclass span.
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
  (void)eg.MakeFunction("range", {a, b});

  BuiltinEstimator estimator{eg};
  auto const fid = FunctionId(eg, "range");
  std::array<EClassId, 2> args{AsCoreId(a), AsCoreId(b)};
  EXPECT_DOUBLE_EQ(estimator.EstimateFunctionCardinality(fid, args, CoreOf(eg)), 6.0);
}

TEST(BuiltinEstimator, RangeWithReversedBoundsClampsAtZero) {
  egraph eg;
  auto a = IntLiteral(eg, 5);
  auto b = IntLiteral(eg, 0);
  (void)eg.MakeFunction("range", {a, b});

  BuiltinEstimator estimator{eg};
  auto const fid = FunctionId(eg, "range");
  std::array<EClassId, 2> args{AsCoreId(a), AsCoreId(b)};
  EXPECT_DOUBLE_EQ(estimator.EstimateFunctionCardinality(fid, args, CoreOf(eg)), 0.0);
}

TEST(BuiltinEstimator, RangeWithParameterFallsBackToDefault) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  auto b = eg.MakeParameterLookup(0);  // not a literal
  (void)eg.MakeFunction("range", {a, b});

  BuiltinEstimator estimator{eg};
  auto const fid = FunctionId(eg, "range");
  std::array<EClassId, 2> args{AsCoreId(a), AsCoreId(b)};
  EXPECT_DOUBLE_EQ(estimator.EstimateFunctionCardinality(fid, args, CoreOf(eg)), kDefaultRowEstimate);
}

TEST(BuiltinEstimator, UnknownFunctionFallsBackToDefault) {
  egraph eg;
  auto a = IntLiteral(eg, 0);
  (void)eg.MakeFunction("unknown_func", {a});

  BuiltinEstimator estimator{eg};
  auto const fid = FunctionId(eg, "unknown_func");
  std::array<EClassId, 1> args{AsCoreId(a)};
  EXPECT_DOUBLE_EQ(estimator.EstimateFunctionCardinality(fid, args, CoreOf(eg)), kDefaultRowEstimate);
}

TEST(BuiltinEstimator, UnknownFunctionIdReturnsDefault) {
  egraph eg;
  BuiltinEstimator estimator{eg};
  // No function ever interned -> any id is unknown.
  EXPECT_DOUBLE_EQ(estimator.EstimateFunctionCardinality(0, {}, CoreOf(eg)), kDefaultRowEstimate);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
