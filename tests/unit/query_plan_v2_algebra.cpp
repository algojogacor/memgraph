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

// Pure-algebra tests for plan_v2: Bind semantics and Alternative dominance.
// These do not construct an egraph or invoke the planner - they exercise
// the alive/dead predicate, cost formulas, required-set algebra, and
// Pareto dominance directly.
//
// Behaviour changes to any of these shift the cost-model and resolver
// simultaneously and benefit from being caught at the algebra level
// rather than via the full pipeline.

#include <gtest/gtest.h>

#include "query/plan_v2/bind_semantics.hpp"
#include "query/plan_v2/plan_alternative.hpp"
#include "query/plan_v2/test_support/sym_sets.hpp"

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2 {
namespace {

using EClassId = planner::core::EClassId;
using ENodeId = planner::core::ENodeId;
using bind::MakeSet;

auto Cmp(Alternative const &a, Alternative const &b) { return AlternativeDominance{}(a, b); }

struct AltDominanceCase {
  std::string_view name;
  Alternative a;
  Alternative b;
  std::partial_ordering expected;
};

class AltDominanceTest : public testing::TestWithParam<AltDominanceCase> {};

TEST_P(AltDominanceTest, Verify) {
  auto const &tc = GetParam();
  EXPECT_EQ(Cmp(tc.a, tc.b), tc.expected);
}

// ============================================================================
// Bind algebra: IsAlive
// ============================================================================

TEST(BindAlgebra_IsAlive, SymInRequiredYieldsAlive) {
  auto required = MakeSet({1, 2, 3});
  EXPECT_TRUE(bind::IsAlive(required, EClassId{2}));
}

TEST(BindAlgebra_IsAlive, SymNotInRequiredYieldsDead) {
  auto required = MakeSet({1, 2, 3});
  EXPECT_FALSE(bind::IsAlive(required, EClassId{4}));
}

TEST(BindAlgebra_IsAlive, EmptyRequiredIsAlwaysDead) {
  auto required = MakeSet({});
  EXPECT_FALSE(bind::IsAlive(required, EClassId{1}));
}

// ============================================================================
// Bind algebra: cost formulas
// ============================================================================

TEST(BindAlgebra_Cost, AliveSumsAllThree) {
  EXPECT_DOUBLE_EQ(bind::AliveCost(2.5, bind::kSymbolCost, 3.5), 2.5 + bind::kSymbolCost + 3.5);
}

TEST(BindAlgebra_Cost, DeadIgnoresSymAndExpr) { EXPECT_DOUBLE_EQ(bind::DeadCost(7.5), 7.5); }

TEST(BindAlgebra_Cost, kSymbolCostIsOne) { EXPECT_DOUBLE_EQ(bind::kSymbolCost, 1.0); }

// ============================================================================
// Bind algebra: AliveRequired set algebra
// ============================================================================

struct AliveRequiredCase {
  std::string_view name;
  bind::AliveRequired::set_type input;
  EClassId sym;
  bind::AliveRequired::set_type expr_demands;
  bind::AliveRequired::set_type expected;
};

class AliveRequiredTest : public testing::TestWithParam<AliveRequiredCase> {};

TEST_P(AliveRequiredTest, Verify) {
  auto const &tc = GetParam();
  EXPECT_EQ(bind::AliveRequired(tc.input, tc.sym, tc.expr_demands), tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    BindAlgebra_AliveRequired, AliveRequiredTest,
    testing::Values(
        AliveRequiredCase{"RemovesSymFromInput", MakeSet({1, 2, 3}), EClassId{2}, MakeSet({}), MakeSet({1, 3})},
        AliveRequiredCase{"AddsExprDemandsToOutput", MakeSet({1, 2}), EClassId{2}, MakeSet({4, 5}), MakeSet({1, 4, 5})},
        AliveRequiredCase{"ExprDemandOverlapsInputProducesUnion",
                          MakeSet({1, 2, 3}),
                          EClassId{2},
                          MakeSet({3, 4}),
                          MakeSet({1, 3, 4})},
        AliveRequiredCase{"ExprDemandReintroducesSym", MakeSet({1, 2}), EClassId{2}, MakeSet({2}), MakeSet({1, 2})},
        AliveRequiredCase{"EmptyInputAndExprYieldsEmpty", MakeSet({}), EClassId{2}, MakeSet({}), MakeSet({})},
        AliveRequiredCase{
            "SymIsMinElementOfInput_NoDemands", MakeSet({1, 2, 3}), EClassId{1}, MakeSet({}), MakeSet({2, 3})},
        AliveRequiredCase{
            "SymIsMinElementOfInput_WithDemands", MakeSet({1, 2, 3}), EClassId{1}, MakeSet({4}), MakeSet({2, 3, 4})},
        AliveRequiredCase{
            "SymIsMaxElementOfInput_NoDemands", MakeSet({1, 2, 3}), EClassId{3}, MakeSet({}), MakeSet({1, 2})},
        AliveRequiredCase{
            "SymIsMaxElementOfInput_WithDemands", MakeSet({1, 2, 3}), EClassId{3}, MakeSet({4}), MakeSet({1, 2, 4})},
        AliveRequiredCase{"SymIsSoleElementOfInput_NoDemands", MakeSet({2}), EClassId{2}, MakeSet({}), MakeSet({})},
        AliveRequiredCase{
            "SymIsSoleElementOfInput_WithDemands", MakeSet({2}), EClassId{2}, MakeSet({1, 3}), MakeSet({1, 3})}),
    [](auto const &info) { return std::string(info.param.name); });

INSTANTIATE_TEST_SUITE_P(
    AltDominance, AltDominanceTest,
    testing::Values(
        AltDominanceCase{"LowerCostDominatesWhenCardinalityAndRequiredEqual",
                         {.cost = 1.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}},
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}},
                         std::partial_ordering::greater},
        AltDominanceCase{"EqualOnAllAxesIsEquivalent",
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}},
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}},
                         std::partial_ordering::equivalent},
        AltDominanceCase{"LowerCostHigherCardinalityIsUnordered",
                         {.cost = 1.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{0}},
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}},
                         std::partial_ordering::unordered},
        AltDominanceCase{"SmallerRequiredDominatesWhenCostAndCardinalityEqual",
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}},
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1, 2}), .enode_id = ENodeId{1}},
                         std::partial_ordering::greater},
        AltDominanceCase{"LowerCardinalityDominates_Forward",
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}},
                         {.cost = 5.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{1}},
                         std::partial_ordering::greater},
        AltDominanceCase{"LowerCardinalityDominates_Reverse",
                         {.cost = 5.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{0}},
                         {.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}},
                         std::partial_ordering::less},
        AltDominanceCase{"IsAliveDoesNotParticipate",
                         {.cost = 5.0,
                          .cardinality = 6.0,
                          .required = MakeSet({1}),
                          .enode_id = ENodeId{0},
                          .is_alive = AliveTag::Alive},
                         {.cost = 5.0,
                          .cardinality = 6.0,
                          .required = MakeSet({1}),
                          .enode_id = ENodeId{1},
                          .is_alive = AliveTag::Dead},
                         std::partial_ordering::equivalent}),
    [](auto const &info) { return std::string(info.param.name); });

}  // namespace
}  // namespace memgraph::query::plan::v2
