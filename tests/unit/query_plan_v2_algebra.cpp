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
using bind::SymbolSet;

auto Cmp(Alternative const &a, Alternative const &b) { return AlternativeDominance{}(a, b); }

auto MakeAlt(double cost, double cardinality, std::initializer_list<uint32_t> required, uint32_t enode_id,
             AliveTag is_alive = AliveTag::Dead) -> Alternative {
  return {.cost = cost,
          .cardinality = cardinality,
          .required = MakeSet(required),
          .enode_id = ENodeId{enode_id},
          .is_alive = is_alive};
}

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

std::string AltDominanceCaseName(const testing::TestParamInfo<AltDominanceCase> &info) {
  return std::string(info.param.name);
}

// ============================================================================
// Bind algebra: IsAlive
// ============================================================================

TEST(BindAlgebra_IsAlive, SymInRequiredYieldsAlive) {
  auto required = MakeSet({1, 2, 3});
  EXPECT_TRUE(required.is_alive(EClassId{2}));
}

TEST(BindAlgebra_IsAlive, SymNotInRequiredYieldsDead) {
  auto required = MakeSet({1, 2, 3});
  EXPECT_FALSE(required.is_alive(EClassId{4}));
}

TEST(BindAlgebra_IsAlive, EmptyRequiredIsAlwaysDead) {
  auto required = MakeSet({});
  EXPECT_FALSE(required.is_alive(EClassId{1}));
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
  SymbolSet input;
  EClassId sym;
  SymbolSet expr_demands;
  SymbolSet expected;

  AliveRequiredCase(std::string_view name_, std::initializer_list<uint32_t> input_ids, uint32_t sym_id,
                    std::initializer_list<uint32_t> expr_ids, std::initializer_list<uint32_t> expected_ids)
      : name(name_),
        input(MakeSet(input_ids)),
        sym(EClassId{sym_id}),
        expr_demands(MakeSet(expr_ids)),
        expected(MakeSet(expected_ids)) {}
};

class AliveRequiredTest : public testing::TestWithParam<AliveRequiredCase> {};

TEST_P(AliveRequiredTest, Verify) {
  auto const &tc = GetParam();
  EXPECT_EQ(tc.input.alive_required(tc.sym, tc.expr_demands), tc.expected);
}

std::string AliveRequiredCaseName(const testing::TestParamInfo<AliveRequiredCase> &info) {
  return std::string(info.param.name);
}

INSTANTIATE_TEST_SUITE_P(
    BindAlgebra_AliveRequired, AliveRequiredTest,
    testing::Values(AliveRequiredCase{"RemovesSymFromInput", {1, 2, 3}, 2, {}, {1, 3}},
                    AliveRequiredCase{"AddsExprDemandsToOutput", {1, 2}, 2, {4, 5}, {1, 4, 5}},
                    AliveRequiredCase{"ExprDemandOverlapsInputProducesUnion", {1, 2, 3}, 2, {3, 4}, {1, 3, 4}},
                    AliveRequiredCase{"ExprDemandReintroducesSym", {1, 2}, 2, {2}, {1, 2}},
                    AliveRequiredCase{"EmptyInputAndExprYieldsEmpty", {}, 2, {}, {}},
                    AliveRequiredCase{"SymIsMinElementOfInput_NoDemands", {1, 2, 3}, 1, {}, {2, 3}},
                    AliveRequiredCase{"SymIsMinElementOfInput_WithDemands", {1, 2, 3}, 1, {4}, {2, 3, 4}},
                    AliveRequiredCase{"SymIsMaxElementOfInput_NoDemands", {1, 2, 3}, 3, {}, {1, 2}},
                    AliveRequiredCase{"SymIsMaxElementOfInput_WithDemands", {1, 2, 3}, 3, {4}, {1, 2, 4}},
                    AliveRequiredCase{"SymIsSoleElementOfInput_NoDemands", {2}, 2, {}, {}},
                    AliveRequiredCase{"SymIsSoleElementOfInput_WithDemands", {2}, 2, {1, 3}, {1, 3}}),
    AliveRequiredCaseName);

INSTANTIATE_TEST_SUITE_P(AltDominance, AltDominanceTest,
                         testing::Values(AltDominanceCase{"LowerCostDominatesWhenCardinalityAndRequiredEqual",
                                                          MakeAlt(1.0, 6.0, {1}, 0),
                                                          MakeAlt(5.0, 6.0, {1}, 1),
                                                          std::partial_ordering::greater},
                                         AltDominanceCase{"EqualOnAllAxesIsEquivalent",
                                                          MakeAlt(5.0, 6.0, {1}, 0),
                                                          MakeAlt(5.0, 6.0, {1}, 1),
                                                          std::partial_ordering::equivalent},
                                         AltDominanceCase{"LowerCostHigherCardinalityIsUnordered",
                                                          MakeAlt(1.0, 1000.0, {1}, 0),
                                                          MakeAlt(5.0, 6.0, {1}, 1),
                                                          std::partial_ordering::unordered},
                                         AltDominanceCase{"SmallerRequiredDominatesWhenCostAndCardinalityEqual",
                                                          MakeAlt(5.0, 6.0, {1}, 0),
                                                          MakeAlt(5.0, 6.0, {1, 2}, 1),
                                                          std::partial_ordering::greater},
                                         AltDominanceCase{"LowerCardinalityDominates_Forward",
                                                          MakeAlt(5.0, 6.0, {1}, 0),
                                                          MakeAlt(5.0, 1000.0, {1}, 1),
                                                          std::partial_ordering::greater},
                                         AltDominanceCase{"LowerCardinalityDominates_Reverse",
                                                          MakeAlt(5.0, 1000.0, {1}, 0),
                                                          MakeAlt(5.0, 6.0, {1}, 1),
                                                          std::partial_ordering::less},
                                         AltDominanceCase{"IsAliveDoesNotParticipate",
                                                          MakeAlt(5.0, 6.0, {1}, 0, AliveTag::Alive),
                                                          MakeAlt(5.0, 6.0, {1}, 1, AliveTag::Dead),
                                                          std::partial_ordering::equivalent}),
                         AltDominanceCaseName);

}  // namespace
}  // namespace memgraph::query::plan::v2
