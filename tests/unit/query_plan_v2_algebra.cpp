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

TEST(BindAlgebra_AliveRequired, RemovesSymFromInput) {
  auto result = bind::AliveRequired(MakeSet({1, 2, 3}), EClassId{2}, MakeSet({}));
  EXPECT_EQ(result, MakeSet({1, 3}));
}

TEST(BindAlgebra_AliveRequired, AddsExprDemandsToOutput) {
  auto result = bind::AliveRequired(MakeSet({1, 2}), EClassId{2}, MakeSet({4, 5}));
  EXPECT_EQ(result, MakeSet({1, 4, 5}));
}

TEST(BindAlgebra_AliveRequired, ExprDemandOverlapsInputProducesUnion) {
  auto result = bind::AliveRequired(MakeSet({1, 2, 3}), EClassId{2}, MakeSet({3, 4}));
  EXPECT_EQ(result, MakeSet({1, 3, 4}));
}

TEST(BindAlgebra_AliveRequired, ExprDemandReintroducesSym) {
  auto result = bind::AliveRequired(MakeSet({1, 2}), EClassId{2}, MakeSet({2}));
  EXPECT_EQ(result, MakeSet({1, 2}));
}

TEST(BindAlgebra_AliveRequired, EmptyInputAndExprYieldsEmpty) {
  auto result = bind::AliveRequired(MakeSet({}), EClassId{2}, MakeSet({}));
  EXPECT_EQ(result, MakeSet({}));
}

TEST(BindAlgebra_AliveRequired, SymIsMinElementOfInput) {
  EXPECT_EQ(bind::AliveRequired(MakeSet({1, 2, 3}), EClassId{1}, MakeSet({})), MakeSet({2, 3}));
  EXPECT_EQ(bind::AliveRequired(MakeSet({1, 2, 3}), EClassId{1}, MakeSet({4})), MakeSet({2, 3, 4}));
}

TEST(BindAlgebra_AliveRequired, SymIsMaxElementOfInput) {
  EXPECT_EQ(bind::AliveRequired(MakeSet({1, 2, 3}), EClassId{3}, MakeSet({})), MakeSet({1, 2}));
  EXPECT_EQ(bind::AliveRequired(MakeSet({1, 2, 3}), EClassId{3}, MakeSet({4})), MakeSet({1, 2, 4}));
}

TEST(BindAlgebra_AliveRequired, SymIsSoleElementOfInput) {
  EXPECT_EQ(bind::AliveRequired(MakeSet({2}), EClassId{2}, MakeSet({})), MakeSet({}));
  EXPECT_EQ(bind::AliveRequired(MakeSet({2}), EClassId{2}, MakeSet({1, 3})), MakeSet({1, 3}));
}

// ============================================================================
// Alternative dominance: cost axis
// ============================================================================

TEST(AltDominance, LowerCostDominatesWhenCardinalityAndRequiredEqual) {
  Alternative a{.cost = 1.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::greater);
}

TEST(AltDominance, EqualOnAllAxesIsEquivalent) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::equivalent);
}

TEST(AltDominance, LowerCostHigherCardinalityIsUnordered) {
  Alternative a{.cost = 1.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::unordered);
}

// ============================================================================
// Alternative dominance: required-set axis
// ============================================================================

TEST(AltDominance, SmallerRequiredDominatesWhenCostAndCardinalityEqual) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1, 2}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::greater);
}

// ============================================================================
// Alternative dominance: cardinality axis
// ============================================================================

TEST(AltDominance, LowerCardinalityDominatesWhenCostAndRequiredEqual) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::greater);
  EXPECT_EQ(Cmp(b, a), std::partial_ordering::less);
}

// ============================================================================
// Alternative dominance: is_alive does not participate
// ============================================================================

TEST(AltDominance, IsAliveDoesNotParticipate) {
  Alternative a{
      .cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}, .is_alive = AliveTag::Alive};
  Alternative b{
      .cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}, .is_alive = AliveTag::Dead};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::equivalent);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
