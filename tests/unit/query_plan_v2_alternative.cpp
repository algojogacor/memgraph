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

// Pure-algebra tests for the Alternative dominance relation.  Covers the
// cardinality axis added alongside cost and required-set so the three-way
// Pareto comparison behaves componentwise.

#include <gtest/gtest.h>

#include "query/plan_v2/plan_alternative.hpp"

import memgraph.planner.core.egraph;

namespace memgraph::query::plan::v2 {
namespace {

using EClassId = planner::core::EClassId;
using ENodeId = planner::core::ENodeId;

auto MakeSet(std::initializer_list<uint32_t> ids) -> bind::SymbolSet {
  bind::SymbolSet s;
  for (auto id : ids) s.insert(EClassId{id});
  return s;
}

auto Cmp(Alternative const &a, Alternative const &b) { return AlternativeDominance{}(a, b); }

TEST(AltDominance, LowerCardinalityDominatesWhenCostAndRequiredEqual) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  // a is better → a dominates b → a > b.
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::greater);
  EXPECT_EQ(Cmp(b, a), std::partial_ordering::less);
}

TEST(AltDominance, EqualOnAllAxesIsEquivalent) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::equivalent);
}

TEST(AltDominance, LowerCostDominatesWhenCardinalityAndRequiredEqual) {
  Alternative a{.cost = 1.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::greater);
}

TEST(AltDominance, LowerCostHigherCardinalityIsUnordered) {
  Alternative a{.cost = 1.0, .cardinality = 1000.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::unordered);
}

TEST(AltDominance, IsAliveDoesNotParticipate) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}, .is_alive = true};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{1}, .is_alive = false};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::equivalent);
}

TEST(AltDominance, SmallerRequiredDominatesWhenCostAndCardinalityEqual) {
  Alternative a{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1}), .enode_id = ENodeId{0}};
  Alternative b{.cost = 5.0, .cardinality = 6.0, .required = MakeSet({1, 2}), .enode_id = ENodeId{1}};
  EXPECT_EQ(Cmp(a, b), std::partial_ordering::greater);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
