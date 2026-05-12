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

#include <gtest/gtest.h>

#include "query/plan_v2/egraph.hpp"
#include "query/plan_v2/egraph_internal.hpp"
#include "query/plan_v2/rewrites.hpp"

namespace memgraph::query::plan::v2 {
namespace {

class EgraphTestBase : public ::testing::Test {
 protected:
  egraph eg_;

  auto &Core() { return internal::get_impl(eg_).egraph_; }

  [[nodiscard]] auto Find(eclass e) { return Core().find(internal::to_core_id(e)); }

  void ExpectDistinct(eclass a, eclass b) { EXPECT_NE(Find(a), Find(b)); }

  void ExpectSame(eclass a, eclass b) { EXPECT_EQ(Find(a), Find(b)); }

  void ExpectAllDistinct(std::vector<eclass> const &ops) {
    for (size_t i = 0; i < ops.size(); ++i)
      for (size_t j = i + 1; j < ops.size(); ++j) ExpectDistinct(ops[i], ops[j]);
  }
};

class InlineRewriteTest : public EgraphTestBase {};

// Test: Simple inline rewrite
// Bind(Once, sym, Literal) + Identifier(sym) -> Identifier merged with Literal
TEST_F(InlineRewriteTest, SimpleInline) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{42});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, literal);
  auto ident = eg_.MakeIdentifier(sym);

  ExpectDistinct(ident, literal);
  EXPECT_EQ(ApplyInlineRewrite(eg_), 1);
  ExpectSame(ident, literal);
}

// Test: Multiple identifiers referencing same binding
TEST_F(InlineRewriteTest, MultipleIdentifiersSameBinding) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{42});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, literal);

  auto ident1 = eg_.MakeIdentifier(sym);
  auto ident2 = eg_.MakeIdentifier(sym);

  ExpectDistinct(ident1, literal);
  ExpectDistinct(ident2, literal);

  auto merges = ApplyInlineRewrite(eg_);
  EXPECT_GE(merges, 1);

  ExpectSame(ident1, literal);
  ExpectSame(ident2, literal);
}

// Test: No rewrite when no binding exists
TEST_F(InlineRewriteTest, NoBindingNoRewrite) {
  // Create just an Identifier without corresponding Bind
  auto sym = eg_.MakeSymbol(0, "x");
  [[maybe_unused]] auto ident = eg_.MakeIdentifier(sym);

  // Apply rewrite - should do nothing
  auto merges = ApplyInlineRewrite(eg_);
  EXPECT_EQ(merges, 0);
}

// Test: Different symbols don't interfere
TEST_F(InlineRewriteTest, DifferentSymbolsIndependent) {
  auto once1 = eg_.MakeOnce();
  auto sym0 = eg_.MakeSymbol(0, "x");
  auto literal1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  [[maybe_unused]] auto bind1 = eg_.MakeBind(once1, sym0, literal1);

  auto once2 = eg_.MakeOnce();
  auto sym1 = eg_.MakeSymbol(1, "y");
  auto literal2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});
  [[maybe_unused]] auto bind2 = eg_.MakeBind(once2, sym1, literal2);

  auto ident0 = eg_.MakeIdentifier(sym0);
  auto ident1 = eg_.MakeIdentifier(sym1);

  EXPECT_EQ(ApplyInlineRewrite(eg_), 2);

  ExpectSame(ident0, literal1);
  ExpectSame(ident1, literal2);
  ExpectDistinct(literal1, literal2);
}

// Test: Chained bindings
// x = 1, y = x => Identifier(y) should be equivalent to Literal(1)
TEST_F(InlineRewriteTest, ChainedBindings) {
  auto once = eg_.MakeOnce();
  auto sym_x = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto bind_x = eg_.MakeBind(once, sym_x, literal);

  auto ident_x = eg_.MakeIdentifier(sym_x);

  auto sym_y = eg_.MakeSymbol(1, "y");
  [[maybe_unused]] auto bind_y = eg_.MakeBind(bind_x, sym_y, ident_x);

  auto ident_y = eg_.MakeIdentifier(sym_y);

  auto result = ApplyAllRewrites(eg_, RewriteConfig{.max_iterations = 5});
  EXPECT_GE(result.rewrites_applied, 2);

  ExpectSame(ident_y, literal);
}

// Test: Iteration limit stops rewriting
TEST_F(InlineRewriteTest, IterationLimit) {
  // Create a chain of bindings that requires multiple iterations
  // x = 1, y = x, z = y => need 2 iterations to propagate to z
  auto once = eg_.MakeOnce();
  auto sym_x = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto bind_x = eg_.MakeBind(once, sym_x, literal);

  auto ident_x = eg_.MakeIdentifier(sym_x);
  auto sym_y = eg_.MakeSymbol(1, "y");
  auto bind_y = eg_.MakeBind(bind_x, sym_y, ident_x);

  auto ident_y = eg_.MakeIdentifier(sym_y);
  auto sym_z = eg_.MakeSymbol(2, "z");
  [[maybe_unused]] auto bind_z = eg_.MakeBind(bind_y, sym_z, ident_y);

  [[maybe_unused]] auto ident_z = eg_.MakeIdentifier(sym_z);

  // With max_iterations=1, should stop early
  auto result = ApplyAllRewrites(eg_, RewriteConfig{.max_iterations = 1});
  EXPECT_EQ(result.stop_reason, RewriteResult::StopReason::IterationLimit);
  EXPECT_EQ(result.iterations, 1);
  EXPECT_GE(result.rewrites_applied, 1);
}

// Test: Default config reaches saturation
TEST_F(InlineRewriteTest, DefaultConfig) {
  // Create simple binding
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{42});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, literal);
  [[maybe_unused]] auto ident = eg_.MakeIdentifier(sym);

  // Default config should saturate
  auto result = ApplyAllRewrites(eg_, RewriteConfig::Default());
  EXPECT_TRUE(result.saturated());
  EXPECT_EQ(result.rewrites_applied, 1);
}

// Test: Unlimited config reaches saturation (no limits hit)
TEST_F(InlineRewriteTest, UnlimitedConfig) {
  // Create simple binding
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{42});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, literal);
  [[maybe_unused]] auto ident = eg_.MakeIdentifier(sym);

  // Unlimited config should saturate
  auto result = ApplyAllRewrites(eg_, RewriteConfig::Unlimited());
  EXPECT_TRUE(result.saturated());
  EXPECT_EQ(result.rewrites_applied, 1);
}

// Test: ApplyAllRewrites reaches fixed point
TEST_F(InlineRewriteTest, FixedPoint) {
  // Create simple binding
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "x");
  auto literal = eg_.MakeLiteral(storage::ExternalPropertyValue{42});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, literal);
  [[maybe_unused]] auto ident = eg_.MakeIdentifier(sym);

  // First call should do work
  auto result1 = ApplyAllRewrites(eg_);
  EXPECT_EQ(result1.rewrites_applied, 1);
  EXPECT_TRUE(result1.saturated());

  // Second call should be at fixed point
  auto result2 = ApplyAllRewrites(eg_);
  EXPECT_EQ(result2.rewrites_applied, 0);
  EXPECT_TRUE(result2.saturated());
}

// =======================================================================
// Operator construction tests
// =======================================================================

class OperatorConstructionTest : public EgraphTestBase {};

// Binary operators produce distinct eclasses for different operands
TEST_F(OperatorConstructionTest, BinaryOperatorsDistinct) {
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});
  auto lit3 = eg_.MakeLiteral(storage::ExternalPropertyValue{3});

  ExpectAllDistinct({eg_.MakeAdd(lit1, lit2), eg_.MakeAdd(lit1, lit3), eg_.MakeSub(lit1, lit2)});
}

// Hash-consing: same operator + same children → same eclass
TEST_F(OperatorConstructionTest, HashConsing) {
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});

  ExpectSame(eg_.MakeAdd(lit1, lit2), eg_.MakeAdd(lit1, lit2));
}

// Operand order matters: Add(1,2) != Add(2,1)
TEST_F(OperatorConstructionTest, OperandOrderMatters) {
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});

  ExpectDistinct(eg_.MakeAdd(lit1, lit2), eg_.MakeAdd(lit2, lit1));
}

// Unary operators
TEST_F(OperatorConstructionTest, UnaryOperators) {
  auto lit = eg_.MakeLiteral(storage::ExternalPropertyValue{5});

  ExpectAllDistinct({eg_.MakeUnaryMinus(lit), eg_.MakeUnaryPlus(lit), eg_.MakeNot(lit)});
}

// Nested operators create correct tree
TEST_F(OperatorConstructionTest, NestedOperators) {
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});
  auto lit3 = eg_.MakeLiteral(storage::ExternalPropertyValue{3});

  auto add = eg_.MakeAdd(lit1, lit2);
  auto mul = eg_.MakeMul(add, lit3);

  ExpectAllDistinct({add, mul, lit1, lit3});
}

// All comparison operators produce distinct eclasses
TEST_F(OperatorConstructionTest, ComparisonOperatorsDistinct) {
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});

  ExpectAllDistinct({eg_.MakeEq(lit1, lit2),
                     eg_.MakeNeq(lit1, lit2),
                     eg_.MakeLt(lit1, lit2),
                     eg_.MakeLte(lit1, lit2),
                     eg_.MakeGt(lit1, lit2),
                     eg_.MakeGte(lit1, lit2)});
}

// Boolean operators produce distinct eclasses
TEST_F(OperatorConstructionTest, BooleanOperatorsDistinct) {
  auto t = eg_.MakeLiteral(storage::ExternalPropertyValue{true});
  auto f = eg_.MakeLiteral(storage::ExternalPropertyValue{false});

  ExpectAllDistinct({eg_.MakeAnd(t, f), eg_.MakeOr(t, f), eg_.MakeXor(t, f)});
}

// =======================================================================
// Inline rewrite through operators
// =======================================================================

class InlineThroughOperatorTest : public EgraphTestBase {};

// Bind(Once, sym, Literal(1)) + Add(Identifier(sym), Literal(2))
// → Identifier should be merged with Literal(1)
TEST_F(InlineThroughOperatorTest, InlineThroughAdd) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "a");
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, lit1);

  auto ident = eg_.MakeIdentifier(sym);
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});
  [[maybe_unused]] auto add = eg_.MakeAdd(ident, lit2);

  auto result = ApplyAllRewrites(eg_);
  EXPECT_GE(result.rewrites_applied, 1);
  EXPECT_TRUE(result.saturated());
  ExpectSame(ident, lit1);
}

// Inline through unary minus: Bind(Once, sym, Lit(5)) + UnaryMinus(Identifier(sym))
TEST_F(InlineThroughOperatorTest, InlineThroughUnaryMinus) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "a");
  auto lit = eg_.MakeLiteral(storage::ExternalPropertyValue{5});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, lit);

  auto ident = eg_.MakeIdentifier(sym);
  [[maybe_unused]] auto neg = eg_.MakeUnaryMinus(ident);

  auto result = ApplyAllRewrites(eg_);
  EXPECT_GE(result.rewrites_applied, 1);
  EXPECT_TRUE(result.saturated());
  ExpectSame(ident, lit);
}

// Inline same variable used twice: Add(Identifier(sym), Identifier(sym))
TEST_F(InlineThroughOperatorTest, InlineSameVarBothSides) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "a");
  auto lit = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, lit);

  auto ident = eg_.MakeIdentifier(sym);
  [[maybe_unused]] auto add = eg_.MakeAdd(ident, ident);

  auto result = ApplyAllRewrites(eg_);
  EXPECT_GE(result.rewrites_applied, 1);
  EXPECT_TRUE(result.saturated());
  ExpectSame(ident, lit);
}

// Inline through comparison: Bind + Lt(Identifier, Literal)
TEST_F(InlineThroughOperatorTest, InlineThroughComparison) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "a");
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, lit1);

  auto ident = eg_.MakeIdentifier(sym);
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});
  [[maybe_unused]] auto lt = eg_.MakeLt(ident, lit2);

  auto result = ApplyAllRewrites(eg_);
  EXPECT_GE(result.rewrites_applied, 1);
  EXPECT_TRUE(result.saturated());
  ExpectSame(ident, lit1);
}

// Inline through boolean: Bind + And(Identifier, Literal)
TEST_F(InlineThroughOperatorTest, InlineThroughBoolean) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "a");
  auto lit_t = eg_.MakeLiteral(storage::ExternalPropertyValue{true});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, lit_t);

  auto ident = eg_.MakeIdentifier(sym);
  auto lit_f = eg_.MakeLiteral(storage::ExternalPropertyValue{false});
  [[maybe_unused]] auto and_op = eg_.MakeAnd(ident, lit_f);

  auto result = ApplyAllRewrites(eg_);
  EXPECT_GE(result.rewrites_applied, 1);
  EXPECT_TRUE(result.saturated());
  ExpectSame(ident, lit_t);
}

// Nested: Bind(Once, sym, Lit(1)) + Mul(Add(Identifier(sym), Lit(2)), Lit(3))
TEST_F(InlineThroughOperatorTest, InlineThroughNestedOperators) {
  auto once = eg_.MakeOnce();
  auto sym = eg_.MakeSymbol(0, "a");
  auto lit1 = eg_.MakeLiteral(storage::ExternalPropertyValue{1});
  [[maybe_unused]] auto bind = eg_.MakeBind(once, sym, lit1);

  auto ident = eg_.MakeIdentifier(sym);
  auto lit2 = eg_.MakeLiteral(storage::ExternalPropertyValue{2});
  auto lit3 = eg_.MakeLiteral(storage::ExternalPropertyValue{3});
  auto add = eg_.MakeAdd(ident, lit2);
  [[maybe_unused]] auto mul = eg_.MakeMul(add, lit3);

  auto result = ApplyAllRewrites(eg_);
  EXPECT_GE(result.rewrites_applied, 1);
  EXPECT_TRUE(result.saturated());
  ExpectSame(ident, lit1);
}

}  // namespace
}  // namespace memgraph::query::plan::v2
