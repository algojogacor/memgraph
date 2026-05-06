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

#include "bench_rewrite.hpp"

#include <array>
#include <optional>

using namespace memgraph::planner::bench;
using namespace memgraph::planner::bench::sizes;
using namespace memgraph::planner::core::pattern::vm;

// ============================================================================
// Simple Pattern: Add(?x, ?y) - VM Version
// ============================================================================

class VMSimplePatternFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t graph_size_ = 0;
  int64_t context_mode_ = 0;

  void SetUp(const benchmark::State &state) override {
    graph_size_ = state.range(0);
    context_mode_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildIndependentAdds(g, graph_size_); });
    pattern_ = compiler_.compile(PatternAdd());
  }
};

BENCHMARK_DEFINE_F(VMSimplePatternFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  BenchmarkWithMatchContext(state, context_mode_, match_context_, [&](EMatchContext &ctx) {
    matches_.clear();
    executor.execute(*pattern_, *matcher_, ctx.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  });
  state.SetItemsProcessed(state.iterations() * graph_size_);
}

BENCHMARK_REGISTER_F(VMSimplePatternFixture, Match)
    ->ArgsProduct({{kSmall, kMedium, kLarge, kMassive}, {kFreshCtx, kReusedCtx}})
    ->ArgNames({"size", "ctx"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Deep Pattern: Neg(Neg(...Neg(?x)...)) - VM Version
// ============================================================================

class VMDeepPatternFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t pattern_depth_ = 0;
  int64_t context_mode_ = 0;

  void SetUp(const benchmark::State &state) override {
    pattern_depth_ = state.range(0);
    context_mode_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildNegChain(g, pattern_depth_); });
    pattern_ = compiler_.compile(PatternNestedNeg(static_cast<int>(pattern_depth_)));
  }
};

BENCHMARK_DEFINE_F(VMDeepPatternFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  BenchmarkWithMatchContext(state, context_mode_, match_context_, [&](EMatchContext &ctx) {
    matches_.clear();
    executor.execute(*pattern_, *matcher_, ctx.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  });
  state.SetItemsProcessed(state.iterations());
}

// VM has 64 registers, deep patterns need 2*depth+1 registers, so max depth ~31.
BENCHMARK_REGISTER_F(VMDeepPatternFixture, Match)
    ->ArgsProduct({{1, 5, 10, 20, 30}, {kFreshCtx, kReusedCtx}})
    ->ArgNames({"depth", "ctx"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Same Variable: Add(?x, ?x) - VM Version
// ============================================================================

class VMSameVariableFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t graph_size_ = 0;
  int64_t context_mode_ = 0;

  void SetUp(const benchmark::State &state) override {
    graph_size_ = state.range(0);
    context_mode_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildMixedAdds(g, graph_size_); });
    pattern_ = compiler_.compile(PatternAddSameVar());
  }
};

BENCHMARK_DEFINE_F(VMSameVariableFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  BenchmarkWithMatchContext(state, context_mode_, match_context_, [&](EMatchContext &ctx) {
    matches_.clear();
    executor.execute(*pattern_, *matcher_, ctx.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  });
  state.SetItemsProcessed(state.iterations() * graph_size_);
}

BENCHMARK_REGISTER_F(VMSameVariableFixture, Match)
    ->ArgsProduct({{kSmall, kMedium, kLarge}, {kFreshCtx, kReusedCtx}})
    ->ArgNames({"size", "ctx"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Merged E-Graph (multiple e-nodes per class) - VM Version
// ============================================================================

class VMMergedEGraphFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t graph_size_ = 0;
  int64_t context_mode_ = 0;

  void SetUp(const benchmark::State &state) override {
    graph_size_ = state.range(0);
    context_mode_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildMergedAddMul(g, graph_size_); });
    pattern_ = compiler_.compile(PatternAdd());
  }
};

BENCHMARK_DEFINE_F(VMMergedEGraphFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  BenchmarkWithMatchContext(state, context_mode_, match_context_, [&](EMatchContext &ctx) {
    matches_.clear();
    executor.execute(*pattern_, *matcher_, ctx.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  });
  state.SetItemsProcessed(state.iterations() * graph_size_);
}

BENCHMARK_REGISTER_F(VMMergedEGraphFixture, Match)
    ->ArgsProduct({{kSmall, kMedium, kLarge}, {kFreshCtx, kReusedCtx}})
    ->ArgNames({"size", "ctx"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Selective Pattern: Add(Neg(?x), ?y) - VM Version
// ============================================================================

class VMSelectivePatternFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t graph_size_ = 0;
  int64_t context_mode_ = 0;

  void SetUp(const benchmark::State &state) override {
    graph_size_ = state.range(0);
    context_mode_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildAddsWithOneNeg(g, graph_size_); });
    pattern_ = compiler_.compile(PatternSelective());
  }
};

BENCHMARK_DEFINE_F(VMSelectivePatternFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  BenchmarkWithMatchContext(state, context_mode_, match_context_, [&](EMatchContext &ctx) {
    matches_.clear();
    executor.execute(*pattern_, *matcher_, ctx.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  });
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(VMSelectivePatternFixture, Match)
    ->ArgsProduct({{kSmall, kMedium, kLarge}, {kFreshCtx, kReusedCtx}})
    ->ArgNames({"size", "ctx"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Multi-Pattern Join: Bind(_, ?sym, ?expr) + ?id = Ident(?sym)
// ============================================================================

// RewriteRule approach: baseline for comparison with VM fused parent traversal
class BindIdentRewriteRuleFixture : public RewriterFixtureBase {
 protected:
  int64_t num_binds_ = 0;
  int64_t idents_per_sym_ = 0;
  std::unique_ptr<TestRewriteRule> rule_;

  void SetUp(const benchmark::State &state) override {
    num_binds_ = state.range(0);
    idents_per_sym_ = state.range(1);
    ResetEGraph();
    BuildBindIdentGraph(egraph_, num_binds_, idents_per_sym_);
    CreateMatcher();

    rule_ = std::make_unique<TestRewriteRule>(TestRewriteRule::Builder{"bind_ident"}
                                                  .pattern(PatternBind(), "bind")
                                                  .pattern(PatternIdent(), "ident")
                                                  .apply([](TestRuleContext &, Match const &) {}));
  }
};

// VM Fused approach: PatternsCompiler with parent traversal
class BindIdentVMFusedFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> compiled_;
  int64_t num_binds_ = 0;
  int64_t idents_per_sym_ = 0;

  void SetUp(const benchmark::State &state) override {
    num_binds_ = state.range(0);
    idents_per_sym_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildBindIdentGraph(g, num_binds_, idents_per_sym_); });

    std::array patterns = {PatternBind(), PatternIdent()};
    compiled_ = compiler_.compile(patterns);
  }
};

BENCHMARK_DEFINE_F(BindIdentVMFusedFixture, ParentTraversal)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  for (auto _ : state) {
    match_context_.clear();
    matches_.clear();
    executor.execute(*compiled_, *matcher_, match_context_.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  }
  state.SetItemsProcessed(state.iterations() * num_binds_ * idents_per_sym_);
}

BENCHMARK_REGISTER_F(BindIdentVMFusedFixture, ParentTraversal)
    ->Args({10, 1})    // 10 binds, 1 ident each = 10 matches
    ->Args({100, 1})   // 100 binds, 1 ident each = 100 matches
    ->Args({10, 10})   // 10 binds, 10 idents each = 100 matches
    ->Args({100, 10})  // 100 binds, 10 idents each = 1000 matches
    ->Args({1000, 1})  // 1000 binds, 1 ident each = 1000 matches
    ->ArgNames({"binds", "idents_per"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// High Parent Count: Hub node with many parents of different symbols
// ============================================================================
//
// Tests symbol-index filtering effectiveness when a node has many parents.
// Both modes use IterParents + CheckSymbol; clean mode may use the parent
// index for further optimisation.

class VMHighParentFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t parents_f_ = 0;
  int64_t parents_neg_ = 0;

  void SetUp(const benchmark::State &state) override {
    parents_f_ = state.range(0);
    parents_neg_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildHighParentHub(g, parents_f_, parents_neg_); });
    pattern_ = compiler_.compile(PatternNeg());
  }
};

BENCHMARK_DEFINE_F(VMHighParentFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  for (auto _ : state) {
    match_context_.clear();
    matches_.clear();
    executor.execute(*pattern_, *matcher_, match_context_.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  }
  state.SetItemsProcessed(state.iterations() * parents_neg_);
}

BENCHMARK_REGISTER_F(VMHighParentFixture, Match)
    ->Args({1000, 100})
    ->Args({5000, 100})
    ->Args({10000, 100})
    ->ArgNames({"F_parents", "Neg_parents"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Self-Referential E-Class: Pattern matching on cyclic structures
// ============================================================================
//
// Setup: EC1 = {F(Const), F(EC1)} - e-class contains an F node pointing
// to itself, so the same e-class appears at multiple depths during matching.

class VMSelfRefFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;

  void SetUp(const benchmark::State &) override {
    SetupGraphAndMatcher([](TestEGraph &g) { BuildSelfReferential(g, 42); });
    pattern_ = compiler_.compile(PatternNestedF());
  }
};

BENCHMARK_DEFINE_F(VMSelfRefFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  for (auto _ : state) {
    match_context_.clear();
    matches_.clear();
    executor.execute(*pattern_, *matcher_, match_context_.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(VMSelfRefFixture, Match)->Unit(benchmark::kNanosecond);

// ============================================================================
// Parent Diversity: Many nodes with diverse parent symbols
// ============================================================================
//
// Tests how well the symbol index filters when parents are distributed
// across many symbols vs a linear scan.

class VMParentDiversityFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t num_leaves_ = 0;
  int64_t parents_per_leaf_ = 0;

  void SetUp(const benchmark::State &state) override {
    num_leaves_ = state.range(0);
    parents_per_leaf_ = state.range(1);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildParentDiversity(g, num_leaves_, parents_per_leaf_, 42); });
    pattern_ = compiler_.compile(PatternNeg());
  }
};

BENCHMARK_DEFINE_F(VMParentDiversityFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  for (auto _ : state) {
    match_context_.clear();
    matches_.clear();
    executor.execute(*pattern_, *matcher_, match_context_.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(VMParentDiversityFixture, Match)
    ->Args({100, 20})
    ->Args({500, 20})
    ->Args({100, 50})
    ->ArgNames({"leaves", "parents_per"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Nested Join Pattern: (F ?v0) JOIN (F (F (F (F ?v0))))
// ============================================================================
//
// Fuzzer-identified VM bottleneck. Pattern 2 needs 4x LoadChild + CheckSymbol,
// and the shared variable forces a join with deep backtracking.

class VMNestedJoinFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> pattern_;
  int64_t num_leaves_ = 0;

  void SetUp(const benchmark::State &state) override {
    num_leaves_ = state.range(0);
    SetupGraphAndMatcher([this](TestEGraph &g) { BuildNestedJoinGraph(g, num_leaves_); });

    std::array patterns = {PatternShallowF(), PatternDeepNestedF()};
    pattern_ = compiler_.compile(patterns);
  }
};

BENCHMARK_DEFINE_F(VMNestedJoinFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  for (auto _ : state) {
    match_context_.clear();
    matches_.clear();
    executor.execute(*pattern_, *matcher_, match_context_.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  }
  state.SetItemsProcessed(state.iterations() * num_leaves_);
}

BENCHMARK_REGISTER_F(VMNestedJoinFixture, Match)
    ->Args({10})
    ->Args({50})
    ->Args({100})
    ->Args({500})
    ->ArgNames({"leaves"})
    ->Unit(benchmark::kMicrosecond);

// ============================================================================
// Eclass-Level Hoisting: ?r=F(?x) + Mul(?r, _)
// ============================================================================
//
// Mul(?r, _) depends only on ?r (the eclass register), not on which F enode
// is selected. With hoisting, parent traversal runs once per eclass; without
// it runs once per F enode. Two independent axes:
//   enodes_per_class: F enodes per eclass (redundancy hoisting eliminates)
//   parents:          Mul parents per F eclass (work per traversal)

class VMEclassHoistFixture : public MatcherFixtureBase {
 protected:
  TestPatternsCompiler compiler_;
  std::optional<TestCompiledMatcher> compiled_;
  int64_t num_eclasses_ = 0;
  int64_t enodes_per_class_ = 0;
  int64_t parents_ = 0;

  void SetUp(const benchmark::State &state) override {
    num_eclasses_ = state.range(0);
    enodes_per_class_ = state.range(1);
    parents_ = state.range(2);
    SetupGraphAndMatcher(
        [this](TestEGraph &g) { BuildEclassHoistGraph(g, num_eclasses_, enodes_per_class_, parents_); });

    std::array patterns = {PatternHoistAnchor(), PatternHoistJoined()};
    compiled_ = compiler_.compile(patterns);
  }
};

BENCHMARK_DEFINE_F(VMEclassHoistFixture, Match)(benchmark::State &state) {
  VMExecutor executor(egraph_);
  for (auto _ : state) {
    match_context_.clear();
    matches_.clear();
    executor.execute(*compiled_, *matcher_, match_context_.arena(), matches_);
    benchmark::DoNotOptimize(matches_);
  }
  auto expected_matches = num_eclasses_ * enodes_per_class_ * parents_;
  state.SetItemsProcessed(state.iterations() * expected_matches);
}

BENCHMARK_REGISTER_F(VMEclassHoistFixture, Match)
    // {eclasses, enodes_per_class, parents}
    // Baseline: 1 enode/class - hoisting has no effect
    ->Args({100, 1, 1})
    ->Args({1000, 1, 1})
    // Moderate redundancy, single parent
    ->Args({100, 10, 1})
    ->Args({1000, 10, 1})
    // High redundancy, single parent
    ->Args({100, 20, 1})
    ->Args({1000, 20, 1})
    // Fixed redundancy, varying parents
    ->Args({100, 10, 5})
    ->Args({100, 10, 20})
    ->Args({100, 10, 50})
    // High redundancy x many parents (worst case for no-hoist)
    ->Args({100, 20, 20})
    ->Args({1000, 20, 20})
    ->ArgNames({"eclasses", "enodes_per", "parents"})
    ->Unit(benchmark::kMicrosecond);
