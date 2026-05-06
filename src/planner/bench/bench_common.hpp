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

#pragma once

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>

#include "test_support/builders.hpp"
#include "test_support/patterns.hpp"
#include "test_support/types.hpp"

namespace memgraph::planner::bench {

// Test-support symbols, types, builders, and pattern factories live in
// memgraph::planner::core::test. Pull them into bench so existing benchmark
// bodies (Op::Add, TestEGraph, kX, BuildNegChain, PatternAdd, ...) keep
// resolving without per-call qualification.
using namespace memgraph::planner::core::test;  // NOLINT(google-build-using-namespace)

namespace core = memgraph::planner::core;
namespace pattern = core::pattern;
namespace vm = pattern::vm;

using core::EClassId;
using core::EGraph;
using core::ProcessingContext;
using pattern::EMatchContext;
using pattern::PatternVar;

// ============================================================================
// Benchmark Size Constants
// ============================================================================

namespace sizes {
constexpr int64_t kSmall = 10;
constexpr int64_t kMedium = 100;
constexpr int64_t kLarge = 1000;
constexpr int64_t kXLarge = 5000;
constexpr int64_t kHuge = 10000;
constexpr int64_t kMassive = 50000;

constexpr int64_t kFreshCtx = 0;
constexpr int64_t kReusedCtx = 1;
}  // namespace sizes

// Pull size constants into bench:: so registration sites stay terse.
using sizes::kFreshCtx;
using sizes::kHuge;
using sizes::kLarge;
using sizes::kMassive;
using sizes::kMedium;
using sizes::kReusedCtx;
using sizes::kSmall;
using sizes::kXLarge;

// ============================================================================
// Benchmark Abstractions
// ============================================================================

// Run a matching benchmark with fresh or reused EMatchContext.
// ApplyFn signature: void(EMatchContext&)
template <typename ApplyFn>
void BenchmarkWithMatchContext(benchmark::State &state, int64_t context_mode, EMatchContext &reusable_context,
                               ApplyFn &&apply_fn) {
  if (context_mode == sizes::kReusedCtx) {
    for (auto _ : state) {
      reusable_context.clear();
      apply_fn(reusable_context);
    }
  } else {
    for (auto _ : state) {
      EMatchContext fresh_context;
      apply_fn(fresh_context);
    }
  }
}

// Base fixture providing common e-graph and matcher setup. The MatcherIndex
// constructor calls rebuild_index() itself, so no explicit rebuild is needed.
class MatcherFixtureBase : public benchmark::Fixture {
 protected:
  TestEGraph egraph_;
  std::unique_ptr<TestMatcherIndex> matcher_;
  EMatchContext match_context_;
  TestMatches matches_;

  void ResetEGraph() { egraph_ = TestEGraph{}; }

  void CreateMatcher() { matcher_ = std::make_unique<TestMatcherIndex>(egraph_); }

  template <typename BuilderFn>
  void SetupGraphAndMatcher(BuilderFn &&build_fn) {
    ResetEGraph();
    build_fn(egraph_);
    CreateMatcher();
  }
};

// Base fixture for VM benchmarks. Adds a TestPatternsCompiler member and
// exposes SetupGraph (a SetupGraphAndMatcher alias retained for legacy call
// sites). The MatcherIndex constructor rebuilds the index, so no extra call
// is needed.
class VMFixtureBase : public benchmark::Fixture {
 protected:
  TestEGraph egraph_;
  std::unique_ptr<TestMatcherIndex> matcher_;
  EMatchContext match_context_;
  TestMatches matches_;
  TestPatternsCompiler compiler_;

  void ResetEGraph() {
    egraph_ = TestEGraph{};
    matcher_.reset();
  }

  template <typename BuilderFn>
  void SetupGraph(BuilderFn &&build_fn) {
    ResetEGraph();
    build_fn(egraph_);
    matcher_ = std::make_unique<TestMatcherIndex>(egraph_);
  }
};

}  // namespace memgraph::planner::bench
