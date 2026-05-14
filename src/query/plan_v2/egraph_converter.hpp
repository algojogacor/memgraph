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

#include <memory>
#include <tuple>

#include "frontend/ast/ast_storage.hpp"
#include "query/frontend/semantic/symbol_table.hpp"
#include "query/plan_v2/egraph.hpp"

namespace memgraph::query::plan {
class LogicalOperator;
}

namespace memgraph::query::plan::v2 {

struct CardinalityEstimator;

/// Result of a successful ConvertToLogicalOperator call.
struct ExtractionResult {
  std::unique_ptr<LogicalOperator> plan;
  double cost;
  AstStorage ast_storage;
  SymbolTable symbol_table;
};

/// Per-session planner state.  Today this owns the FrontierContext buffers
/// (frontier map, resolver scratch) so their allocated capacity is reused
/// across queries instead of being freed and re-grown each time; it will grow
/// to hold any other per-session planner state (caches, scratch arenas) as
/// the planner v2 stabilises.  Hold one per Interpreter and pass it to
/// ConvertToLogicalOperator.  Pimpl so callers don't see CostFrontier /
/// Alternative.
///
/// Optionally holds a user-provided CardinalityEstimator override.  Default
/// construction leaves the override empty: ConvertToLogicalOperator builds
/// a BuiltinEstimator over the current egraph for that call.  Tests inject
/// a mock by passing one to the estimator-taking constructor; the override
/// then takes precedence over the per-call BuiltinEstimator.
///
/// The override cannot be the production estimator at construction time
/// because BuiltinEstimator binds to a specific egraph (it walks e-classes
/// for literal deduction), and a single QueryPlannerContext is reused
/// across queries that each have their own egraph.
class QueryPlannerContext {
 public:
  QueryPlannerContext();
  explicit QueryPlannerContext(std::unique_ptr<CardinalityEstimator> estimator);
  ~QueryPlannerContext();
  QueryPlannerContext(QueryPlannerContext &&) noexcept;
  QueryPlannerContext &operator=(QueryPlannerContext &&) noexcept;
  QueryPlannerContext(QueryPlannerContext const &) = delete;
  QueryPlannerContext &operator=(QueryPlannerContext const &) = delete;

  struct Impl;

  /// Cardinality of the root alt the most recent ConvertToLogicalOperator
  /// call selected (NaN if no plan has been extracted yet).  Surfaces the
  /// per-query result so tests can pin cardinality semantics directly,
  /// without round-tripping through cost arithmetic.
  // TODO: this should be removed, in favour of using ExtractionResult for cardinality estimate
  double last_root_cardinality() const;

 private:
  Impl &impl() { return *impl_; }

  friend auto ConvertToLogicalOperator(egraph const &e, eclass root, QueryPlannerContext &planner_context)
      -> ExtractionResult;

  std::unique_ptr<Impl> impl_;
};

/// Returns the extracted plan together with the SymbolTable to use for
/// downstream lookups, in place of the parse-time SymbolTable.
///
/// `planner_context` is required: callers must own one and pass it in.
/// Long-lived owners (Interpreter) get amortised buffer allocations across
/// queries; short-lived callers (triggers, tests) just declare a local one.
auto ConvertToLogicalOperator(egraph const &e, eclass root, QueryPlannerContext &planner_context) -> ExtractionResult;
}  // namespace memgraph::query::plan::v2
