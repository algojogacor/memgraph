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

#include "query/plan_v2/egraph_converter.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include "planner/extract/extractor.hpp"
#include "query/exceptions.hpp"
#include "query/plan/operator.hpp"
#include "query/plan_v2/bind_semantics.hpp"
#include "query/plan_v2/builtin_estimator.hpp"
#include "query/plan_v2/cardinality.hpp"
#include "query/plan_v2/cardinality_estimator.hpp"
#include "query/plan_v2/egraph_internal.hpp"
#include "query/plan_v2/expression_cost.hpp"
#include "query/plan_v2/plan_alternative.hpp"
#include "utils/tag.hpp"

namespace memgraph::query::plan::v2 {

// ============================================================================
// Plan extraction cost model - Pareto frontier with symbol demand tracking
// ----------------------------------------------------------------------------
// The TU is structured bottom-up; each section depends on the ones above:
//   1. Alternatives    : the (cost, required, enode_id, is_alive) tuple, its
//                        dominance relation, and the Pareto frontier type.
//   2. Frontier ops    : Cartesian product (CombineAlts) and view-style
//                        re-stamping (CostFrontier::LazyMap) used by the
//                        cost model.
//   3. Policies        : PlanCostModel (per-enode dispatch into ops) and
//                        PlanResolver (demand-aware top-down DAG walk).
//                        These are the two customisation points the generic
//                        extractor takes.
//   4. Builder         : turns the resolver's topological order into the
//                        LogicalOperator / Expression tree the executor runs.
// The public surface (ConvertToLogicalOperator, BuiltinEstimator impl,
// QueryPlannerContext impl) sits below the four sections.
// ============================================================================
namespace {

using bind::SymbolSet;

/// Build the SymbolSet of exposed symbols from Subquery children[2..].
/// Children at positions >= 2 are Symbol e-classes that the subquery exposes
/// to the outer scope. Sort and dedup is required because e-graph children are
/// not guaranteed unique.
auto ExposedSymsFromChildren(std::span<planner::core::EClassId const> children_from_2) -> SymbolSet {
  return SymbolSet{children_from_2};
}

// --- Alternatives -----------------------------------------------------------
// Alternative and its Pareto dims live in plan_alternative.hpp so the
// pure-algebra dimensions can be unit-tested without the rest of this TU.

/// CostFrontier: ParetoFrontier with resolve/min_cost for the extraction contract.
/// merge is inherited from ParetoFrontier (union + prune).
struct CostFrontier
    : planner::core::extract::CostResultBase<Alternative, AlternativeDim_Cost, AlternativeDim_Cardinality,
                                             AlternativeDim_Required, AlternativeDim_Introduces> {
  using CostResultBase::CostResultBase;
};

// ============================================================================
// Cost-model algebra
// ============================================================================
// Named operations the per-symbol switch below composes.  Each helper describes
// *what* the cost model does for a particular enode shape; the underlying
// ParetoFrontier primitives (LazyMap / cartesian_product / flat_map) are an
// implementation detail of these helpers.
//
// Cardinality defaults to 1.0 on per-evaluation operators (binary expressions,
// NamedOutput): each invocation packages one value regardless of the per-value
// shape.  Row-pipe operators (Output, Unwind, Subquery) override cardinality
// inside their dedicated helper.

/// Single-alt frontier with no demand.  Terminal cost-model leaves
/// (Once, Literal, Symbol, ParamLookup).
auto LeafAlt(double cost, planner::core::ENodeId enode_id) -> CostFrontier {
  return CostFrontier{{{.cost = cost, .required = {}, .enode_id = enode_id}}};
}

/// Single-alt frontier demanding `sym_eclass`.  Identifier nodes use this.
auto IdentifierAlt(double cost, planner::core::EClassId sym_eclass, planner::core::ENodeId enode_id) -> CostFrontier {
  return CostFrontier{{{.cost = cost, .required = SymbolSet{{sym_eclass}}, .enode_id = enode_id}}};
}

/// Re-stamp source under `enode_id`, optionally bumping cost.  Used by unary
/// expression operators and by the leading-child re-stamp in Output / Function
/// chains.  View-style: no materialisation if downstream doesn't iterate.
auto Restamp(CostFrontier const &source, double extra_cost, planner::core::ENodeId enode_id) -> CostFrontier {
  return CostFrontier::LazyMap(source, extra_cost, enode_id);
}

/// Cartesian product with cost summation and required-set union.  Default
/// shape for binary expressions and NamedOutput pairs.
auto BinaryCombine(CostFrontier const &lhs, CostFrontier const &rhs, double extra_cost, planner::core::ENodeId enode_id)
    -> CostFrontier {
  return CostFrontier::cartesian_product(lhs, rhs, [extra_cost, enode_id](Alternative const &l, Alternative const &r) {
    return Alternative{
        .cost = extra_cost + l.cost + r.cost, .required = l.required.set_union(r.required), .enode_id = enode_id};
  });
}

/// Output × NamedOutput combine.  The row pipe's per-output-row evaluation
/// scales `named_out`'s scalar cost by the input pipe's cardinality.  The
/// NamedOutput is evaluated INSIDE the input pipe's row scope, so demands
/// satisfied by the input's bindings (alive Bind / Unwind) are subtracted
/// from the residual `required`.  This is what lets `WITH x AS y UNWIND ...
/// RETURN y` satisfy y's demand from the Bind/Unwind below the Output.
auto OutputCombine(CostFrontier const &row_pipe, CostFrontier const &named_out, planner::core::ENodeId enode_id)
    -> CostFrontier {
  return CostFrontier::cartesian_product(row_pipe, named_out, [enode_id](Alternative const &l, Alternative const &r) {
    auto const remaining = r.required.difference(l.introduces);
    return Alternative{.cost = l.cost + l.cardinality * r.cost,
                       .cardinality = l.cardinality,
                       .required = l.required.set_union(remaining),
                       .introduces = l.introduces,
                       .enode_id = enode_id};
  });
}

/// Bind flat-map: for each input alt, emit an alive variant when sym is
/// demanded (locally or globally) and a dead variant when input doesn't
/// already demand sym.  See the suppression comments inline for why each
/// branch is conditional.
auto BindFlatMap(CostFrontier const &input, CostFrontier const &expr, planner::core::EClassId sym_eclass,
                 double sym_cost, SymbolSet const &referenced_syms, planner::core::ENodeId enode_id) -> CostFrontier {
  return CostFrontier::flat_map(input, [&, enode_id](Alternative const &input_alt, auto emit) {
    // Emit alive when there's a chance someone will demand sym:
    //   - Input subtree directly demands it (classic case), or
    //   - Some Identifier(sym) lives elsewhere in the e-graph and
    //     might cross a sibling boundary at an enclosing Output.
    // If neither holds, sym has no consumer and the alive alt would
    // bloat the frontier unbounded - up to 2^N for an N-Bind chain.
    bool const input_demands_sym = input_alt.required.is_alive(sym_eclass);
    bool const should_emit_alive = input_demands_sym || referenced_syms.contains(sym_eclass);
    if (should_emit_alive) {
      for (auto const &expr_alt : expr.alts()) {
        auto required = input_alt.required.alive_required(sym_eclass, expr_alt.required);
        auto introduces = input_alt.introduces;
        introduces.insert(sym_eclass);
        // Bind is one-shot, not a row-pipe: passes input's cardinality
        // through unchanged.  expr is evaluated once at bind-time.
        emit({.cost = bind::AliveCost(input_alt.cost, sym_cost, expr_alt.cost),
              .cardinality = input_alt.cardinality,
              .required = std::move(required),
              .introduces = std::move(introduces),
              .enode_id = enode_id,
              .is_alive = AliveTag::Alive});
      }
    }
    // Emit dead only when input doesn't already demand sym.  When
    // input_demands_sym is true, the dead alt has sym still in `required`,
    // so it is only compatible with ancestors that already provide sym.  But
    // sym is provided only when an ancestor Bind for it is alive - which is
    // exactly the alive-alt condition.  A dead alt under that condition is
    // therefore unreachable: no resolver context can pick it that couldn't
    // also pick alive.  Suppressing it avoids bloating the frontier.
    if (!input_demands_sym) {
      emit({.cost = bind::DeadCost(input_alt.cost),
            .cardinality = input_alt.cardinality,
            .required = input_alt.required,
            .introduces = input_alt.introduces,
            .enode_id = enode_id,
            .is_alive = AliveTag::Dead});
    }
  });
}

/// Unwind flat-map: row-generative.  Output cardinality is the product of
/// input's and list's cardinalities; cost is input's pipeline plus per-row
/// evaluation of the list expression with a structural overhead.  Always
/// emits Alive because Unwind always introduces sym.
auto UnwindFlatMap(CostFrontier const &input, CostFrontier const &list, planner::core::EClassId sym_eclass,
                   double sym_cost, planner::core::ENodeId enode_id) -> CostFrontier {
  return CostFrontier::flat_map(input, [&, enode_id](Alternative const &input_alt, auto emit) {
    for (auto const &list_alt : list.alts()) {
      // sym is always introduced by Unwind, so remove it from input's
      // required and union list_expr's required (its needs become ours).
      auto required = input_alt.required.alive_required(sym_eclass, list_alt.required);
      auto introduces = input_alt.introduces;
      introduces.insert(sym_eclass);
      auto const cost = input_alt.cost + (list_alt.cost + kUnwindPerRowOverhead) * input_alt.cardinality + sym_cost;
      auto const cardinality = input_alt.cardinality * list_alt.cardinality;
      emit({.cost = cost,
            .cardinality = cardinality,
            .required = std::move(required),
            .introduces = std::move(introduces),
            .enode_id = enode_id,
            .is_alive = AliveTag::Alive});
    }
  });
}

/// Subquery flat-map: scope-barrier row-pipe.  Inner alts with non-empty
/// required are silently rejected (non-importing CALL only).  Inner
/// introductions are STRIPPED at the boundary; only `exposed_syms`
/// (children[2..]) cross into the outer scope, unioned with outer's own
/// introductions.
auto SubqueryFlatMap(CostFrontier const &outer, CostFrontier const &inner, SymbolSet exposed_syms,
                     planner::core::ENodeId enode_id) -> CostFrontier {
  return CostFrontier::flat_map(
      outer, [&inner, exposed_syms = std::move(exposed_syms), enode_id](Alternative const &outer_alt, auto emit) {
        for (auto const &inner_alt : inner.alts()) {
          // Non-importing subquery: inner must be self-contained.
          if (!inner_alt.required.empty()) continue;

          // BARRIER: outer.introduces ∪ exposed_syms; inner.introduces is dropped.
          emit({.cost = outer_alt.cost + outer_alt.cardinality * inner_alt.cost,
                .cardinality = outer_alt.cardinality * inner_alt.cardinality,
                .required = outer_alt.required,
                .introduces = outer_alt.introduces.set_union(exposed_syms),
                .enode_id = enode_id,
                .is_alive = AliveTag::Alive});
        }
      });
}

/// Function combine: cartesian product over arg frontiers (cost-sum and
/// required-union via BinaryCombine), then per-alt cardinality override
/// because function cardinality is not the product of arg cardinalities -
/// args are scalars by construction.  Structural +1 cost accounts for the
/// function-call overhead.
auto FunctionCombine(std::span<CostFrontier const *const> args, double cardinality, planner::core::ENodeId enode_id)
    -> CostFrontier {
  auto result = args.empty() ? LeafAlt(0.0, enode_id) : Restamp(*args[0], 0.0, enode_id);
  for (auto const *arg : args.empty() ? args : args.subspan(1)) {
    result = BinaryCombine(result, *arg, 0.0, enode_id);
  }
  result.mutate_pruning_invariant_preserving([&](Alternative &alt) {
    alt.cost += 1.0;
    alt.cardinality = cardinality;
    alt.enode_id = enode_id;
  });
  return result;
}

// --- Policies ---------------------------------------------------------------

struct PlanCostModel {
  using CostResult = CostFrontier;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  CardinalityEstimator const &estimator;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  EGraph const &egraph;  // passed to estimator for e-class walks (e.g. constant deduction)
  /// Set of Symbol e-classes referenced by *some* Identifier e-node anywhere
  /// in the e-graph.  Used as a cheap global filter: if sym is not in this
  /// set, no Output ever demands it across a sibling boundary, so a Bind
  /// for sym only needs to emit its dead alt.  Without this filter we'd
  /// emit alive on every Bind to handle the cross-boundary case, blowing
  /// the frontier to 2^N for an N-Bind chain with no Identifiers.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  SymbolSet const &referenced_syms;

  auto operator()(planner::core::ENode<symbol> const &current, planner::core::ENodeId enode_id,
                  std::span<CostResult const *const> children) const -> CostResult {
    switch (current.symbol()) {
      // Leaf nodes: single alternative, no demand.
      case symbol::Once:
      case symbol::Literal:
      case symbol::Symbol:  // Leaf invariant - see bind::kSymbolCost.
      case symbol::ParamLookup:
        return LeafAlt(bind::kSymbolCost, enode_id);

      // Identifier: demands its symbol child to be bound.
      case symbol::Identifier: {
        assert(!children.empty() && "Identifier must have its symbol child frontier");
        auto const sym_eclass = current.children()[0];
        auto const &[_, child_cost] = children[0]->resolve();
        return IdentifierAlt(expression_cost::kIdentifier + child_cost, sym_eclass, enode_id);
      }

      // Bind: alive/dead variants per input alt; is_alive tag rides on each
      // alt so the resolver dispatches alive/dead by reading the chosen alt.
      case symbol::Bind: {
        auto const sym_eclass = current.children()[1];
        auto const &[_, sym_cost] = children[1]->resolve();
        return BindFlatMap(*children[0], *children[2], sym_eclass, sym_cost, referenced_syms, enode_id);
      }

      // Binary expression operators (arithmetic / comparison / boolean):
      // cartesian product with per-class cost from the symbol's descriptor.
      // Adding a new binary operator is one descriptor specialisation in
      // private_symbol.hpp - no new case arms here.
      case symbol::Add:
      case symbol::Sub:
      case symbol::Mul:
      case symbol::Div:
      case symbol::Mod:
      case symbol::Exp:
      case symbol::Eq:
      case symbol::Neq:
      case symbol::Lt:
      case symbol::Lte:
      case symbol::Gt:
      case symbol::Gte:
      case symbol::And:
      case symbol::Or:
      case symbol::Xor:
        return BinaryCombine(
            *children[0], *children[1], expression_cost::FromClass(CostClassOf(current.symbol())), enode_id);

      // Unary expression operators: re-stamp child with per-class cost.
      // View-style; chains of unary ops collapse into one materialisation.
      case symbol::Not:
      case symbol::UnaryMinus:
      case symbol::UnaryPlus:
        return Restamp(*children[0], expression_cost::FromClass(CostClassOf(current.symbol())), enode_id);

      // Output: row-pipe.  Re-stamp child[0]'s row pipe, then fold each
      // NamedOutput in with per-row-scaled evaluation cost.  Per-row scaling
      // lets the planner prefer a one-shot Bind over an inlined alternative
      // when the row pipe is wide (e.g. UNWIND range(0, 100)).
      case symbol::Output: {
        auto result = Restamp(*children[0], 0.0, enode_id);
        for (auto const *named_out : children.subspan(1)) {
          result = OutputCombine(result, *named_out, enode_id);
        }
        return result;
      }

      // NamedOutput: sym × expr cartesian product, +1 per pair.  Structural
      // (not an expression operator), so a fixed cost rather than expression_cost.
      case symbol::NamedOutput:
        return BinaryCombine(*children[0], *children[1], 1.0, enode_id);

      // Unwind: row-generative.  Cardinality is input × list; cost mirrors
      // per-row evaluation of the list expression.  Always Alive (Unwind
      // always introduces sym), so ResolveChildren dispatches like alive Bind.
      case symbol::Unwind: {
        auto const sym_eclass = current.children()[1];
        auto const &[_, sym_cost] = children[1]->resolve();
        return UnwindFlatMap(*children[0], *children[2], sym_eclass, sym_cost, enode_id);
      }

      // Subquery (CALL block): scope-barrier row-pipe.  Children layout is
      // [outer_input, inner_root, exposed_sym_1, ...].  Inner's introduces
      // are STRIPPED at the barrier; only `exposed_syms` cross into the
      // outer scope.  Importing-CALL is unsupported - guard up front so a
      // failing query surfaces the real cause instead of a downstream
      // "no self-contained alternative".
      case symbol::Subquery: {
        bool const has_self_contained_inner =
            std::ranges::any_of(children[1]->alts(), [](Alternative const &a) { return a.required.empty(); });
        if (!has_self_contained_inner) {
          throw NotYetImplemented{"importing CALL subqueries"};
        }
        return SubqueryFlatMap(
            *children[0], *children[1], ExposedSymsFromChildren(current.children().subspan(2)), enode_id);
      }

      // Function call: per-class cost-sum chain over args, then override
      // cardinality with the estimator's output.  Function cardinality is
      // *not* the product of arg cardinalities (args are scalars).
      case symbol::Function:
        return FunctionCombine(children, estimator.Estimate(current, current.children(), egraph), enode_id);
    }
    std::unreachable();
  }
};

/// Identifies one resolved "node" in the extracted plan.  Two parent paths
/// that visit the same e-class with different sets of variables in scope
/// get different ResolvedKeys, so each picks its own optimal alternative.
/// Without this, a parent path with a richer scope could be forced to
/// reuse an earlier path's pick that doesn't lean on the extra bindings.
struct ResolvedKey {
  planner::core::EClassId eclass;
  SymbolSet provided;
  /// Symbols this subtree's chosen alt must introduce.  Set non-empty by
  /// Output (when its NamedOutputs reference symbols the input row pipe
  /// must bind) and propagated down through Bind/Unwind.  Empty means "no
  /// additional demand from above" - the picker chooses on cost alone.
  SymbolSet demanded_introduces;

  bool operator==(ResolvedKey const &) const = default;
};

struct ResolvedKeyHash {
  std::size_t operator()(ResolvedKey const &k) const noexcept {
    auto h = boost::hash<planner::core::EClassId>{}(k.eclass);
    auto hash_set = [&](SymbolSet const &s) {
      for (auto const &id : s) boost::hash_combine(h, boost::hash<planner::core::EClassId>{}(id));
    };
    hash_set(k.provided);
    hash_set(k.demanded_introduces);
    return h;
  }
};

/// One entry in the resolver's topological output: this key resolved to
/// `enode_id`.  `is_alive` is meaningful only for Bind enodes; the builder
/// reads it to decide whether sym/expr children participate.
///
/// `child_begin`/`child_end` index into the shared `child_indices` CSR
/// buffer on `QueryPlannerContext::impl()`.  Each slot holds the
/// `build_order` index of one child the resolver visited - in resolver-visit
/// order, which matches enode-children order for the dispatch arms in
/// `ResolveChildren` (alive Bind/Unwind: [input, sym, expr];
/// dead Bind/Unwind: [input]; Subquery: [outer, inner, syms...];
/// Output: [pipe, named_outs...]; generic: enode children).
struct TopoEntry {
  planner::core::ENodeId enode_id;
  AliveTag is_alive = AliveTag::NotApplicable;
  std::uint32_t child_begin = 0;
  std::uint32_t child_end = 0;
};

// ============================================================================
// Scope-threading operator resolution
// ============================================================================
//
// Symbols with scope-threading semantics (Bind, Unwind, Subquery, Output) are
// handled by dedicated functions below.  Each function documents its child
// roles and scope propagation semantics.
//
// Adding a new scope-threading operator requires:
//   1. Add the operator to `kScopeThreadingOperators` in the generic arm below
//      (compile-time enforcement that it was not forgotten here)
//   2. Add a new ResolveXxxChild() function following the pattern of existing ones
//   3. Call the new function from ResolveChildren()
//
// Generic expression operators (Leaf, Unary, Binary) fall through to the generic
// arm - they carry no scope context and their children all get provided=unchanged,
// demanded={}.

void ResolveBindUnwindAlive(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, auto visit) {
  auto const &children = enode.children();
  auto const sym_eclass = children[1];
  auto alive_provided = parent_key.provided;
  alive_provided.insert(sym_eclass);
  auto downstream_demand = parent_key.demanded_introduces.difference_one(sym_eclass);
  visit(ResolvedKey{children[0], std::move(alive_provided), std::move(downstream_demand)});
  visit(ResolvedKey{sym_eclass, parent_key.provided, {}});
  visit(ResolvedKey{children[2], parent_key.provided, {}});
}

void ResolveBindDead(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, auto visit) {
  visit(ResolvedKey{enode.children()[0], parent_key.provided, parent_key.demanded_introduces});
}

void ResolveSubqueryChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key,
                             SymbolSet const &exposed_syms, auto visit) {
  auto const &children = enode.children();
  auto outer_demand = parent_key.demanded_introduces.difference(exposed_syms);
  visit(ResolvedKey{children[0], parent_key.provided, std::move(outer_demand)});
  visit(ResolvedKey{children[1], SymbolSet{}, SymbolSet{}});
  for (auto sym_child : children.subspan(2)) {
    visit(ResolvedKey{sym_child, parent_key.provided, {}});
  }
}

void ResolveOutputChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key,
                           SymbolSet const &chosen_introduces, auto visit) {
  auto const &children = enode.children();
  visit(ResolvedKey{children[0], parent_key.provided, chosen_introduces});
  auto enriched_provided = parent_key.provided;
  for (auto sym : chosen_introduces) enriched_provided.insert(sym);
  for (auto named_out : children.subspan(1)) {
    visit(ResolvedKey{named_out, enriched_provided, {}});
  }
}

void ResolveGenericChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, auto visit) {
  for (auto child : enode.children()) {
    visit(ResolvedKey{child, parent_key.provided, {}});
  }
}

/// Dispatch to the appropriate child-resolution function based on enode shape.
///
/// Called only by `PlanResolver`.  The builder reads forward child indices the
/// resolver recorded into `child_indices`, so the child-key derivation rule
/// lives in exactly one place.
///
/// `exposed_syms` must be pre-computed by the caller for Subquery enodes
/// (pass nullptr for all other enode types).
void ResolveChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, AliveTag is_alive,
                     SymbolSet const &chosen_introduces, SymbolSet const *exposed_syms, auto visit) {
  auto const sym_op = enode.symbol();
  auto const &children = enode.children();
  bool const is_bind_or_unwind = (sym_op == symbol::Bind || sym_op == symbol::Unwind) && children.size() == 3;

  if (is_bind_or_unwind && is_alive == AliveTag::Alive) {
    ResolveBindUnwindAlive(enode, parent_key, visit);
  } else if (is_bind_or_unwind) {
    ResolveBindDead(enode, parent_key, visit);
  } else if (sym_op == symbol::Subquery && children.size() >= 2) {
    assert(exposed_syms && "caller must precompute exposed_syms for Subquery enodes");
    ResolveSubqueryChildren(enode, parent_key, *exposed_syms, visit);
  } else if (sym_op == symbol::Output && !children.empty()) {
    ResolveOutputChildren(enode, parent_key, chosen_introduces, visit);
  } else {
    if (IsScopeThreadingOp(sym_op)) {
      throw QueryException{
          "Planner internal error: unhandled scope-threading symbol in ResolveChildren - "
          "please report this bug at https://github.com/memgraph/memgraph/issues"};
    }
    ResolveGenericChildren(enode, parent_key, visit);
  }
}

[[noreturn]] void ThrowPlannerBug(std::string_view detail) {
  throw QueryException{std::string{"Plan extraction failed: "} + std::string{detail} +
                       " This is a planner bug - please report it at "
                       "https://github.com/memgraph/memgraph/issues"};
}

/// Context-aware resolver with PER-PATH caching.
///
/// Each (eclass, provided) pair is resolved exactly once.  Different parent
/// paths that visit the same eclass under different scopes get distinct
/// selections - each path picks the cheapest alternative feasible under
/// its own provided set, so no path is forced to settle for a sub-optimal
/// alt that another path's smaller scope already chose.
///
/// Output is a topological order (children-before-parents) of TopoEntries
/// plus a packed CSR `child_indices` buffer where each entry's slice gives
/// the `build_order` indices of the children the resolver visited (in the
/// order `ResolveChildren` emits them).  The builder reads the CSR directly
/// and never re-derives child keys, so the child-resolution rule lives in
/// exactly one place.
struct PlanResolver {
  using EClassId = planner::core::EClassId;
  using FrontierMap = planner::core::extract::FrontierMap<CostFrontier>;
  using EGraph = planner::core::EGraph<symbol, analysis>;
  using TopoOrder = std::vector<TopoEntry>;

  void operator()(EGraph const &egraph, FrontierMap const &frontier_map, EClassId root, TopoOrder &out_order,
                  std::vector<std::uint32_t> &child_indices,
                  boost::unordered_flat_map<ResolvedKey, std::uint32_t, ResolvedKeyHash> &seen) const {
    if (!out_order.empty()) ThrowPlannerBug("resolver output must be empty on entry.");
    if (!child_indices.empty()) ThrowPlannerBug("resolver child_indices must be empty on entry.");
    planner::core::extract::DfsPostOrder(
        ResolvedKey{root, SymbolSet{}, SymbolSet{}},
        seen,
        out_order,
        [&](ResolvedKey const &key, auto visit_child) -> TopoEntry {
          auto const fr_it = frontier_map.find(key.eclass);
          if (fr_it == frontier_map.end() || !fr_it->second.has_value())
            ThrowPlannerBug("eclass has no frontier during resolution.");
          auto valid_alt = [&provided = key.provided, &demanded = key.demanded_introduces](Alternative const &alt) {
            return alt.required.is_compatible(provided) && std::ranges::includes(alt.introduces, demanded);
          };
          auto const *best = planner::core::extract::PickBest(fr_it->second->alts(), valid_alt);
          if (!best) ThrowPlannerBug("no compatible alternative at this node.");
          auto const &chosen = *best;
          auto const &enode = egraph.get_enode(chosen.enode_id);
          auto const &enode_children = enode.children();
          auto const exposed = (enode.symbol() == symbol::Subquery && enode_children.size() >= 2)
                                   ? std::make_optional(ExposedSymsFromChildren(enode_children.subspan(2)))
                                   : std::nullopt;
          // Per-frame scratch: collect this entry's child indices contiguously
          // here, then bulk-append to the shared CSR at emit time.  Direct
          // append-on-visit would interleave with grandchildren's appends.
          boost::container::small_vector<std::uint32_t, 4> scratch;
          ResolveChildren(enode,
                          key,
                          chosen.is_alive,
                          chosen.introduces,
                          exposed ? &*exposed : nullptr,
                          [&](ResolvedKey child_key) { scratch.push_back(visit_child(std::move(child_key))); });
          auto const begin = static_cast<std::uint32_t>(child_indices.size());
          child_indices.insert(child_indices.end(), scratch.begin(), scratch.end());
          auto const end = static_cast<std::uint32_t>(child_indices.size());
          return TopoEntry{
              .enode_id = chosen.enode_id, .is_alive = chosen.is_alive, .child_begin = begin, .child_end = end};
        });
  }
};

}  // namespace

// We can build operators or expressions
// Operators -> used once in the build
// Expression -> can be reused
using LogicalOperatorPtr = std::shared_ptr<LogicalOperator>;
using BuildResult = std::variant<LogicalOperatorPtr, Expression *, Symbol, NamedExpression *>;
using enode_ref = planner::core::ENode<symbol> const &;
using child_ref = std::reference_wrapper<BuildResult const>;
using children_ref = std::span<child_ref const>;

namespace {
template <typename T, std::size_t idx>
[[nodiscard]] auto ExtractAndValidate(children_ref children) -> const T & {
  if (children.size() <= idx) throw QueryException{"Planner error, missing child node"};
  const auto *ptr = std::get_if<T>(&children[idx].get());
  if (!ptr) throw QueryException{"Planner error, child node is incorrect type"};
  return *ptr;
}

template <typename T>
[[nodiscard]] auto Validate(child_ref child) -> const T & {
  const auto *ptr = std::get_if<T>(&child.get());
  if (!ptr) throw QueryException{"Planner error, child node is incorrect type"};
  return *ptr;
}
}  // namespace

struct Builder {
  auto Build(enode_ref node, children_ref children) -> BuildResult {
    // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define X(SYM)      \
  case symbol::SYM: \
    return Build(utils::tag_v<symbol::SYM>, node, children)

    switch (node.symbol()) {
      X(Once);
      X(Bind);
      X(Symbol);
      X(Literal);
      X(Identifier);
      X(Output);
      X(NamedOutput);
      X(ParamLookup);
      X(Function);
      X(Unwind);
      X(Subquery);
#define MG_DISPATCH_OP(Name, ...) X(Name);
      EGRAPH_BINARY_OPS(MG_DISPATCH_OP)
      EGRAPH_UNARY_OPS(MG_DISPATCH_OP)
#undef MG_DISPATCH_OP
    }
#undef X
  }

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto Build(utils::tag_value<symbol::Once> /*tag*/, enode_ref /*node*/, children_ref /*children*/) -> BuildResult {
    return std::make_unique<Once>();
  }

  auto Build(utils::tag_value<symbol::Bind> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult {
    auto const &input = ExtractAndValidate<LogicalOperatorPtr, 0>(children);
    auto const &sym = ExtractAndValidate<Symbol, 1>(children);
    auto const &expression = ExtractAndValidate<Expression *, 2>(children);

    // TODO/NOTE: lost token_position_, and is_aliased_
    auto *named_expression = ast_storage_.Create<NamedExpression>(sym.name(), expression);
    named_expression->MapTo(sym);

    if (input->GetTypeInfo() == Produce::kType) {
      auto const &produce = static_pointer_cast<Produce>(input);
      // TODO: check if its ok to steal from the other produce (Operators make a
      // tree, we are skipping hence unused)
      auto named_expressions = produce->named_expressions_;
      named_expressions.emplace_back(named_expression);
      return std::make_shared<Produce>(produce->input(), named_expressions);
    }
    return std::make_shared<Produce>(input, std::vector{named_expression});
  }

  auto Build(utils::tag_value<symbol::Symbol> /*tag*/, enode_ref node, children_ref /*children*/) -> BuildResult {
    auto const sym_pos = static_cast<int32_t>(node.disambiguator());
    auto const it = reverse_symbol_name_store_.find(sym_pos);
    if (it == reverse_symbol_name_store_.end()) [[unlikely]] {
      throw QueryException{"Planner error, symbol not found in store"};
    }
    // TODO/NOTE: lost user_declared_, type_, and token_position_
    return symbol_table_.CreateSymbol(it->second, false /*TODO*/);
  }

  auto Build(utils::tag_value<symbol::Literal> /*tag*/, enode_ref node, children_ref /*children*/) -> BuildResult {
    auto const dis = node.disambiguator();
    auto const it = reverse_literal_store_.find(dis);
    if (it == reverse_literal_store_.end()) [[unlikely]] {
      throw QueryException{"Planner error, literal not found in store"};
    }
    return ast_storage_.Create<PrimitiveLiteral>(it->second);
  }

  auto Build(utils::tag_value<symbol::Identifier> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult {
    auto const &sym = ExtractAndValidate<Symbol, 0>(children);
    auto *identifier = ast_storage_.Create<Identifier>(sym.name(), sym.user_declared());
    identifier->MapTo(sym);
    return identifier;
  }

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  auto Build(utils::tag_value<symbol::Output> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult {
    auto const &input = ExtractAndValidate<LogicalOperatorPtr, 0>(children);
    auto named_expressions =
        children | std::views::drop(1) | std::views::transform(Validate<NamedExpression *>) | ranges::to<std::vector>;
    return std::make_shared<Produce>(input, std::move(named_expressions));
  }

  auto Build(utils::tag_value<symbol::NamedOutput> /*tag*/, enode_ref node, children_ref children) -> BuildResult {
    auto const &sym = ExtractAndValidate<Symbol, 0>(children);
    auto const &expression = ExtractAndValidate<Expression *, 1>(children);

    auto const name_it = reverse_name_store_.find(node.disambiguator());
    DMG_ASSERT(name_it != reverse_name_store_.end());

    // TODO/NOTE: lost token_position_, and is_aliased_
    auto *named_expression = ast_storage_.Create<NamedExpression>(name_it->second, expression);
    named_expression->MapTo(sym);
    return named_expression;
  }

  auto Build(utils::tag_value<symbol::ParamLookup> /*tag*/, enode_ref node, children_ref /*children*/) -> BuildResult {
    auto const dis = node.disambiguator();
    return ast_storage_.Create<ParameterLookup>(dis);
  }

  auto Build(utils::tag_value<symbol::Unwind> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult {
    auto const &input = ExtractAndValidate<LogicalOperatorPtr, 0>(children);
    auto const &sym = ExtractAndValidate<Symbol, 1>(children);
    auto const &list_expr = ExtractAndValidate<Expression *, 2>(children);
    return std::static_pointer_cast<LogicalOperator>(std::make_shared<query::plan::Unwind>(input, list_expr, sym));
  }

  auto Build(utils::tag_value<symbol::Subquery> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult {
    auto const &outer_input = ExtractAndValidate<LogicalOperatorPtr, 0>(children);
    auto const &inner_root = ExtractAndValidate<LogicalOperatorPtr, 1>(children);
    // exposed_sym children at children[2..] are Symbol values that were
    // resolved into the build cache; they're structural metadata for the
    // Subquery e-node (so the cost / resolver can see what crosses the
    // barrier) but the v1 Apply operator doesn't consume them directly.
    return std::static_pointer_cast<LogicalOperator>(
        std::make_shared<query::plan::Apply>(outer_input, inner_root, /*subquery_has_return=*/true));
  }

  auto Build(utils::tag_value<symbol::Function> /*tag*/, enode_ref node, children_ref children) -> BuildResult {
    auto const dis = node.disambiguator();
    if (dis >= function_info_.size()) [[unlikely]] {
      throw QueryException{"Planner error, function id not found in store"};
    }
    auto const &name = function_info_[dis].name;
    auto args = std::vector<Expression *>{};
    args.reserve(children.size());
    for (auto child : children) {
      args.push_back(Validate<Expression *>(child));
    }
    return static_cast<Expression *>(ast_storage_.Create<Function>(name, args));
  }

  // Binary operator helpers
  template <typename AstOp>
  auto BuildBinaryOp(children_ref children) -> BuildResult {
    auto const &lhs = ExtractAndValidate<Expression *, 0>(children);
    auto const &rhs = ExtractAndValidate<Expression *, 1>(children);
    return ast_storage_.Create<AstOp>(lhs, rhs);
  }

  // Unary operator helpers
  template <typename AstOp>
  auto BuildUnaryOp(children_ref children) -> BuildResult {
    auto const &operand = ExtractAndValidate<Expression *, 0>(children);
    return ast_storage_.Create<AstOp>(operand);
  }

  // Binary / unary Build overloads - generated from the X-lists.
  // NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define MG_BUILD_BINARY(Name, AstOp)                                                                             \
  auto Build(utils::tag_value<symbol::Name> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult { \
    return BuildBinaryOp<AstOp>(children);                                                                       \
  }
  EGRAPH_BINARY_OPS(MG_BUILD_BINARY)
#undef MG_BUILD_BINARY

#define MG_BUILD_UNARY(Name, AstOp)                                                                              \
  auto Build(utils::tag_value<symbol::Name> /*tag*/, enode_ref /*node*/, children_ref children) -> BuildResult { \
    return BuildUnaryOp<AstOp>(children);                                                                        \
  }
  EGRAPH_UNARY_OPS(MG_BUILD_UNARY)
#undef MG_BUILD_UNARY

  // NOLINTEND(cppcoreguidelines-macro-usage)

  Builder(std::map<storage::ExternalPropertyValue, uint64_t> const &literal_store,
          std::map<std::string, uint64_t> const &name_store, std::map<int32_t, std::string> const &symbol_name_store,
          std::vector<FunctionInfo> const &function_info)
      : function_info_(function_info) {
    reverse_literal_store_.reserve(literal_store.size());
    for (auto const &[val, id] : literal_store) {
      reverse_literal_store_.emplace(id, val);
    }

    reverse_name_store_.reserve(name_store.size());
    for (auto const &[val, id] : name_store) {
      reverse_name_store_.emplace(id, val);
    }

    reverse_symbol_name_store_.reserve(symbol_name_store.size());
    for (auto const &[pos, name] : symbol_name_store) {
      reverse_symbol_name_store_.emplace(pos, name);
    }
  }

  boost::unordered_flat_map<uint64_t, storage::ExternalPropertyValue> reverse_literal_store_;
  boost::unordered_flat_map<uint64_t, std::string> reverse_name_store_;
  boost::unordered_flat_map<int32_t, std::string> reverse_symbol_name_store_;
  std::vector<FunctionInfo> function_info_;

  AstStorage ast_storage_;
  SymbolTable symbol_table_;
};

/// Convert an egraph (after rewrite saturation) into a concrete LogicalOperator tree.
///
/// Precondition: after all rewrites, every Identifier eclass must be merged with at least one
/// alternative whose `required == {}`. This is guaranteed when InlineRule has been applied to
/// saturation for all Bind/Identifier pairs in the egraph. Extensions that add new binding
/// forms (scan operators, UNWIND, etc.) must ensure a corresponding rewrite or direct enode
/// alternative satisfies this invariant.
///
/// If the invariant is violated, the function throws QueryException rather than invoking
/// undefined behaviour.
struct QueryPlannerContext::Impl {
  planner::core::extract::FrontierContext<CostFrontier> frontier_context;
  std::vector<TopoEntry> build_order;
  /// CSR child-index buffer: each `TopoEntry`'s [child_begin, child_end)
  /// slice references children by their `build_order` index.  Filled by the
  /// resolver in entry-emit order, so each entry's children are contiguous.
  std::vector<std::uint32_t> child_indices;
  boost::unordered_flat_map<ResolvedKey, std::uint32_t, ResolvedKeyHash> resolver_seen;
  /// User-provided estimator override.  null -> ConvertToLogicalOperator
  /// builds a BuiltinEstimator over the current egraph for this call.
  std::unique_ptr<CardinalityEstimator> estimator_override;
  /// Cardinality of the root alt picked by the most recent
  /// ConvertToLogicalOperator call.  NaN before the first call.
  double last_root_cardinality = std::numeric_limits<double>::quiet_NaN();

  void clear() {
    frontier_context.clear();
    build_order.clear();
    child_indices.clear();
    resolver_seen.clear();
  }
};

QueryPlannerContext::QueryPlannerContext() : impl_(std::make_unique<Impl>()) {}

QueryPlannerContext::QueryPlannerContext(std::unique_ptr<CardinalityEstimator> estimator)
    : impl_(std::make_unique<Impl>(Impl{.estimator_override = std::move(estimator)})) {}

QueryPlannerContext::~QueryPlannerContext() = default;
QueryPlannerContext::QueryPlannerContext(QueryPlannerContext &&) noexcept = default;
QueryPlannerContext &QueryPlannerContext::operator=(QueryPlannerContext &&) noexcept = default;

auto QueryPlannerContext::last_root_cardinality() const -> double { return impl_->last_root_cardinality; }

auto ConvertToLogicalOperator(egraph const &e, eclass root, QueryPlannerContext &planner_context) -> ExtractionResult {
  auto const &impl = internal::get_impl(e);

  /// STAGE: Multi-alt extraction from EGraph using PlanCostModel
  auto const true_root = internal::to_core_id(root);
  namespace extract = planner::core::extract;

  // Cleared on every call so capacity is preserved across queries but
  // contents start empty.
  auto &ctx = planner_context.impl();
  ctx.clear();

  // Root-satisfiability precondition: ComputeFrontiers must have produced at
  // least one self-contained alternative for the root (required == {}).
  // We compute frontiers eagerly here so we can validate before resolve.
  // Estimator: user-provided override takes precedence; otherwise build a
  // BuiltinEstimator over the current egraph for this call.  BuiltinEstimator
  // is per-call-stateful (binds to one egraph) so it can't be the long-lived
  // QueryPlannerContext default.
  auto const builtin = BuiltinEstimator{e};
  auto const *override_est = ctx.estimator_override.get();
  CardinalityEstimator const &active_estimator = override_est ? *override_est : builtin;

  // Pre-pass: collect the set of Symbol e-classes referenced by some
  // Identifier e-node anywhere in the e-graph.  This is the demand signal
  // Bind's cost case uses to decide whether to emit an alive alt (see
  // PlanCostModel::referenced_syms).  O(num_enodes) one-time scan.
  SymbolSet referenced_syms = [&] {
    boost::container::small_vector<planner::core::EClassId, 32> buf;
    for (auto eclass_id : impl.egraph_.canonical_eclass_ids()) {
      for (auto enode_id : impl.egraph_.eclass(eclass_id).nodes()) {
        auto const &enode = impl.egraph_.get_enode(enode_id);
        if (enode.symbol() == symbol::Identifier && !enode.children().empty()) {
          // children()[0] is canonical: EGraph::emplace canonicalizes children
          // at insert time and rebuild() re-canonicalizes via canonicalize_in_place().
          buf.push_back(enode.children()[0]);
        }
      }
    }
    return SymbolSet{std::move(buf)};
  }();

#ifndef NDEBUG
  // Invariant: Symbol e-classes are never merged with each other (see bind_semantics.hpp).
  // Each e-class containing a Symbol e-node must be a singleton - mixing two
  // Symbol e-nodes in one e-class would alias distinct variables and corrupt
  // demand tracking.
  for (auto eclass_id : impl.egraph_.canonical_eclass_ids()) {
    auto const &cls = impl.egraph_.eclass(eclass_id);
    bool const has_symbol = std::ranges::any_of(
        cls.nodes(), [&](auto enode_id) { return impl.egraph_.get_enode(enode_id).symbol() == symbol::Symbol; });
    DMG_ASSERT(!has_symbol || cls.nodes().size() == 1,
               "planner bug: Symbol e-class merged with another e-node - bind semantics invariant violated");
  }
#endif

  (void)extract::ComputeFrontiers(
      impl.egraph_, PlanCostModel{active_estimator, impl.egraph_, referenced_syms}, true_root, ctx.frontier_context);

  auto const root_it = ctx.frontier_context.frontier_map.find(true_root);
  if (root_it == ctx.frontier_context.frontier_map.end() || !root_it->second.has_value()) {
    throw QueryException{"Plan extraction failed: root eclass has no frontier"};
  }
  auto const &root_frontier = *root_it->second;
  bool root_satisfiable =
      std::ranges::any_of(root_frontier.alts(), [](Alternative const &a) { return a.required.empty(); });
  if (!root_satisfiable) {
    throw QueryException{
        "Plan extraction failed: root frontier has no self-contained alternative. "
        "All paths through the plan demand symbols that cannot be provided. "
        "Ensure all Identifier references are resolved by rewrites before extraction."};
  }

  // Resolve produces a children-before-parents topological order of
  // (eclass, provided) pairs - one entry per distinct path-context the
  // resolver visited, so each path can pick the alt that's optimal under
  // its own scope.
  PlanResolver{}(impl.egraph_,
                 ctx.frontier_context.frontier_map,
                 true_root,
                 ctx.build_order,
                 ctx.child_indices,
                 ctx.resolver_seen);

  /// STAGE: Build selected (LogicalOperator, Expression *, Symbol, NamedExpression *, etc.)
  auto builder = Builder{impl.storage<symbol::Literal>().store,
                         impl.storage<symbol::NamedOutput>().store,
                         impl.storage<symbol::Symbol>().store,
                         impl.storage<symbol::Function>().info};

  // Dense build cache indexed by `build_order` position.  The resolver's
  // `seen` map guarantees each (eclass, provided) is emitted exactly once,
  // so a flat vector suffices - no hashing, no rehash-invalidation hazards.
  // The vector is sized once and never resized, so refs into it are stable
  // for the duration of the loop.
  auto built = std::vector<BuildResult>(ctx.build_order.size());

  auto children_refs = std::vector<child_ref>{};
  for (std::uint32_t i = 0; i < ctx.build_order.size(); ++i) {
    auto const &entry = ctx.build_order[i];
    auto const &enode = impl.egraph_.get_enode(entry.enode_id);
    bool const is_bind = enode.symbol() == symbol::Bind && enode.children().size() == 3;

    // Dead Bind: forward the input child's BuildResult.  sym/expr were never
    // resolved; the resolver emitted exactly one child (the input).
    if (is_bind && entry.is_alive != AliveTag::Alive) {
      DMG_ASSERT(entry.child_end - entry.child_begin == 1, "dead Bind must have one resolver-emitted child");
      built[i] = std::move(built[ctx.child_indices[entry.child_begin]]);
      continue;
    }

    children_refs.clear();
    children_refs.reserve(entry.child_end - entry.child_begin);
    for (auto j = entry.child_begin; j < entry.child_end; ++j) {
      children_refs.push_back(std::cref(built[ctx.child_indices[j]]));
    }
    built[i] = builder.Build(enode, children_refs);
  }

  // STAGE: Get the built root as std::unique_ptr<LogicalOperator>.
  // Post-order emits the root last (it is the outermost recursion).
  DMG_ASSERT(!built.empty(), "build order must contain at least the root entry");
  auto *ptr = std::get_if<LogicalOperatorPtr>(&built.back());
  if (!ptr) throw QueryException{"Root should be LogicalOperator"};
  auto &result = *ptr;

  auto unique_result = result->Clone(&builder.ast_storage_);
  // Root alt: cheapest self-contained (the one the resolver would pick
  // under provided={}).  Existence is guaranteed by the root_satisfiable
  // precondition checked above.
  auto self_contained =
      root_frontier.alts() | std::views::filter([](Alternative const &a) { return a.required.empty(); });
  auto const &best = *std::ranges::min_element(self_contained, std::less<>{}, &Alternative::cost);
  ctx.last_root_cardinality = best.cardinality;
  return ExtractionResult{.plan = std::move(unique_result),
                          .cost = best.cost,
                          .ast_storage = std::move(builder.ast_storage_),
                          .symbol_table = std::move(builder.symbol_table_)};
}
}  // namespace memgraph::query::plan::v2
