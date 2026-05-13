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
//   2. Frontier ops    : Cartesian product (CombineAlts) and in-place
//                        re-stamping (MapAlts) used by the cost model.
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

// --- Frontier ops -----------------------------------------------------------

/// Cartesian product of two frontiers with cost summation and required-set
/// union.  Each (l, r) pair becomes one alternative in the result, re-stamped
/// with `enode_id` and `extra_cost`.
///
/// Cardinality is set to the scalar default (1.0): every current caller is a
/// per-evaluation operator (binary expressions, NamedOutput) whose result is
/// one value per call.  Multiplying child cardinalities would be wrong here -
/// e.g. NamedOutput(sym, range(0,5)) packages ONE named pair per call even
/// though the value is a 6-element list, so its cardinality is 1, not 6.
/// Callers that need a non-scalar result (Function, Output) override
/// cardinality after combining or use a bespoke combine lambda.
auto CombineAlts(CostFrontier const &lhs, CostFrontier const &rhs, double extra_cost, planner::core::ENodeId enode_id)
    -> CostFrontier {
  return CostFrontier::cartesian_product(lhs, rhs, [&](Alternative const &l, Alternative const &r) {
    return Alternative{
        .cost = extra_cost + l.cost + r.cost, .required = l.required.set_union(r.required), .enode_id = enode_id};
  });
}

/// Map over a single frontier - adjust each alternative's cost by `extra_cost`
/// and re-stamp `enode_id`.  Single-frontier sibling of CombineAlts.
/// Pareto invariant is preserved: a uniform cost shift does not change relative
/// ordering, `required` is untouched, and dominance does not read enode_id or
/// is_alive.  This lets us mutate in place without re-pruning or copying the
/// per-alt SymbolSet.  Callers used only for non-Bind enodes, so is_alive is
/// reset to NotApplicable (it is meaningful only when the alt's enode is a Bind or Unwind).
auto MapAlts(CostFrontier input, double extra_cost, planner::core::ENodeId enode_id) -> CostFrontier {
  input.mutate_pruning_invariant_preserving([&](Alternative &alt) {
    alt.cost += extra_cost;
    alt.enode_id = enode_id;
    alt.is_alive = AliveTag::NotApplicable;
  });
  return input;
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
      // Leaf nodes: single alternative, no demand
      case symbol::Once:
      case symbol::Literal:
      case symbol::Symbol:  // Leaf invariant - see bind::kSymbolCost.
      case symbol::ParamLookup:
        return CostResult{{{.cost = bind::kSymbolCost, .required = {}, .enode_id = enode_id}}};

      // Identifier: demands its symbol child to be bound
      case symbol::Identifier: {
        assert(!children.empty() && "Identifier must have its symbol child frontier");
        auto sym_eclass = current.children()[0];
        auto const &[_, child_cost] = children[0]->resolve();
        return CostResult{
            {{.cost = expression_cost::kIdentifier + child_cost, .required = {sym_eclass}, .enode_id = enode_id}}};
      }

      // Bind: emits one alt per (input_alt, expr_alt) for alive input alts and
      // one alt per dead input alt.  The is_alive tag rides on each alt so the
      // resolver can dispatch alive/dead by reading the chosen alt rather than
      // recomputing a comparison against a separate cost-bound estimate.
      case symbol::Bind: {
        auto const &input_frontier = *children[0];
        auto const &sym_frontier = *children[1];
        auto const &expr_frontier = *children[2];
        auto sym_eclass = current.children()[1];
        auto const &[_, sym_cost] = sym_frontier.resolve();

        return CostFrontier::flat_map(input_frontier, [&](auto const &input_alt, auto emit) {
          // Emit alive when there's a chance someone will demand sym:
          //   - Input subtree directly demands it (classic case), or
          //   - Some Identifier(sym) lives elsewhere in the e-graph and
          //     might cross a sibling boundary at an enclosing Output.
          // If neither holds, sym has no consumer and the alive alt would
          // bloat the frontier unbounded - up to 2^N for an N-Bind chain.
          bool const input_demands_sym = input_alt.required.is_alive(sym_eclass);
          bool const should_emit_alive = input_demands_sym || referenced_syms.contains(sym_eclass);
          if (should_emit_alive) {
            for (auto const &expr_alt : expr_frontier.alts()) {
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
          // Emit dead only when input doesn't already demand sym.
          // When input_demands_sym is true, the dead alt has sym still in
          // `required`, so it is only compatible with ancestors that already
          // provide sym.  But sym is provided only when an ancestor Bind for
          // it is alive - which is exactly the alive-alt condition.  A dead
          // alt under that condition is therefore unreachable: no resolver
          // context can pick it that couldn't also pick alive.  Suppressing
          // it avoids bloating the frontier with a semantically useless alt.
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

      // Binary expression operators (arithmetic / comparison / boolean):
      // lhs × rhs cartesian product, with the per-class cost looked up via the
      // symbol's descriptor.  Adding a new binary operator is one descriptor
      // specialisation in private_symbol.hpp - no new case arms here.
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
      case symbol::Xor: {
        auto const cost = expression_cost::FromClass(CostClassOf(current.symbol()));
        return CombineAlts(*children[0], *children[1], cost, enode_id);
      }

      // Unary expression operators: pass through child, +kUnary, re-stamp
      // enode_id so the Builder dispatches *this* unary node via enode.symbol().
      // Same dispatch story as binary: cost via descriptor.
      //
      // Uses LazyMap (view-style): no materialisation; chains of unary
      // operators collapse to a single materialisation at the eventual
      // iteration point.  is_alive is left untouched - children of unary
      // expressions never come from Bind/Unwind (those are row-pipe shapes,
      // and unary operators only run inside per-row expression evaluation),
      // so is_alive is already NotApplicable on every alt of children[0].
      case symbol::Not:
      case symbol::UnaryMinus:
      case symbol::UnaryPlus: {
        auto const cost = expression_cost::FromClass(CostClassOf(current.symbol()));
        return CostFrontier::LazyMap(*children[0], cost, enode_id);
      }

      // Output: row-pipe.  Re-stamp child[0]'s frontier (no extra cost) so
      // all alternatives dispatch through this Output enode in the Builder,
      // then fold in each NamedOutput child with its per-eval cost scaled
      // by the input row pipe's cardinality.  Per-row scaling is what lets
      // the planner prefer a one-shot Bind over an inlined alternative
      // when the row pipe is wide (e.g. UNWIND range(0, 100)).
      case symbol::Output: {
        auto result = MapAlts(*children[0], 0.0, enode_id);
        for (auto const *named_out : children.subspan(1)) {
          result = CostFrontier::cartesian_product(
              result, *named_out, [enode_id](Alternative const &l, Alternative const &r) {
                // l: input row pipe.  r: per-evaluation NamedOutput (scalar, 1
                // pair per call).
                // cost = l.cost (whole input pipeline) + l.cardinality * r.cost
                //        (per-output-row evaluation of this NamedOutput).
                // cardinality = l.cardinality.  Output produces exactly the
                // input row count regardless of what value-shape each
                // NamedOutput packages per row.
                //
                // required = l.required ∪ (r.required \ l.introduces).  The
                // NamedOutput is evaluated INSIDE the input pipe's row scope,
                // so any sym the input pipe binds (alive Bind / Unwind) covers
                // matching demands in r without needing an ancestor to provide
                // them.  This is what lets `WITH x AS y UNWIND ... RETURN y`
                // satisfy y's demand from the Bind / Unwind below the Output.
                auto const remaining = r.required.difference(l.introduces);
                return Alternative{.cost = l.cost + l.cardinality * r.cost,
                                   .cardinality = l.cardinality,
                                   .required = l.required.set_union(remaining),
                                   .introduces = l.introduces,
                                   .enode_id = enode_id};
              });
        }
        return result;
      }

      // NamedOutput: sym × expr cartesian product, +1 per pair.  Structural
      // (not an expression operator), so kept at a fixed cost rather than
      // sourced from expression_cost.
      case symbol::NamedOutput:
        return CombineAlts(*children[0], *children[1], 1.0, enode_id);

      // Unwind: row-generative.  Output cardinality = input.cardinality *
      // list_expr.cardinality (e.g. range(0, 100) over a 1-row input emits
      // 101 rows).  Cost composition mirrors a per-row evaluation of the
      // list expression, with kUnwindPerRowOverhead structural overhead per
      // produced row.  Alive/dead variable-introduction matches Bind:
      // Unwind always introduces sym, so input always sees `provided + sym`
      // in the resolver - we tag every Unwind alt with is_alive = true so
      // ResolveChildren dispatches like alive Bind.
      case symbol::Unwind: {
        auto const &input_frontier = *children[0];
        auto const &sym_frontier = *children[1];
        auto const &list_frontier = *children[2];
        auto sym_eclass = current.children()[1];
        auto const &[_, sym_cost] = sym_frontier.resolve();

        return CostFrontier::flat_map(input_frontier, [&](auto const &input_alt, auto emit) {
          for (auto const &list_alt : list_frontier.alts()) {
            // sym is always introduced by Unwind, so remove it from input's
            // required and union list_expr's required (its needs become
            // ours).  Same algebra as alive_required.
            auto required = input_alt.required.alive_required(sym_eclass, list_alt.required);
            auto introduces = input_alt.introduces;
            introduces.insert(sym_eclass);
            auto const cost =
                input_alt.cost + (list_alt.cost + kUnwindPerRowOverhead) * input_alt.cardinality + sym_cost;
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

      // Subquery (CALL block): scope-barrier row-pipe.  Children layout is
      // [outer_input, inner_root, exposed_sym_1, ...]; cost is per-row
      // evaluation of the inner plan; cardinality is the product; introduces
      // is `outer.introduces ∪ exposed_syms` - the inner's own introduces
      // are STRIPPED at the boundary, which is the (B) "available downstream"
      // semantic in action.  required is just the outer pipe's; for
      // non-importing subqueries the inner's required must be empty (any
      // alt the inner produces with non-empty required is rejected here).
      case symbol::Subquery: {
        auto const &outer_frontier = *children[0];
        auto const &inner_frontier = *children[1];

        // Importing-CALL guard: every inner alt with non-empty `required`
        // is silently rejected below.  If no inner alt is self-contained,
        // the Subquery eclass would emit an empty frontier and the root
        // satisfiability check would later throw an opaque "no self-
        // contained alternative".  Surface the actual cause here.
        bool const has_self_contained_inner =
            std::ranges::any_of(inner_frontier.alts(), [](Alternative const &a) { return a.required.empty(); });
        if (!has_self_contained_inner) {
          throw NotYetImplemented{"importing CALL subqueries"};
        }

        auto const exposed_syms = ExposedSymsFromChildren(current.children().subspan(2));

        return CostFrontier::flat_map(outer_frontier, [&](auto const &outer_alt, auto emit) {
          for (auto const &inner_alt : inner_frontier.alts()) {
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

      // Function call: cartesian product over arg frontiers (cost-sum and
      // required-union via the standard CombineAlts chain), then override
      // cardinality with the estimator's output for this function id.
      // Cardinality of a function call is *not* the product of its arg
      // cardinalities (those are scalars by construction), so the
      // CombineAlts product is replaced uniformly per-alt.
      case symbol::Function: {
        CostResult result;
        if (children.empty()) {
          result = CostResult{{{.cost = 0.0, .required = {}, .enode_id = enode_id}}};
        } else {
          result = MapAlts(*children[0], 0.0, enode_id);
          for (auto const *arg : children.subspan(1)) {
            result = CombineAlts(result, *arg, 0.0, enode_id);
          }
        }
        auto const cardinality = estimator.Estimate(current, current.children(), egraph);
        // Uniform per-alt edit: structural +1 cost and cardinality from
        // estimator.  Same value across alts -> dominance ordering
        // preserved, mutate_pruning_invariant_preserving holds.
        result.mutate_pruning_invariant_preserving([&](Alternative &alt) {
          alt.cost += 1.0;
          alt.cardinality = cardinality;
          alt.enode_id = enode_id;
          alt.is_alive = AliveTag::NotApplicable;
        });
        return result;
      }
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
/// reads it to decide whether sym/expr children participate.  `introduces`
/// is carried so the builder can compute the same per-child keys the
/// resolver did when the chosen alt was picked.
struct TopoEntry {
  ResolvedKey key;
  planner::core::ENodeId enode_id;
  AliveTag is_alive = AliveTag::NotApplicable;
  SymbolSet introduces;
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

template <typename Visit>
void ResolveBindUnwindAlive(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, Visit visit) {
  auto const &children = enode.children();
  auto const sym_eclass = children[1];
  auto alive_provided = parent_key.provided;
  alive_provided.insert(sym_eclass);
  auto downstream_demand = parent_key.demanded_introduces.difference_one(sym_eclass);
  visit(ResolvedKey{children[0], std::move(alive_provided), std::move(downstream_demand)});
  visit(ResolvedKey{sym_eclass, parent_key.provided, {}});
  visit(ResolvedKey{children[2], parent_key.provided, {}});
}

template <typename Visit>
void ResolveBindDead(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, Visit visit) {
  visit(ResolvedKey{enode.children()[0], parent_key.provided, parent_key.demanded_introduces});
}

template <typename Visit>
void ResolveSubqueryChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key,
                             SymbolSet const &exposed_syms, Visit visit) {
  auto const &children = enode.children();
  auto outer_demand = parent_key.demanded_introduces.difference(exposed_syms);
  visit(ResolvedKey{children[0], parent_key.provided, std::move(outer_demand)});
  visit(ResolvedKey{children[1], SymbolSet{}, SymbolSet{}});
  for (auto sym_child : children.subspan(2)) {
    visit(ResolvedKey{sym_child, parent_key.provided, {}});
  }
}

template <typename Visit>
void ResolveOutputChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key,
                           SymbolSet const &chosen_introduces, Visit visit) {
  auto const &children = enode.children();
  visit(ResolvedKey{children[0], parent_key.provided, chosen_introduces});
  auto enriched_provided = parent_key.provided;
  for (auto sym : chosen_introduces) enriched_provided.insert(sym);
  for (auto named_out : children.subspan(1)) {
    visit(ResolvedKey{named_out, enriched_provided, {}});
  }
}

template <typename Visit>
void ResolveGenericChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, Visit visit) {
  for (auto child : enode.children()) {
    visit(ResolvedKey{child, parent_key.provided, {}});
  }
}

/// Dispatch to the appropriate child-resolution function based on enode shape.
///
/// `exposed_syms` must be pre-computed by the caller for Subquery enodes
/// (pass nullptr for all other enode types).  Both the resolver and the builder
/// call this function once per selected enode; a Subquery may be visited under
/// multiple provided-sets, so computing the set at the callsite and passing it
/// in ensures it is derived only once per visit rather than once per call.
template <typename Visit>
void ResolveChildren(planner::core::ENode<symbol> const &enode, ResolvedKey const &parent_key, AliveTag is_alive,
                     SymbolSet const &chosen_introduces, SymbolSet const *exposed_syms, Visit visit) {
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
/// Output is a topological order (children-before-parents) of TopoEntries.
/// The builder consumes this order, keying its build cache by ResolvedKey,
/// so subtrees are shared exactly when they appear under the same provided
/// context and instantiated independently when they don't.
struct PlanResolver {
  using EClassId = planner::core::EClassId;
  using FrontierMap = planner::core::extract::FrontierMap<CostFrontier>;
  using EGraph = planner::core::EGraph<symbol, analysis>;
  using TopoOrder = std::vector<TopoEntry>;

  void operator()(EGraph const &egraph, FrontierMap const &frontier_map, EClassId root, TopoOrder &out_order,
                  boost::unordered_flat_set<ResolvedKey, ResolvedKeyHash> &seen) const {
    if (!out_order.empty()) ThrowPlannerBug("resolver output must be empty on entry.");
    planner::core::extract::DfsPostOrder(
        ResolvedKey{root, SymbolSet{}, SymbolSet{}},
        seen,
        out_order,
        [&](ResolvedKey key, auto visit_child) -> TopoEntry {
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
          ResolveChildren(enode,
                          key,
                          chosen.is_alive,
                          chosen.introduces,
                          exposed ? &*exposed : nullptr,
                          [&](ResolvedKey child_key) { visit_child(std::move(child_key)); });
          return TopoEntry{.key = std::move(key),
                           .enode_id = chosen.enode_id,
                           .is_alive = chosen.is_alive,
                           .introduces = chosen.introduces};
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
  boost::unordered_flat_set<ResolvedKey, ResolvedKeyHash> resolver_seen;
  /// User-provided estimator override.  null -> ConvertToLogicalOperator
  /// builds a BuiltinEstimator over the current egraph for this call.
  std::unique_ptr<CardinalityEstimator> estimator_override;
  /// Cardinality of the root alt picked by the most recent
  /// ConvertToLogicalOperator call.  NaN before the first call.
  double last_root_cardinality = std::numeric_limits<double>::quiet_NaN();

  void clear() {
    frontier_context.clear();
    build_order.clear();
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
  PlanResolver{}(impl.egraph_, ctx.frontier_context.frontier_map, true_root, ctx.build_order, ctx.resolver_seen);

  /// STAGE: Build selected (LogicalOperator, Expression *, Symbol, NamedExpression *, etc.)
  auto builder = Builder{impl.storage<symbol::Literal>().store,
                         impl.storage<symbol::NamedOutput>().store,
                         impl.storage<symbol::Symbol>().store,
                         impl.storage<symbol::Function>().info};

  // ---------------------------------------------------------------------------
  // build_cache reference-stability contract - DO NOT REGRESS.
  // ---------------------------------------------------------------------------
  //
  // build_cache uses open-addressing (boost::unordered_flat_map).  On rehash,
  // ALL references / iterators / pointers into the table are invalidated -
  // there are no stable nodes to fall back to.  Two consequences for the
  // Builder loop:
  //
  //   (1) Never write `cache[k] = cache.at(other_k)` or any expression where
  //       the LHS subscript and the RHS read the same map.  The LHS [] may
  //       insert (rehashing), invalidating the RHS reference before the
  //       assignment runs.  Always sequence: read RHS into a local first,
  //       then assign.
  //
  //   (2) Never hold a reference into build_cache across an insertion
  //       (including via children_refs / reference_wrapper / span).  If the
  //       insert rehashes, the captured ref dangles and any subsequent read
  //       through it is undefined.  The builder.Build() call below takes a
  //       span of refs into build_cache; we materialise its return into a
  //       local BEFORE the LHS [] runs.
  //
  // Belt-and-braces: reserve(build_order.size()) up-front so the loop's []
  // inserts can never rehash.
  auto build_cache = boost::unordered_flat_map<ResolvedKey, BuildResult, ResolvedKeyHash>{};
  build_cache.reserve(ctx.build_order.size());

  auto const cache_lookup = [&](ResolvedKey const &child_key) {
    auto const it = build_cache.find(child_key);
    DMG_ASSERT(it != build_cache.end(), "Building bottom up we should be able to find our child");
    return std::cref(it->second);
  };
  // build_order is children-before-parents (post-order from the resolver);
  // walk it forward so every child is in build_cache before its parent.
  auto children_refs = std::vector<child_ref>{};
  for (auto const &entry : ctx.build_order) {
    auto const &enode = impl.egraph_.get_enode(entry.enode_id);
    auto const &children = enode.children();
    bool const is_bind = enode.symbol() == symbol::Bind && children.size() == 3;

    // Dead Bind: pass through input.  sym/expr were never resolved for
    // this key, so they're absent from build_cache.  The dead branch
    // forwards demanded_introduces unchanged (see ResolveChildren).
    //
    // The input child key is guaranteed to be in build_cache: the resolver
    // visited it (via ResolveChildren) before emitting this dead Bind entry,
    // and each (eclass, provided) pair is resolved exactly once (seen set).
    if (is_bind && entry.is_alive != AliveTag::Alive) {
      auto const input_key = ResolvedKey{children[0], entry.key.provided, entry.key.demanded_introduces};
      auto const it = build_cache.find(input_key);
      DMG_ASSERT(it != build_cache.end(), "Dead Bind input key must be in build_cache - resolver invariant violated");
      build_cache[entry.key] = std::move(it->second);
      continue;
    }

    children_refs.clear();
    children_refs.reserve(children.size());
    // Resolve children using the same rule the resolver used (see ResolveChildren).
    auto const exposed = (enode.symbol() == symbol::Subquery && children.size() >= 2)
                             ? std::make_optional(ExposedSymsFromChildren(children.subspan(2)))
                             : std::nullopt;
    ResolveChildren(
        enode, entry.key, entry.is_alive, entry.introduces, exposed ? &*exposed : nullptr, [&](ResolvedKey child_key) {
          children_refs.push_back(cache_lookup(child_key));
        });
    // See contract (2) above: materialise Build's result before the LHS [] runs.
    auto build_result = builder.Build(enode, children_refs);
    build_cache[entry.key] = std::move(build_result);
  }

  // STAGE: Get the built root as std::unique_ptr<LogicalOperator>.
  auto root_key = ResolvedKey{true_root, SymbolSet{}, SymbolSet{}};
  auto *ptr = std::get_if<LogicalOperatorPtr>(&build_cache[root_key]);
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
