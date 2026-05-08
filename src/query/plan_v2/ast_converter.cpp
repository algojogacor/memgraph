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

#include "query/plan_v2/ast_converter.hpp"

#include "query/exceptions.hpp"
#include "query/frontend/ast/ast.hpp"
#include "query/frontend/ast/ast_visitor.hpp"
#include "query/frontend/semantic/symbol_table.hpp"

using memgraph::query::plan::v2::egraph;

namespace memgraph::query {
namespace {

/// Surface an unsupported AST node as NotYetImplemented so the client gets
/// a clean error response and the server keeps running.  Picks up the AST
/// node's name from its TypeInfo so each unsupported case doesn't have to
/// hand-spell its own label.
[[noreturn]] void ThrowNotImplementedYet(Tree const &op) { throw NotYetImplemented(op.GetTypeInfo().name); }

[[noreturn]] void ThrowNotImplementedYet(std::string_view feature) { throw NotYetImplemented(feature); }

struct AstConverterVisitor : HierarchicalTreeVisitor {
  using HierarchicalTreeVisitor::PostVisit;
  using HierarchicalTreeVisitor::PreVisit;
  using HierarchicalTreeVisitor::ReturnType;
  using HierarchicalTreeVisitor::Visit;

  explicit AstConverterVisitor(SymbolTable const &symbol_table) : symbol_table_(symbol_table) {}

  ReturnType Visit(ParameterLookup &expr) override {
    // NOTE: with the stripper it mans that literals are also parameters
    //       hence we can't distinguish between literals vs parameters
    //       we will not do any optimisations around ParameterLookup at plan time (because we still need to cache)
    // TODO: is token_position_ right? rename
    builder_stack_.emplace_back(egraph_.MakeParameterLookup(expr.token_position_));
    return true;
  }

  ReturnType Visit(PrimitiveLiteral &expr) override {
    builder_stack_.emplace_back(egraph_.MakeLiteral(expr.value_));
    return true;
  }

  ReturnType Visit(Identifier &expr) override {
    auto const &sym = symbol_table_.at(expr);
    auto sym_eclass = egraph_.MakeSymbol(expr.symbol_pos_, sym.name());
    builder_stack_.emplace_back(egraph_.MakeIdentifier(sym_eclass));
    return true;
  }

  ReturnType Visit(EnumValueAccess &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(CypherQuery &op) override {
    if (op.memory_limit_ != nullptr) ThrowNotImplementedYet("query memory limit");
    (void)op.memory_scale_;  // TODO
    if (op.pre_query_directives_.commit_frequency_ != nullptr) ThrowNotImplementedYet("commit frequency directive");
    if (op.pre_query_directives_.hops_limit_ != nullptr) ThrowNotImplementedYet("hop limit directive");
    if (!op.pre_query_directives_.index_hints_.empty()) ThrowNotImplementedYet("index hints directive");
    return true;
  }

  bool PostVisit(CypherQuery & /*cypher_query*/) override { return true; }

  // bool PreVisit(SingleQuery &) override { return true; }
  // bool PostVisit(SingleQuery &) override { return true; }

  // bool PreVisit(With &tree) override {
  //   // visiting order of the With is done by the With::Accept
  //   return true;
  // }
  // bool PostVisit(With &) override { return true; }

  bool PreVisit(NamedExpression & /*expr*/) override {
    EnsureInput();
    return true;
  }

  bool PostVisit(NamedExpression &op) override {
    auto expr = PopStack();
    auto input = PopStack();
    DMG_ASSERT(op.symbol_pos_ != -1, "AST symbol should have already been mapped into the frame");
    auto const &sym = symbol_table_.at(op);
    auto sym_eclass = egraph_.MakeSymbol(op.symbol_pos_, sym.name());
    auto bind = egraph_.MakeBind(input, sym_eclass, expr);
    builder_stack_.emplace_back(bind);

    // Any analysis to add to the bind EClass?
    // -> Bind has a name `named_expression.name_`
    // for debug purpose
    // NamedOutput its part of the ENode info

    return true;
  }

  bool PreVisit(Return &op) override {
    // Note: SymbolGenerator has already semantically checked that the names are unique
    std::vector<plan::v2::eclass> named_outputs;
    for (auto &named_expr : op.body_.named_expressions) {
      if (!named_expr->expression_->Accept(*this)) {
        return false;
      }

      auto expr = PopStack();
      auto const &sym = symbol_table_.at(*named_expr);
      auto sym_eclass = egraph_.MakeSymbol(named_expr->symbol_pos_, sym.name());
      auto named_output = egraph_.MakeNamedOutput(named_expr->name_, sym_eclass, expr);
      named_outputs.emplace_back(named_output);
    }

    EnsureInput();
    auto input = PopStack();
    builder_stack_.emplace_back(egraph_.MakeOutputs(input, std::move(named_outputs)));

    auto const &body = op.body_;
    for (const auto &order_by : body.order_by) {
      if (!order_by.expression->Accept(*this)) {
        return false;
      }
    }
    if (body.skip && !body.skip->Accept(*this)) {
      return false;
    }
    if (body.limit && !body.limit->Accept(*this)) {
      return false;
    }
    return false;
  }

  bool PostVisit(Return & /*return_*/) override { return true; }

  bool PreVisit(Where &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(CallSubquery &op) override {
    // Minimum scope (issue 0004 follow-up): non-importing CALL { ... } RETURN ...
    if (op.has_variable_scope_) ThrowNotImplementedYet("importing CALL with explicit scope clause");
    if (op.all_variables_scoped_) ThrowNotImplementedYet("CALL with implicit star scope clause");

    // Save outer state and let the inner cypher_query_'s clauses build a
    // fresh row pipe.  EnsureInput on an empty stack pushes a fresh Once
    // for the inner, independent of any outer Bind chain.
    saved_outer_stacks_.push_back(std::move(builder_stack_));
    builder_stack_.clear();
    return true;
  }

  bool PreVisit(Exists &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Foreach &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(LoadCsv &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(RegexMatch &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Unwind &op) override {
    // UNWIND list AS x: bypass the named_expression_'s default visit (which
    // would PostVisit as a Bind, the wrong operator) and instead visit only
    // the list expression to push it onto the builder stack, then assemble
    // the Unwind e-node directly.
    DMG_ASSERT(op.named_expression_ != nullptr, "Unwind must have a named expression");
    if (!op.named_expression_->expression_->Accept(*this)) return false;
    auto list_expr = PopStack();
    EnsureInput();
    auto input = PopStack();
    DMG_ASSERT(op.named_expression_->symbol_pos_ != -1, "AST symbol should have already been mapped into the frame");
    auto const &sym = symbol_table_.at(*op.named_expression_);
    auto sym_eclass = egraph_.MakeSymbol(op.named_expression_->symbol_pos_, sym.name());
    builder_stack_.emplace_back(egraph_.MakeUnwind(input, sym_eclass, list_expr));
    return false;  // children already visited; don't descend through named_expression_
  }

  bool PreVisit(Merge &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(RemoveLabels &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(RemoveProperty &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(SetLabels &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(SetProperties &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(SetProperty &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Delete &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(EdgeAtom &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(NodeAtom &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Pattern &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Match &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Create &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(CallProcedure &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(ListComprehension &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(None &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Any &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Single &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(All &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Extract &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Coalesce &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Reduce &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(Function & /*function*/) override { return true; }

  bool PreVisit(Aggregation &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(LabelsTest &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(AllPropertiesLookup &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(PropertyLookup &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(MapProjectionLiteral &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(MapLiteral &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(ListLiteral &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(IsNullOperator &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(IfOperator &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(ListSlicingOperator &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(SubscriptOperator &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(InListOperator &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(RangeOperator &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(CypherUnion &op) override { ThrowNotImplementedYet(op); }

  bool PreVisit(PatternComprehension &op) override { ThrowNotImplementedYet(op); }

  bool PostVisit(CallSubquery &op) override {
    DMG_ASSERT(builder_stack_.size() == 1, "subquery body must produce exactly one root e-class");
    auto inner_root = builder_stack_.back();
    builder_stack_.pop_back();

    // Restore outer state and ensure there's an outer row pipe to chain off.
    DMG_ASSERT(!saved_outer_stacks_.empty(), "subquery PostVisit without matching PreVisit");
    builder_stack_ = std::move(saved_outer_stacks_.back());
    saved_outer_stacks_.pop_back();
    EnsureInput();
    auto outer_input = PopStack();

    // Exposed syms come from the inner cypher_query_'s last RETURN clause.
    // Hashconsing makes MakeSymbol(pos, name) return the SAME e-class the
    // inner visit already created, so the Subquery e-node references the
    // same Symbol leaves the inner Output's NamedOutputs do.
    DMG_ASSERT(op.cypher_query_ != nullptr && op.cypher_query_->single_query_ != nullptr,
               "CALL block missing inner cypher query");
    std::vector<plan::v2::eclass> exposed_syms;
    for (auto *clause : op.cypher_query_->single_query_->clauses_) {
      auto *ret = utils::Downcast<query::Return>(clause);
      if (ret == nullptr) continue;
      // Last RETURN wins (Cypher chains can have intermediate WITH but only
      // one final RETURN inside a CALL block in (b)'s scope).
      exposed_syms.clear();
      for (auto *ne : ret->body_.named_expressions) {
        auto const &sym = symbol_table_.at(*ne);
        exposed_syms.push_back(egraph_.MakeSymbol(ne->symbol_pos_, sym.name()));
      }
    }
    MG_ASSERT(!exposed_syms.empty(), "CALL block must end in RETURN; unit subqueries are TODO");

    builder_stack_.emplace_back(egraph_.MakeSubquery(outer_input, inner_root, std::move(exposed_syms)));
    return true;
  }

  bool PostVisit(Exists & /*exists*/) override { return true; }

  bool PostVisit(Foreach & /*foreach*/) override { return true; }

  bool PostVisit(LoadCsv & /*load_csv*/) override { return true; }

  bool PostVisit(RegexMatch & /*regex_match*/) override { return true; }

  bool PostVisit(Unwind & /*unwind*/) override { return true; }

  bool PostVisit(Merge & /*merge*/) override { return true; }

  bool PostVisit(RemoveLabels & /*remove_labels*/) override { return true; }

  bool PostVisit(RemoveProperty & /*remove_property*/) override { return true; }

  bool PostVisit(SetLabels & /*set_labels*/) override { return true; }

  bool PostVisit(SetProperties & /*set_properties*/) override { return true; }

  bool PostVisit(SetProperty & /*set_property*/) override { return true; }

  bool PostVisit(Where & /*where*/) override { return true; }

  bool PostVisit(Delete & /*delete_*/) override { return true; }

  bool PostVisit(EdgeAtom & /*edge_atom*/) override { return true; }

  bool PostVisit(NodeAtom & /*node_atom*/) override { return true; }

  bool PostVisit(Pattern & /*pattern*/) override { return true; }

  bool PostVisit(Match & /*match*/) override { return true; }

  bool PostVisit(Create & /*create*/) override { return true; }

  bool PostVisit(CallProcedure & /*call_procedure*/) override { return true; }

  bool PostVisit(ListComprehension & /*list_comprehension*/) override { return true; }

  bool PostVisit(None & /*none*/) override { return true; }

  bool PostVisit(Any & /*any*/) override { return true; }

  bool PostVisit(Single & /*single*/) override { return true; }

  bool PostVisit(All & /*all*/) override { return true; }

  bool PostVisit(Extract & /*extract*/) override { return true; }

  bool PostVisit(Coalesce & /*coalesce*/) override { return true; }

  bool PostVisit(Reduce & /*reduce*/) override { return true; }

  bool PostVisit(Function &function) override {
    auto const arity = function.arguments_.size();
    DMG_ASSERT(builder_stack_.size() >= arity, "Function arguments missing from builder stack");
    auto const first = builder_stack_.end() - static_cast<std::ptrdiff_t>(arity);
    auto args = std::vector<plan::v2::eclass>(first, builder_stack_.end());
    builder_stack_.erase(first, builder_stack_.end());
    builder_stack_.emplace_back(egraph_.MakeFunction(function.function_name_, std::move(args)));
    return true;
  }

  bool PostVisit(Aggregation & /*aggregation*/) override { return true; }

  bool PostVisit(LabelsTest & /*labels_test*/) override { return true; }

  bool PostVisit(AllPropertiesLookup & /*all_properties_lookup*/) override { return true; }

  bool PostVisit(PropertyLookup & /*property_lookup*/) override { return true; }

  bool PostVisit(MapProjectionLiteral & /*map_projection_literal*/) override { return true; }

  bool PostVisit(MapLiteral & /*map_literal*/) override { return true; }

  bool PostVisit(ListLiteral & /*list_literal*/) override { return true; }

  bool PostVisit(IsNullOperator & /*is_null_operator*/) override { return true; }

  bool PostVisit(IfOperator & /*if_operator*/) override { return true; }

  bool PostVisit(ListSlicingOperator & /*list_slicing_operator*/) override { return true; }

  bool PostVisit(SubscriptOperator & /*subscript_operator*/) override { return true; }

  bool PostVisit(InListOperator & /*in_list_operator*/) override { return true; }

  bool PostVisit(RangeOperator & /*range_operator*/) override { return true; }

  // Binary / unary AST-operator PostVisits - generated from the X-lists.
  // NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define MG_POST_VISIT_BINARY(Name, AstOp)                      \
  bool PostVisit(AstOp & /*op*/) override {                    \
    auto rhs = PopStack();                                     \
    auto lhs = PopStack();                                     \
    builder_stack_.emplace_back(egraph_.Make##Name(lhs, rhs)); \
    return true;                                               \
  }
  EGRAPH_BINARY_OPS(MG_POST_VISIT_BINARY)
#undef MG_POST_VISIT_BINARY

#define MG_POST_VISIT_UNARY(Name, AstOp)                      \
  bool PostVisit(AstOp & /*op*/) override {                   \
    auto operand = PopStack();                                \
    builder_stack_.emplace_back(egraph_.Make##Name(operand)); \
    return true;                                              \
  }
  EGRAPH_UNARY_OPS(MG_POST_VISIT_UNARY)
#undef MG_POST_VISIT_UNARY

  // NOLINTEND(cppcoreguidelines-macro-usage)

  bool PostVisit(CypherUnion & /*cypher_union*/) override { return true; }

  bool PostVisit(PatternComprehension & /*pattern_comprehension*/) override { return true; }

  auto GetEgraph() && -> std::tuple<egraph, plan::v2::eclass> {
    MG_ASSERT(builder_stack_.size() == 1, "should have a root");
    return {std::move(egraph_), builder_stack_.back()};
  }

 private:
  auto PopStack() -> plan::v2::eclass {
    DMG_ASSERT(!builder_stack_.empty());
    const auto res = builder_stack_.back();
    builder_stack_.pop_back();
    return res;
  }

  void EnsureInput() {
    if (builder_stack_.empty()) {
      builder_stack_.emplace_back(egraph_.MakeOnce());
    }
  }

  SymbolTable const &symbol_table_;
  egraph egraph_;
  std::vector<plan::v2::eclass> builder_stack_;
  // Stack of outer builder_stack_ snapshots saved across CALL { ... } scope
  // boundaries.  Push at PreVisit(CallSubquery), pop at PostVisit.  Vector
  // (not std::stack) so nested subqueries Just Work.
  std::vector<std::vector<plan::v2::eclass>> saved_outer_stacks_;
};
}  // namespace
}  // namespace memgraph::query

namespace memgraph::query::plan::v2 {

auto ConvertToEgraph(CypherQuery const &query, SymbolTable const &symbol_table) -> std::tuple<egraph, eclass> {
  auto visitor = AstConverterVisitor{symbol_table};
  // TODO: fix HierarchicalTreeVisitor to allow const
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const_cast<CypherQuery &>(query).Accept(visitor);
  return std::move(visitor).GetEgraph();
}

}  // namespace memgraph::query::plan::v2
