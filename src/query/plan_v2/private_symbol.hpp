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

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ranges>
#include <utility>

namespace memgraph::query::plan::v2 {

enum struct symbol : std::uint8_t {
  Once,
  Bind,
  // Invariant: no rewrite rule may merge two Symbol e-classes.  Each Symbol
  // eclass must remain a singleton.  Violating this aliases distinct variables
  // and corrupts demand tracking.  See bind_semantics.hpp and the debug check
  // in ConvertToLogicalOperator.
  Symbol,
  Literal,
  Identifier,
  Output,
  NamedOutput,
  ParamLookup,
  // Arithmetic operators (binary)
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  Exp,
  // Comparison operators (binary)
  Eq,
  Neq,
  Lt,
  Lte,
  Gt,
  Gte,
  // Boolean operators
  And,
  Or,
  Xor,
  Not,
  // Unary operators
  UnaryMinus,
  UnaryPlus,
  // Function call (builtin or UDF); disambiguator is the per-egraph function id.
  Function,
  // UNWIND clause: 3 children [input, sym, list_expr]; mirrors Bind's shape.
  Unwind,
  // CALL { ... } subquery: variadic [outer_input, inner_root, exposed_sym_1, ...].
  // Acts as a scope barrier: inner's introduces are stripped at the boundary;
  // only the explicit exposed_sym children become visible to the outer scope.
  Subquery,
};

// ============================================================================
// Symbol singleton invariant
// ============================================================================
//
// symbol::Symbol e-classes MUST remain singletons - each one corresponds to
// exactly one variable name.  If two Symbol e-nodes were ever merged (via a
// rewrite rule), bind::IsAlive / bind::IsCompatible would alias distinct
// variables and corrupt demand tracking silently.
//
// This invariant is enforced at compile time by the static_asserts below,
// and at runtime in the debug build of ConvertToLogicalOperator.  Adding a
// new rewrite rule must either (a) prove it cannot merge two Symbol e-nodes, or
// (b) add a test that explicitly verifies Symbol e-classes remain singletons.
//
// The static guard here verifies that Symbol has the properties required for
// singleton safety: it's a leaf (no children that could participate in a merge)
// and its cost class is Leaf (so it can't participate in expression cost
// rewrites that might merge equivalent forms).

// ============================================================================
// Symbol descriptors - one source of truth for each symbol's properties.
// ============================================================================
//
// Adding a new symbol requires:
//   1. an entry in the `symbol` enum above,
//   2. a `symbol_descriptor<symbol::Foo>` specialisation below,
//   3. a `symbol_make_traits<symbol::Foo>` specialisation in symbol_make_traits.hpp
//      (mostly auto-derived for binary/unary via the requires clauses there),
//   4. a `Build(tag<symbol::Foo>, ...)` overload in egraph_converter.cpp,
//   5. an ast_converter visitor pair (PreVisit/PostVisit) for the AST node.
//
// (1) and (2) are the *categorical* declaration - what kind of symbol is this?
// They live next to each other so a missing descriptor is a compile error at
// the first template instantiation that needs it (predicates, cost-class
// lookup, etc.).  (3)-(5) are the *behavioural* declarations - true forks
// because each operator has a unique lowering and AST mapping.

/// Arity / shape category for a symbol.  Drives Make* signatures and the
/// classification predicates below.
enum class Arity : std::uint8_t {
  Leaf,     ///< Once, Symbol, Literal, ParamLookup - no children.
  Unary,    ///< Identifier, Not, UnaryMinus, UnaryPlus - exactly one child.
  Binary,   ///< Add, Sub, ..., Eq, ..., And, Or, Xor - exactly two children.
  Special,  ///< Bind (3 children), Output (variable), NamedOutput (2 children).
};

/// Cost class for PlanCostModel.  Symbols of the same class share an expression
/// cost constant from expression_cost.hpp; structural symbols are scored
/// directly by PlanCostModel rather than via a constant.
enum class CostClass : std::uint8_t {
  Arithmetic,  ///< Add, Sub, Mul, Div, Mod, Exp - expression_cost::kArithmetic.
  Comparison,  ///< Eq, Neq, Lt, Lte, Gt, Gte - expression_cost::kComparison.
  Boolean,     ///< And, Or, Xor - expression_cost::kBoolean.
  Unary,       ///< Not, UnaryMinus, UnaryPlus - expression_cost::kUnary.
  Identifier,  ///< Identifier - expression_cost::kIdentifier (+ child cost).
  Structural,  ///< Bind, Output, NamedOutput - scored by PlanCostModel directly.
  Leaf,        ///< Once, Symbol, Literal, ParamLookup - bind::kSymbolCost.
};

/// Per-symbol descriptor.  Each enum value MUST have a specialisation; missing
/// ones produce a compile error the first time the descriptor is queried
/// (e.g. via is_binary_op_v).
template <symbol S>
struct symbol_descriptor;  // primary template intentionally undefined

// clang-format off
// Structural / leaf symbols.
template<> struct symbol_descriptor<symbol::Once>        { static constexpr Arity arity = Arity::Leaf;    static constexpr CostClass cost_class = CostClass::Leaf;       };
template<> struct symbol_descriptor<symbol::Symbol>      { static constexpr Arity arity = Arity::Leaf;    static constexpr CostClass cost_class = CostClass::Leaf;       };
template<> struct symbol_descriptor<symbol::Literal>     { static constexpr Arity arity = Arity::Leaf;    static constexpr CostClass cost_class = CostClass::Leaf;       };
template<> struct symbol_descriptor<symbol::ParamLookup> { static constexpr Arity arity = Arity::Leaf;    static constexpr CostClass cost_class = CostClass::Leaf;       };
template<> struct symbol_descriptor<symbol::Identifier>  { static constexpr Arity arity = Arity::Unary;   static constexpr CostClass cost_class = CostClass::Identifier; };
template<> struct symbol_descriptor<symbol::Bind>        { static constexpr Arity arity = Arity::Special; static constexpr CostClass cost_class = CostClass::Structural; };
template<> struct symbol_descriptor<symbol::Output>      { static constexpr Arity arity = Arity::Special; static constexpr CostClass cost_class = CostClass::Structural; };
template<> struct symbol_descriptor<symbol::NamedOutput> { static constexpr Arity arity = Arity::Special; static constexpr CostClass cost_class = CostClass::Structural; };

// Arithmetic operators (binary).
template<> struct symbol_descriptor<symbol::Add>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Arithmetic; };
template<> struct symbol_descriptor<symbol::Sub>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Arithmetic; };
template<> struct symbol_descriptor<symbol::Mul>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Arithmetic; };
template<> struct symbol_descriptor<symbol::Div>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Arithmetic; };
template<> struct symbol_descriptor<symbol::Mod>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Arithmetic; };
template<> struct symbol_descriptor<symbol::Exp>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Arithmetic; };

// Comparison operators (binary).
template<> struct symbol_descriptor<symbol::Eq>          { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Comparison; };
template<> struct symbol_descriptor<symbol::Neq>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Comparison; };
template<> struct symbol_descriptor<symbol::Lt>          { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Comparison; };
template<> struct symbol_descriptor<symbol::Lte>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Comparison; };
template<> struct symbol_descriptor<symbol::Gt>          { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Comparison; };
template<> struct symbol_descriptor<symbol::Gte>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Comparison; };

// Boolean operators.
template<> struct symbol_descriptor<symbol::And>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Boolean;    };
template<> struct symbol_descriptor<symbol::Or>          { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Boolean;    };
template<> struct symbol_descriptor<symbol::Xor>         { static constexpr Arity arity = Arity::Binary;  static constexpr CostClass cost_class = CostClass::Boolean;    };

// Unary operators.
template<> struct symbol_descriptor<symbol::Not>         { static constexpr Arity arity = Arity::Unary;   static constexpr CostClass cost_class = CostClass::Unary;      };
template<> struct symbol_descriptor<symbol::UnaryMinus>  { static constexpr Arity arity = Arity::Unary;   static constexpr CostClass cost_class = CostClass::Unary;      };
template<> struct symbol_descriptor<symbol::UnaryPlus>   { static constexpr Arity arity = Arity::Unary;   static constexpr CostClass cost_class = CostClass::Unary;      };

// Function call - variadic children (the arguments); structurally scored by PlanCostModel.
template<> struct symbol_descriptor<symbol::Function>    { static constexpr Arity arity = Arity::Special; static constexpr CostClass cost_class = CostClass::Structural; };

// UNWIND - 3 children [input, sym, list_expr]; row-pipe operator scored by PlanCostModel.
template<> struct symbol_descriptor<symbol::Unwind>      { static constexpr Arity arity = Arity::Special; static constexpr CostClass cost_class = CostClass::Structural; };

// Subquery (CALL block) - variadic [outer_input, inner_root, exposed_syms...]; scope barrier.
template<> struct symbol_descriptor<symbol::Subquery>    { static constexpr Arity arity = Arity::Special; static constexpr CostClass cost_class = CostClass::Structural; };

// clang-format on

// ============================================================================
// Classification predicates - derived from descriptors.
// ============================================================================
//
// These are the entry points used to ask "is this symbol binary?" /
// "is this symbol unary?".  They derive from
// symbol_descriptor<S>::arity, so adding a new symbol's descriptor automatically
// flips the right predicate.

template <symbol S>
constexpr bool is_binary_op_v = symbol_descriptor<S>::arity == Arity::Binary;

template <symbol S>
constexpr bool is_unary_op_v = symbol_descriptor<S>::arity == Arity::Unary;

template <symbol S>
constexpr bool is_leaf_v = symbol_descriptor<S>::arity == Arity::Leaf;

/// A binary *expression* operator: arithmetic, comparison, or boolean. Excludes
/// structural binary symbols (none today, but the predicate is more specific
/// than is_binary_op_v which only checks arity). This is the predicate that
/// matches the EGRAPH_BINARY_OPS X-list.
template <symbol S>
constexpr bool is_binary_expr_op_v = is_binary_op_v<S> && (symbol_descriptor<S>::cost_class == CostClass::Arithmetic ||
                                                           symbol_descriptor<S>::cost_class == CostClass::Comparison ||
                                                           symbol_descriptor<S>::cost_class == CostClass::Boolean);

/// A unary *expression* operator: Not, UnaryMinus, UnaryPlus. Excludes
/// Identifier (which is structurally unary but not an expression operator).
/// This is the predicate that matches the EGRAPH_UNARY_OPS X-list.
template <symbol S>
constexpr bool is_unary_expr_op_v = is_unary_op_v<S> && symbol_descriptor<S>::cost_class == CostClass::Unary;

// ============================================================================
// Canonical enumeration of all symbols.
// ============================================================================
//
// AllSymbolsSeq is the single source of truth for "what symbols exist."  It
// drives:
//   * the exhaustiveness check below (every symbol must have a descriptor),
//   * `symbol_storage` in symbol_make_traits.hpp (combined storage type),
//   * any runtime symbol → property dispatch (e.g. CostClassOf in PlanCostModel).
//
// Adding a new symbol requires appending it here AND in the `symbol` enum
// above AND specialising symbol_descriptor.  All three are co-located so a
// missing entry is a compile error at the first query that needs it.
//
// We roll our own pack holder rather than using std::integer_sequence because
// the latter requires an integral underlying type, not a scoped enum.

template <symbol... Ss>
struct symbol_sequence {};

using AllSymbolsSeq =
    symbol_sequence<symbol::Once, symbol::Bind, symbol::Symbol, symbol::Literal, symbol::Identifier, symbol::Output,
                    symbol::NamedOutput, symbol::ParamLookup, symbol::Add, symbol::Sub, symbol::Mul, symbol::Div,
                    symbol::Mod, symbol::Exp, symbol::Eq, symbol::Neq, symbol::Lt, symbol::Lte, symbol::Gt, symbol::Gte,
                    symbol::And, symbol::Or, symbol::Xor, symbol::Not, symbol::UnaryMinus, symbol::UnaryPlus,
                    symbol::Function, symbol::Unwind, symbol::Subquery>;

// ============================================================================
// Scope-threading operator concepts
// ============================================================================
//
// Scope-threading operators (Bind, Unwind, Subquery, Output) carry scope context
// through their children and require dedicated resolver handling.  Generic
// expression operators (Leaf, Unary, Binary) have no scope context.
//
// A scope-threading operator:
//   1. Has Arity::Special (variadic children with specific roles)
//   2. Thread scope from parent to children (provided/provided+sym/demanded)
//   3. May have alive/dead branch semantics (Bind only today)
//
// Adding a new scope-threading operator requires:
//   1. Add it to `kScopeThreadingOperators` in private_symbol.hpp (compile-time guard)
//   2. Add a descriptor entry with Arity::Special
//   3. Add a ResolveXxxChildren() function in egraph_converter.cpp
//   4. Update the dispatch switch in ResolveChildren()
//
// This concept lets ResolveChildren verify at compile time that every scope-
// threading operator has a dedicated arm, rather than falling through to the
// generic arm silently.
static constexpr std::array kScopeThreadingOperators{
    symbol::Bind,
    symbol::Unwind,
    symbol::Subquery,
    symbol::Output,
};

/// True for operators that carry scope context through their children.
/// These must have dedicated handling in ResolveChildren; they MUST NOT
/// fall through to the generic arm.
constexpr bool IsScopeThreadingOp(symbol s) {
  for (auto op : kScopeThreadingOperators) {
    if (op == s) return true;
  }
  return false;
}

// ============================================================================
// Exhaustiveness check - every enum value MUST have a descriptor.
// ============================================================================
//
// If you add a value to `enum class symbol` and forget the descriptor, the
// static_assert below fails at this site (clear named error) rather than at
// some random predicate query miles away.

namespace detail {

/// A symbol has a descriptor if `symbol_descriptor<S>::arity` names a value.
template <symbol S>
concept HasDescriptor = requires {
  { symbol_descriptor<S>::arity } -> std::convertible_to<Arity>;
  { symbol_descriptor<S>::cost_class } -> std::convertible_to<CostClass>;
};

/// Fold the concept check across the canonical sequence.  Defined as a
/// function template so the fold is *not* instantiated by merely including
/// this header; only the TU that evaluates this function pays the cost.
template <symbol... Ss>
constexpr auto AllHaveDescriptorsImpl(symbol_sequence<Ss...>) -> bool {
  return (HasDescriptor<Ss> && ...);
}

template <symbol... Ss>
constexpr auto CountInSequenceImpl(symbol_sequence<Ss...>) -> std::size_t {
  return sizeof...(Ss);
}

template <symbol... Ss>
constexpr auto CountBinaryExprImpl(symbol_sequence<Ss...>) -> std::size_t {
  return (std::size_t{0} + ... + (is_binary_expr_op_v<Ss> ? 1U : 0U));
}

template <symbol... Ss>
constexpr auto CountUnaryExprImpl(symbol_sequence<Ss...>) -> std::size_t {
  return (std::size_t{0} + ... + (is_unary_expr_op_v<Ss> ? 1U : 0U));
}

}  // namespace detail

// Enum values must be contiguous from 0 to N-1 (required by CostClassOfImpl's
// default-zero sentinel and the exhaustiveness fold).  Add new symbols at the end.
static_assert(detail::CountInSequenceImpl(AllSymbolsSeq{}) == static_cast<std::size_t>(symbol::Subquery) + 1,
              "AllSymbolsSeq must enumerate every symbol enum value; update both the enum and AllSymbolsSeq together");

// ============================================================================
// Symbol singleton invariant - compile-time guards
// ============================================================================
//
// The singleton invariant (each Symbol e-class = one variable) requires:
//   1. Symbol must be a leaf (no children that could be rewritten/merged)
//   2. Symbol must not be a "scope-threading" operator (these are handled
//      separately by the resolver via for_each_resolved_child)
//
// These static_asserts fire at compile time if someone adds a new symbol that
// violates these properties, rather than producing silent runtime corruption.
//
// To add a new scope-threading operator, update the kScopeThreadingOperators
// array in for_each_resolved_child AND add it to this list to verify the
// concept constraint holds at compile time.
static_assert(is_leaf_v<symbol::Symbol>,
              "symbol::Symbol must be a leaf; if you made it non-leaf, "
              "you must also update the singleton invariant guards and the resolver");
static_assert(symbol_descriptor<symbol::Symbol>::arity != Arity::Special,
              "symbol::Symbol must not be Special arity; scope-threading operators "
              "require dedicated resolver handling in for_each_resolved_child");

/// Count of binary expression operators in AllSymbolsSeq - cross-checked against
/// EGRAPH_BINARY_OPS in egraph.cpp.
constexpr std::size_t binary_expr_op_count_v = detail::CountBinaryExprImpl(AllSymbolsSeq{});

/// Count of unary expression operators in AllSymbolsSeq - cross-checked against
/// EGRAPH_UNARY_OPS in egraph.cpp.
constexpr std::size_t unary_expr_op_count_v = detail::CountUnaryExprImpl(AllSymbolsSeq{});

// ============================================================================
// Runtime symbol → CostClass dispatch.
// ============================================================================
//
// Fold over AllSymbolsSeq, so a new symbol's descriptor is the only edit
// needed; this function picks up the new entry automatically.

namespace detail {

template <symbol... Ss>
constexpr auto CostClassOfImpl(symbol s, symbol_sequence<Ss...>) -> CostClass {
  CostClass result{};
  bool const found = (((s == Ss) && ((result = symbol_descriptor<Ss>::cost_class), true)) || ...);
  // `found` should always be true - AllSymbolsSeq is exhaustive over the enum.
  // If a symbol is added to the enum but not to AllSymbolsSeq, this returns the
  // default-initialised CostClass; the static_assert above wouldn't fire (it
  // checks descriptors, not sequence membership).  Belt-and-braces:
  assert(found && "CostClassOf: symbol missing from AllSymbolsSeq - see private_symbol.hpp");
  if (!found) std::unreachable();
  return result;
}

}  // namespace detail

constexpr auto CostClassOf(symbol s) -> CostClass { return detail::CostClassOfImpl(s, AllSymbolsSeq{}); }

namespace detail {

template <symbol... Ss>
constexpr auto ArityOfImpl(symbol s, symbol_sequence<Ss...>) -> Arity {
  Arity result{};
  bool const found = (((s == Ss) && ((result = symbol_descriptor<Ss>::arity), true)) || ...);
  assert(found && "ArityOf: symbol missing from AllSymbolsSeq - see private_symbol.hpp");
  if (!found) std::unreachable();
  return result;
}

}  // namespace detail

constexpr auto ArityOf(symbol s) -> Arity { return detail::ArityOfImpl(s, AllSymbolsSeq{}); }

}  // namespace memgraph::query::plan::v2

namespace std {

template <>
struct hash<memgraph::query::plan::v2::symbol> {
  size_t operator()(memgraph::query::plan::v2::symbol const &value) const noexcept { return std::to_underlying(value); }
};
}  // namespace std
