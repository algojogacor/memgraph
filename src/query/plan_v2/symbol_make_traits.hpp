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

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "query/plan_v2/builtin_functions.hpp"
#include "query/plan_v2/egraph.hpp"
#include "query/plan_v2/private_symbol.hpp"

namespace memgraph::query::plan::v2 {

// ========================================================================
// symbol_make_traits - single place where per-symbol semantics live.
//
// Every specialisation provides:
//   - storage_type: what auxiliary side-data this symbol needs (interning
//     maps, counters, etc). Empty struct if none.
//   - make(storage, user_args...) -> lowered_node: maps the user-facing
//     constructor arguments to (children, optional disambiguator), updating
//     storage as needed. The caller (egraph::impl::Make) does the actual
//     emplace into the core e-graph using the returned lowered form.
//
// The lowering contract lives in exactly one place (this header). Adding a
// new symbol is one specialisation plus one facade method on egraph.
// ========================================================================

struct lowered_node {
  utils::small_vector<eclass> children;
  std::optional<uint64_t> disambiguator;
};

template <symbol S>
struct symbol_make_traits;

/// Concept every symbol_make_traits<S> specialisation must satisfy for a given
/// user-arg pack. Verifies both that the storage_type is well-formed and that
/// make() is callable with (storage&, Args...) returning lowered_node, so a
/// malformed trait or a wrong-arity call fails at the constraint with a clear
/// message rather than deep inside impl::Make.
template <typename T, typename... Args>
concept SymbolMakeTraits =
    std::is_default_constructible_v<typename T::storage_type> && requires(typename T::storage_type &s, Args &&...args) {
      { T::make(s, std::forward<Args>(args)...) } -> std::same_as<lowered_node>;
    };

/// Once: auto-incrementing counter
template <>
struct symbol_make_traits<symbol::Once> {
  struct storage_type {
    uint64_t counter = 0;
  };

  static auto make(storage_type &s) -> lowered_node { return {.children = {}, .disambiguator = s.counter++}; }
};

/// Symbol: position -> name mapping
template <>
struct symbol_make_traits<symbol::Symbol> {
  struct storage_type {
    std::map<int32_t, std::string> store;
  };

  static auto make(storage_type &s, int32_t pos, std::string_view name) -> lowered_node {
    s.store.try_emplace(pos, std::string{name});
    return {.children = {}, .disambiguator = static_cast<uint64_t>(pos)};
  }
};

/// Literal: value -> id mapping
template <>
struct symbol_make_traits<symbol::Literal> {
  struct storage_type {
    std::map<storage::ExternalPropertyValue, uint64_t> store;
    uint64_t next_id = 0;
  };

  static auto make(storage_type &s, storage::ExternalPropertyValue const &value) -> lowered_node {
    auto [it, inserted] = s.store.try_emplace(value, s.next_id);
    if (inserted) ++s.next_id;
    return {.children = {}, .disambiguator = it->second};
  }
};

/// ParamLookup: no storage, position IS the disambiguator
template <>
struct symbol_make_traits<symbol::ParamLookup> {
  struct storage_type {};

  static auto make(storage_type &, int32_t pos) -> lowered_node {
    return {.children = {}, .disambiguator = static_cast<uint64_t>(pos)};
  }
};

/// Bind: no storage, just children
template <>
struct symbol_make_traits<symbol::Bind> {
  struct storage_type {};

  static auto make(storage_type &, eclass input, eclass sym, eclass expr) -> lowered_node {
    return {.children = utils::small_vector{input, sym, expr}, .disambiguator = std::nullopt};
  }
};

/// Identifier: no storage, just child
template <>
struct symbol_make_traits<symbol::Identifier> {
  struct storage_type {};

  static auto make(storage_type &, eclass sym) -> lowered_node {
    return {.children = utils::small_vector{sym}, .disambiguator = std::nullopt};
  }
};

/// Output: no storage, prepends input to children
template <>
struct symbol_make_traits<symbol::Output> {
  struct storage_type {};

  static auto make(storage_type &, eclass input, std::vector<eclass> named_outputs) -> lowered_node {
    auto children = utils::small_vector<eclass>{};
    children.reserve(named_outputs.size() + 1);
    children.push_back(input);
    std::ranges::copy(named_outputs, std::back_inserter(children));
    return {.children = std::move(children), .disambiguator = std::nullopt};
  }
};

/// NamedOutput: name -> id mapping + children
template <>
struct symbol_make_traits<symbol::NamedOutput> {
  struct storage_type {
    std::map<std::string, uint64_t> store;
    uint64_t next_id = 0;
  };

  static auto make(storage_type &s, std::string_view name, eclass sym, eclass expr) -> lowered_node {
    auto [it, inserted] = s.store.try_emplace(std::string{name}, s.next_id);
    if (inserted) ++s.next_id;
    return {.children = utils::small_vector{sym, expr}, .disambiguator = it->second};
  }
};

/// Function: name -> id mapping, with BuiltinKind cached at insertion time
/// so the cost model and estimator can dispatch on an array lookup instead
/// of a per-call string compare.  Children are the argument e-classes; the
/// disambiguator is the function id, namespace-shared between builtins and
/// UDFs.
template <>
struct symbol_make_traits<symbol::Function> {
  struct storage_type {
    /// name -> id, mirroring NamedOutput / Symbol / Literal's `store`
    /// convention so the Builder picks up the same field name across all
    /// interner-backed traits.
    std::map<std::string, uint64_t> store;
    /// id -> FunctionInfo (parallel to `store`'s id range).  Read by the
    /// estimator and the Builder to recover the exact name + cached
    /// BuiltinKind without re-classifying.  Kept in lockstep with `store`
    /// by `intern` - external code must not insert into either directly.
    std::vector<FunctionInfo> info;

    /// Intern a function name, classifying its BuiltinKind on first sight.
    /// Returns the stable id for that name; identical names always return
    /// the same id.  This is the only place `store` and `info` are written,
    /// keeping the parallel-array invariant local to the trait.
    auto intern(std::string_view name) -> uint64_t {
      auto [it, inserted] = store.try_emplace(std::string{name}, info.size());
      if (inserted) {
        info.push_back(FunctionInfo{.name = std::string{name}, .kind = BuiltinKindFor(name)});
      }
      return it->second;
    }
  };

  static auto make(storage_type &s, std::string_view name, std::vector<eclass> args) -> lowered_node {
    return {.children = utils::small_vector<eclass>(args.begin(), args.end()), .disambiguator = s.intern(name)};
  }
};

/// Unwind: no storage, mirrors Bind's [input, sym, list_expr] shape so the
/// resolver's alive-Bind dispatch covers Unwind without a second branch.
template <>
struct symbol_make_traits<symbol::Unwind> {
  struct storage_type {};

  static auto make(storage_type &, eclass input, eclass sym, eclass list_expr) -> lowered_node {
    return {.children = utils::small_vector{input, sym, list_expr}, .disambiguator = std::nullopt};
  }
};

/// Subquery: no storage; children are [outer_input, inner_root, exposed_syms...].
/// Variadic to encode the projection set of the inner block as direct e-graph
/// children, so the cost case and resolver don't have to peek into inner_root's
/// enode shape to discover what crosses the scope barrier.
template <>
struct symbol_make_traits<symbol::Subquery> {
  struct storage_type {};

  static auto make(storage_type &, eclass outer_input, eclass inner_root, std::vector<eclass> exposed_syms)
      -> lowered_node {
    auto children = utils::small_vector<eclass>{};
    children.reserve(2 + exposed_syms.size());
    children.push_back(outer_input);
    children.push_back(inner_root);
    std::ranges::copy(exposed_syms, std::back_inserter(children));
    return {.children = std::move(children), .disambiguator = std::nullopt};
  }
};

/// Binary operator: no storage, just two children
template <symbol S>
  requires(is_binary_op_v<S>)
struct symbol_make_traits<S> {
  struct storage_type {};

  static auto make(storage_type & /*s*/, eclass lhs, eclass rhs) -> lowered_node {
    return {.children = utils::small_vector{lhs, rhs}, .disambiguator = std::nullopt};
  }
};

/// Unary operator: no storage, just one child
template <symbol S>
  requires(is_unary_op_v<S>)
struct symbol_make_traits<S> {
  struct storage_type {};

  static auto make(storage_type & /*s*/, eclass operand) -> lowered_node {
    return {.children = utils::small_vector{operand}, .disambiguator = std::nullopt};
  }
};

// ========================================================================
// Combined storage using inheritance (like the overloads trick)
// ========================================================================

template <typename... Ts>
struct combined_storage : Ts... {};

template <symbol... Ss>
using symbol_storage_for = combined_storage<typename symbol_make_traits<Ss>::storage_type...>;

namespace detail {

// Derive symbol_storage from the canonical AllSymbolsSeq instead of repeating
// the full list here.  Adding a symbol to AllSymbolsSeq picks up its storage
// automatically; no second manual list to keep in sync.
template <typename>
struct SymbolStorageFromSeq;

template <symbol... Ss>
struct SymbolStorageFromSeq<symbol_sequence<Ss...>> {
  using type = symbol_storage_for<Ss...>;
};

}  // namespace detail

using symbol_storage = detail::SymbolStorageFromSeq<AllSymbolsSeq>::type;

}  // namespace memgraph::query::plan::v2
