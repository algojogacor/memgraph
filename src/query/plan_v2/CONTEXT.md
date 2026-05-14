# plan_v2 — Cypher Cost Model & Resolution

Plan-v2 sits on top of the generic `mg-planner` e-graph (`src/planner/`, see [its CONTEXT.md](../../planner/CONTEXT.md)) and adds the Cypher-specific shape: scope semantics, the cost model over `Alternative`, and resolver threading. This document fixes the vocabulary used in `egraph_converter.cpp`, `bind_semantics.hpp`, and the resolver.

## Language

### Alt kinds

**Alternative**:
One candidate plan reachable at an e-class. Carries `cost`, `cardinality`, a `kind`, plus two scope-related fields (`introduces`, `required`) whose meaning depends on the kind.
_Avoid_: "alt" (informal but acceptable in code), "candidate plan" (overloaded with rewrite-time alternatives)

**Kind**:
A property of every **Alternative**, either *operator* or *expression*, set at construction by the cost-model helper that emits the Alt. Determines which of `introduces` / `required` carries information and which is constantly empty. Operator e-classes only ever hold operator Alts; expression e-classes only ever hold expression Alts; no rewrite merges the two.
_Avoid_: "type" (overloaded), "category"

**Operator Alt**:
An **Alternative** for a row-pipe or scope-threading enode: `Once`, `Bind`, `Unwind`, `Output`, `Subquery`, (future) `Filter`, `MATCH`, etc. Carries `introduces`; its `required` is always ∅.

**Expression Alt**:
An **Alternative** for an expression enode: `Identifier`, `Literal`, `Symbol`, `Add`/`Sub`/…, `Function`, `NamedOutput`, etc. Carries `required`; its `introduces` is always ∅.

### The two scope sets on an Alt

**introduces** (operator Alts):
The set of symbols this Alt's pipeline establishes for operators above it. Computed bottom-up at cost-model time as `input.introduces ∪ own_syms`. Property of the subtree, frozen on the Alt.
_Avoid_: "provides" (too close to **in_scope** below), "exports" (only fits Output)

**required** (expression Alts):
The set of symbols this expression's free `Identifier` references demand from scope. Computed bottom-up as the union over expression children. Property of the subtree, frozen on the Alt.
_Avoid_: "demands", "uses" — "uses" specifically reserved for a future additive set that does not subtract at binders.

**own_syms** (operator Alts, derived):
The symbols this single operator binds in its own right, before unioning with its operator child's `introduces`. `Bind`/`Unwind`: `{enode.children()[1]}`. `Output`: `{NamedOutput.sym for each NamedOutput child}`. `Subquery`: `exposed_syms` from `enode.children().subspan(2)`. `Once` and expression operators: `∅`.
_Avoid_: "binds" (verb confusion with the `Bind` operator)

### Resolver-side terms

**in_scope** (resolver, top-down):
The set of symbols currently visible at the resolver's descent position, from any source. Distinct from any Alt's `introduces`: `in_scope` is a property of the walk position, not of a subtree. Always equal to or larger than the relevant Alt's `introduces` (it includes ancestor contributions; `introduces` does not).
_Avoid_: "provided", "available", "visible_syms"

**must_introduce** (resolver, top-down):
The set of symbols a child is obliged to establish at its own `introduces`. Set by the parent on entering a child: `must_introduce = parent.chosen.introduces − parent.own_syms`. Resolver picks the cheapest operator-Alt at the child e-class with `introduces ⊇ must_introduce`.
_Avoid_: "demanded_introduces" (older name; superseded)

**Resolver's choice predicate**:
At an operator e-class: pick the min-cost Alt with `introduces ⊇ must_introduce`. At an expression e-class: pick the min-cost Alt with `required ⊆ in_scope`. Equivalent forms of the same fitness check, one per kind.

## Relationships

- An **Alternative** is exactly one **Kind** (operator or expression) for its lifetime
- An operator **Alternative**'s `introduces = input.introduces ∪ own_syms` (bottom-up, at construction)
- Operator-Alt construction validates `expr_child.required ⊆ input.introduces` for each expression child; failing combinations are not emitted
- The resolver's `in_scope` flows top-down; child's `in_scope = parent.in_scope ∪ (parent.chosen.introduces − parent.own_syms)` for expression children, `parent.in_scope` for operator children (Subquery's inner is the barrier exception: `in_scope = exposed_syms`)
- The resolver's `must_introduce` flows top-down; child operator gets `must_introduce = parent.chosen.introduces − parent.own_syms`; child expression gets `must_introduce = ∅`

## Construction-time validation, by operator

**`Bind`** (children `[input_op, sym_leaf, expr]`):
- Alive Alt: emit when `expr.required ⊆ input.introduces` AND `sym ∈ referenced_syms` (egraph-wide filter). `introduces = input.introduces ∪ {sym}`.
- Dead Alt: emit always. `introduces = input.introduces`. Expression is not evaluated; no validation on expr.

**`Unwind`** (children `[input_op, sym_leaf, list_expr]`):
- Alt is always alive (Unwind always binds). Emit when `list_expr.required ⊆ input.introduces`. `introduces = input.introduces ∪ {sym}`.

**`Output`** (children `[pipe, named_out_1, named_out_2, …]`):
- Emit when `∪ named_out_i.required ⊆ pipe.introduces`. `introduces = pipe.introduces ∪ {NO.sym for each NamedOutput child}`.

**`Subquery`** (children `[outer, inner, sym_1, sym_2, …]`):
- Inner is constructed in isolation; its `required` is always `∅` by virtue of being an operator Alt. Emit for each (outer × inner) pair. `introduces = outer.introduces ∪ exposed_syms`. The barrier strips `inner.introduces` from what crosses outward — only `exposed_syms` does.

## Example dialogue

> **Dev:** "Why doesn't the Bind alt's `required` carry the expr's demand upward like it does in the OutputCombine code I'm reading from before the refactor?"
> **Domain expert:** "Operator Alts don't carry `required` at all — that's the kind dichotomy. Bind's expr-required is consumed at Bind's *own* construction: if `expr.required ⊆ input.introduces`, the alive Alt is emitted; if not, that combo is rejected. Nothing escapes upward through an operator. The resolver later visits expr with `in_scope = parent.in_scope ∪ (chosen.introduces − {sym})`, which is guaranteed to cover `expr.required` by virtue of the construction-time check."

> **Dev:** "Then `chosen.introduces − {sym}` is the same thing as `input.introduces` for an alive Bind?"
> **Domain expert:** "For an alive Bind Alt specifically, yes — `chosen.introduces = input.introduces ∪ {sym}`, so subtracting `{sym}` recovers `input.introduces`. The subtraction is written that way to read uniformly: every operator's resolver rule subtracts `own_syms` from its `chosen.introduces`. For Output, `own_syms` is the NamedOutput syms; for Subquery, the exposed syms; for Once, the empty set."

> **Dev:** "And `in_scope` vs `introduces`?"
> **Domain expert:** "`introduces` is a static property of an Alt — what its subtree's pipeline contributes, frozen at cost-model time. `in_scope` is the resolver's current position: every symbol visible *here, right now*, from any source. They coincide at the root with `parent.in_scope = ∅`, but they diverge inside Subquery's inner subtree, where `in_scope = exposed_syms` rather than the parent's value. Use `introduces` when asking 'what does this Alt offer?'; use `in_scope` when asking 'what's visible at this descent point?'."

## Flagged ambiguities

- "provided" / "provides" was previously used for both the Alt field and the resolver context. Resolved: Alt field is **introduces**; resolver context is **in_scope**.
- "required" used to live on operator Alts as well, carrying residual demand bubbled up from descendant expressions. Resolved under the kind dichotomy: operator Alts have `required = ∅`; expression demand is absorbed at the construction-time validation of the enclosing operator Alt, not propagated upward.
- "demanded_introduces" was the older name for **must_introduce**. Code may still reference the older name during the transition.
- "alive Bind" vs "dead Bind" is derived at read sites from the chosen Alt's `introduces`: alive iff `sym ∈ chosen.introduces` where `sym = enode.children()[1]`. The resolver dispatches Bind/Unwind child handling on this check; the builder distinguishes alive (3 emitted children: input, sym, expr) from dead (1 emitted child: input) by the resolver-recorded child count. No `AliveTag` field exists on `Alternative` or `TopoEntry`.
