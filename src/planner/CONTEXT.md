# mg-planner — Extraction Pipeline

The `mg-planner` e-graph library (`src/planner/`) provides the generic infrastructure for building and extracting from e-graphs. This context covers the extraction pipeline: the stages that turn a saturated e-graph into a concrete, ordered plan.

## Language

**Frontier**:
The Pareto set of cost alternatives computed for one e-class during bottom-up cost propagation. An e-class with multiple non-dominated enodes has multiple alternatives; dominated ones are pruned.
_Avoid_: "cost result", "selection set"

**Extraction**:
The two-stage pipeline — `ComputeFrontiers` then a Resolver — that turns a saturated e-graph into a children-before-parents ordered sequence of (key, enode) entries ready for a builder to consume.
_Avoid_: "plan extraction" when referring to the generic pipeline (use that term only for the `plan_v2`-specific step that produces a `LogicalOperator`)

**Resolver**:
A callable `(egraph, frontier_map, root, vector<Entry> &out)` that traverses the frontier map and fills `out` in children-before-parents (post-order) order. The resolver owns both alt selection and traversal ordering. It must guarantee every child key appears in `out` before its parent.
_Avoid_: "selector" (selection is a consequence, traversal is the contract)

**DfsPostOrder**:
Generic DFS scaffolding provided by `extractor.hpp`. Handles deduplication via a caller-supplied `seen` set and post-order emission. The caller supplies a `resolve(key, visit_child) -> Entry` callback that does the per-node work; `DfsPostOrder` supplies the recursion and emit structure.
_Avoid_: calling it "the resolver" — it is scaffolding a resolver uses, not a resolver itself

**PickBest**:
Generic helper in `pareto_frontier.hpp`. Iterates `frontier.alts()` and returns a pointer to the min-cost alternative satisfying a caller-supplied predicate. Returns `nullptr` if none matches. Callers decide what a null result means (throw, skip, assert).
_Avoid_: "pick compatible" as a generic term — `pick_compatible` is a `plan_v2`-specific wrapper around `PickBest`

**Dim**:
A single axis of a Pareto frontier, expressed as `Dim<MemPtr, Cmp>` where `MemPtr` projects an Alt to a comparable value and `Cmp` is a comparator struct (`LowerIsBetter`, `SmallerSubsetIsBetter`, `LargerSubsetIsBetter`). `ParetoFrontier` is parameterised by `Alt` and a pack of `Dim`s; there is no separate "dominance functor" type. See [ADR 0007](../../docs/adr/0007-pareto-frontier-dims-as-type-parameter.md).
_Avoid_: "dominance function", "axis comparator"

**DefaultResolver**:
A test-only resolver in `test_support/extract.hpp`. Wraps `DfsPostOrder` with min-cost alt selection (`CostResult::resolve()`) and unconditional child traversal. Not suitable for cost models whose chosen alt may exclude some children (e.g. alive/dead Bind semantics).
_Avoid_: using `DefaultResolver` in production code

**FrontierContext**:
Reusable buffer owning the frontier map and the per-frame buffer pool used by `ComputeFrontiers`. Declared outside hot loops; `clear()` preserves capacity across calls.
_Avoid_: "extraction context" — that name belonged to the old four-stage wrapper (`ExtractionContext`) which no longer exists

## Relationships

- `ComputeFrontiers` fills a **FrontierContext**; the **FrontierContext** is then passed to a **Resolver**
- A **Resolver** uses **DfsPostOrder** as its traversal scaffold
- **DfsPostOrder** calls `PickBest` (indirectly, via the per-node callback the resolver supplies)
- **PickBest** operates on a **Frontier**'s `alts()` range
- `plan_v2`'s `PlanResolver` is a **Resolver** with a context-sensitive key (`ResolvedKey{eclass, provided, demanded}`) and scope-threading child dispatch (`ResolveChildren`)
- **DefaultResolver** is a **Resolver** for tests and benchmarks only

## Example dialogue

> **Dev:** "I want to add a new resolver that tries two different alt-selection strategies and picks the better final cost."
> **Domain expert:** "You'd write a new Resolver callable. Use `DfsPostOrder` for the traversal — it handles the seen-set and post-order emit. Your per-node callback calls `PickBest` twice with different predicates and picks the winner. `FrontierContext` gives you the frontier map after `ComputeFrontiers` runs."
> **Dev:** "Do I need `ExtractionContext`?"
> **Domain expert:** "That no longer exists. You own a `FrontierContext` for `ComputeFrontiers` and a `vector<Entry>` for your resolver's output. Those are two separate variables."

## Flagged ambiguities

- "extraction context" was used loosely to mean both the old four-stage `ExtractionContext` wrapper (deleted) and the current `FrontierContext`. Resolved: use **FrontierContext** exclusively for the reusable ComputeFrontiers buffer.
- "resolver" was used to mean both the generic callable contract and the specific `DefaultResolver` test helper. Resolved: **Resolver** is the contract; **DefaultResolver** is one implementation for test use only.
