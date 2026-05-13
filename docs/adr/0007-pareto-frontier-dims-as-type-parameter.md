# Pareto frontier: dimensions are the type parameter

`ParetoFrontier` was parameterised by an opaque `DominanceFn` callable `(Alt const&, Alt const&) -> std::partial_ordering`. Every production dominance functor (`AlternativeDominance`, `TestDominance`, `DemandDominance`) was a thin wrapper that called `pareto_compare(a, b, dim<...>, dim<...>, ...)`. The opaque-callable abstraction hid structural information the pruner could exploit, while delivering flexibility nobody used.

The frontier is now parameterised by `Alt` and a variadic pack of `Dim<MemPtr, Cmp>` types directly: `ParetoFrontier<Alt, Dims...>`. The dominance wrapper structs are deleted; the `pareto_compare` free helper and the `dim<MemPtr>(cmp)` factory are deleted; the comparator lambdas are upgraded to structs (`LowerIsBetter`, `SmallerSubsetIsBetter`, `LargerSubsetIsBetter`) so capability hooks can hang off them.

## Why

With dims visible to the frontier as types, the pruner can:

- **Replace the two-pass mark-and-compact prune with a single forward sweep.** `prune_with_pruned_prefix`'s `pruned_prefix` parameter, the `boost::small_vector<bool, 64>` dominated-flag buffer, and the two-pass structure all go away. The remaining algorithm is textbook skyline maintenance.
- **Compile-time dispatch over the per-dim comparators**, removing the indirect call through `DominanceFn` and letting the compiler inline the fold over `Dims::compare`.

The complexity stays O(n²) in pairs — the set-inclusion dims are partial orders, so the divide-and-conquer skyline approach that gets to O(n log^(d-1) n) for purely totally-ordered axes does not apply. The wins here are structural (less surface, fewer abstraction layers) rather than asymptotic.

## What was tried and rolled back

An initial follow-on optimisation maintained a lex-sorted invariant on `alts_` using the totally-ordered dims as a sort key, with the intent of letting the pruner skip those dim compares during dominance checks. The sort scaffolding was added (`LexLess`, `lex_compare`, `lex_step`, `has_totally_ordered_dim_v`, the `TotallyOrderedDim` concept, `std::inplace_merge` in `merge_in_place`) but never paid back: skipping the totally-ordered dim compares saves ~2 cycles out of ~100–1000 per pair-check (the partial-ordered set-merges dominate), which is sub-1% of prune time and well under 0.1% of overall planner time. The sort itself cost O(n log n) per public op plus the maintenance surface. `resolve()` still scans via `min_element`. No caller depended on a sorted `alts()`. The scaffolding was removed in a follow-up commit; the variadic-dims restructure stands on its own.

## Considered alternatives

**Keep `DominanceFn` opaque; push the sort optimisation down into `CostResultBase` only.** Rejected: `CostResultBase` would need a protected hook into `ParetoFrontier`'s private state, and the generic class would carry an algorithm it never used. The "generic Pareto over any dominance" framing was paying rent in indirection without any caller exercising it — no production code writes a non-decomposable dominance.

**Keep `dim<MemPtr>(cmp)` as a lambda factory; bolt totality on via a wrapper trait.** Rejected: the lambdas have no place to expose `sort_key`, so anything beyond "(Alt, Alt) -> ordering" requires a parallel `sortable(...)` decorator. Two ways to spell the same thing, and the wrapper soup gets ugly when extending to prefilters.

**Expose a separate `prefilter(a, b)` hook on comparators and have the pruner combine prefilter results across dims before any full `compare`.** Captures a real cross-dim short-circuit (when two set-dim prefilters disagree on direction, both set merges can be skipped). Rejected for Day 1: pays a permanent complexity cost in the pruner (two-phase pair check, fold-of-narrowed-orderings logic) for a constant-factor win whose magnitude depends on graph shape. The cardinality short-circuit moves inside each `compare` instead — single-dim wins are kept; cross-dim wins are deferred until profiling justifies them. Reverse migration (adding `prefilter` later) is local.

## Consequences

- `pareto_frontier.hpp` public surface shrinks: `DominanceRelation`, `pareto_compare`, `dim`, `lower_is_better`, `smaller_subset_is_better`, `larger_subset_is_better` are removed. New surface: `ParetoDimension`, `Dim`, three comparator structs, `dominance_compare<Dims...>(a, b)` free function.
- `alts()`'s iteration order remains implementation-defined. Callers use `PickBest` (linear scan) or iterate without ordering assumptions.
- `mutate_pruning_invariant_preserving`'s contract is unchanged: "preserves the result of `dominance_compare<Dims...>(A, B)` for every pair (A, B)". Both production call sites (uniform `cost += K`, same-value `cardinality = c`) satisfy it.
- `CostFrontier`, `TestFrontier`, `DemandFrontier` lose their dominance-wrapper structs; the per-axis docstrings move to comment blocks above the `using` aliases.
