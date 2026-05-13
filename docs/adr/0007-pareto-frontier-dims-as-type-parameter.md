# Pareto frontier: dimensions are the type parameter

`ParetoFrontier` was parameterised by an opaque `DominanceFn` callable `(Alt const&, Alt const&) -> std::partial_ordering`. Every production dominance functor (`AlternativeDominance`, `TestDominance`, `DemandDominance`) was a thin wrapper that called `pareto_compare(a, b, dim<...>, dim<...>, ...)`. The opaque-callable abstraction hid structural information the pruner could exploit, while delivering flexibility nobody used.

The frontier is now parameterised by `Alt` and a variadic pack of `Dim<MemPtr, Cmp>` types directly: `ParetoFrontier<Alt, Dims...>`. The dominance wrapper structs are deleted; the `pareto_compare` free helper and the `dim<MemPtr>(cmp)` factory are deleted; the comparator lambdas are upgraded to structs (`LowerIsBetter`, `SmallerSubsetIsBetter`, `LargerSubsetIsBetter`) so capability hooks can hang off them.

## Why

With dims visible to the frontier, the pruner can:

- **Sort `alts_` once by the lex key of all totally-ordered dims** (auto-detected via the comparator's `<=>` return type; if `compare` returns `strong_ordering` or `weak_ordering`, the dim is totally ordered and contributes a sort key). For production `Alternative` this fixes the `cost` and `cardinality` axes by position, so the inner pairwise check only invokes the set-merge comparators.
- **Maintain a sorted invariant on `alts_` across every public op.** Constructor / `flat_map` / `cartesian_product` sort; `merge_in_place` uses `std::inplace_merge` over the two pre-sorted halves.
- **Replace the two-pass mark-and-compact prune with a single forward sweep.** `prune_with_pruned_prefix`'s `pruned_prefix` parameter, the `boost::small_vector<bool, 64>` dominated-flag buffer, and the two-pass structure all go away. The remaining algorithm is textbook skyline maintenance.

The complexity stays O(n²) in pairs — the set-inclusion dims (`smaller_subset_is_better` / `larger_subset_is_better`) are partial orders, so the divide-and-conquer skyline approach that gets to O(n log^(d-1) n) for purely totally-ordered axes does not apply. The wins here are constant-factor (fewer dim comparisons per pair, cache-friendlier sweep, no flag-vector allocation), not asymptotic.

## Considered alternatives

**Keep `DominanceFn` opaque; push the sort optimisation down into `CostResultBase` only.** Rejected: `CostResultBase` would need a protected hook into `ParetoFrontier`'s private state, and the generic class would carry an algorithm it never used. The "generic Pareto over any dominance" framing was paying rent in indirection without any caller exercising it — no production code writes a non-decomposable dominance.

**Keep `dim<MemPtr>(cmp)` as a lambda factory; bolt totality on via a wrapper trait.** Rejected: the lambdas have no place to expose `sort_key`, so anything beyond "(Alt, Alt) -> ordering" requires a parallel `sortable(...)` decorator. Two ways to spell the same thing, and the wrapper soup gets ugly when extending to prefilters.

**Expose a separate `prefilter(a, b)` hook on comparators and have the pruner combine prefilter results across dims before any full `compare`.** Captures a real cross-dim short-circuit (when two set-dim prefilters disagree on direction, both set merges can be skipped). Rejected for Day 1: pays a permanent complexity cost in the pruner (two-phase pair check, fold-of-narrowed-orderings logic) for a constant-factor win whose magnitude depends on graph shape. The cardinality short-circuit moves inside each `compare` instead — single-dim wins are kept; cross-dim wins are deferred until profiling justifies them. Reverse migration (adding `prefilter` later) is local.

## Consequences

- `pareto_frontier.hpp` public surface shrinks: `DominanceRelation`, `pareto_compare`, `dim`, `lower_is_better`, `smaller_subset_is_better`, `larger_subset_is_better` are removed. New surface: `ParetoDimension`, `TotallyOrderedDim`, `Dim`, three comparator structs.
- `alts()`'s iteration order is now part of the contract: lex-sorted by the totally-ordered dims in declaration order. No production caller depended on the previous unspecified order (`PickBest` is a linear scan; resolvers iterate without ordering assumptions); test snapshots that compared frontiers by exact element order may need updating.
- `mutate_pruning_invariant_preserving`'s contract strengthens from "preserves dominance partial order" to "monotone on each totally-ordered dim". Both production call sites (uniform `cost += K`, same-value `cardinality = c`) already satisfy the stronger contract.
- `CostFrontier`, `TestFrontier`, `DemandFrontier` lose their dominance-wrapper structs; the per-axis docstrings move to comment blocks above the `using` aliases.
- `resolve()` on `CostResultBase` is now O(1) — the min-cost alt is at the head of the sorted vector — instead of an O(n) `min_element` scan. Not a load-bearing change; resolvers don't call `resolve()` in tight loops.
