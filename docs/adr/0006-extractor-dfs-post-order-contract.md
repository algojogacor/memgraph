# Extractor resolver contract: DFS post-order vector, not SelectionMap + Kahn sort

The generic extractor in `mg-planner` previously used a two-pass pipeline: a Resolver filled a flat `SelectionMap<cost_t>` (one entry per e-class), then `CollectDependencies` + `TopologicalSort` derived a children-before-parents order via Kahn's algorithm. The production resolver (`PlanResolver` in `plan_v2`) never used this pipeline — it produced a `vector<TopoEntry>` in DFS post-order directly, bypassing all three stages after `ComputeFrontiers`. The two approaches diverged silently, leaving dead infrastructure in the generic library.

The resolver contract is now: **a Resolver fills a `vector<Entry>` in children-before-parents order**. The library provides `DfsPostOrder` as scaffolding; `SelectionMap`, `CollectDependencies`, `TopologicalSort`, `ExtractionContext`, `ExtractView`, and `Extract` are deleted.

## Why DFS post-order instead of SelectionMap + Kahn

The SelectionMap contract is context-insensitive: one entry per e-class, chosen globally. `PlanResolver` requires context-sensitivity — the same e-class can appear under different provided-symbol sets and must pick a different alternative each time. A flat map cannot represent this; `PlanResolver` uses `ResolvedKey{eclass, provided, demanded}` as its key and emits multiple entries for the same e-class under different scopes.

Kahn's sort is a post-hoc fix for a flat map having no inherent ordering. DFS post-order produces the ordering directly as a consequence of the traversal. Any resolver that traverses the e-graph recursively — which all context-sensitive resolvers must — gets children-before-parents order for free by emitting after the recursive calls return. Kahn's sort adds two extra passes (`CollectDependencies` and `TopologicalSort`) and two extra scratch buffers (`InDegreeMap` and `FifoQueue`) to recover what the DFS already knew.

## Considered alternatives

**Keep SelectionMap + Kahn as utilities, add DfsPostOrder alongside.** Rejected: utilities with no production caller and no planned future use are maintenance surface. The tests that covered Kahn's sort were testing the sort's own correctness, not any behaviour the production pipeline exercises. Keeping them would imply a commitment to maintain a two-pass path that nothing uses.

**Add a second `ResolverV2` concept alongside the existing one.** Rejected: the only callers are inside `src/planner/`. There is no cross-library compatibility concern. A parallel concept creates a short-lived mess and doubles the surface a reader must understand.

## Consequences

- `extractor.hpp` public surface shrinks to: `CostResultType`, `FrontierMap`, `FrontierContext`, `ComputeFrontiers`, `DfsPostOrder`.
- `pareto_frontier.hpp` gains `PickBest(alts, pred)`, the predicate-filtered min-cost helper used by Pareto-frontier resolvers.
- `DefaultResolver` in `test_support/` is rewritten as a thin `DfsPostOrder` wrapper. It remains test-only; production resolvers supply their own key type and child-dispatch logic.
- Ten tests (`Extract_Dependencies.*`, `Extract_TopologicalSort.*`) are deleted — they tested Kahn-sort correctness, not extraction behaviour.
- `PlanResolver` loses its `Impl` struct; `operator()` becomes a direct `DfsPostOrder` call.
