# Planner Module

Core e-graph infrastructure for the V2 query planner based on equality saturation.

> **Status:** Core infrastructure complete. See `src/query/plan_v2/README.md` for integration status.

## Directory Structure

```
src/planner/
├── *.cppm            # C++20 modules (egraph, eclass, enode, etc.)
├── include/planner/
│   ├── pattern/      # Pattern matching (including VM executor)
│   ├── rewrite/      # Rewrite rules and saturation
│   └── extract/      # Cost-based extraction
├── src/              # Implementation files
├── test/             # Unit tests
├── fuzz/             # Fuzz tests
└── bench/            # Performance benchmarks
```

## Components

### E-Graph (C++20 Modules)
Core e-graph types as C++20 modules:
- `egraph.cppm` - E-graph with union-find and congruence closure
- `eclass.cppm` - E-class (equivalence class of e-nodes)
- `enode.cppm` - E-node (expression node)
- `eids.cppm` - Strong type IDs (EClassId, ENodeId)
- `concepts.cppm` - ENodeSymbol concept
- `union_find.cppm` - Union-find data structure
- `constants.cppm` - Shared constants
- `strong_type.cppm` - Strong type utilities
- Use: `import memgraph.planner.core.egraph;`

### Pattern Matching (`pattern/`)
- `pattern.hpp` - Pattern DSL for e-matching
  - `PatternVar` - Pattern variables (?x, ?y)
  - `Var`, `Wildcard` - Pattern elements
  - `Pattern::build()` - Fluent pattern construction
- `match_index.hpp` - E-matching engine
  - `MatcherIndex` - Finds all pattern matches in an e-graph
  - `EMatchContext` - Reusable matching buffers
- `match.hpp` - Match results
  - `Match` - Variable bindings from successful matches
  - `MatchArena` - Pool for match storage
- `match_storage.hpp` - Storage utilities for matches

#### VM Executor (`pattern/vm/`)
Bytecode-based pattern matching for performance:
- `compiler.hpp` - Compiles patterns to bytecode
- `compiled_matcher.hpp` - Compiled pattern representation
- `executor.hpp` - Executes bytecode against e-graph
- `instruction.hpp` - VM instruction set
- `state.hpp` - VM execution state with deduplication and parent iteration
- `tracer.hpp` - Execution tracing for debugging

### Rewrite System (`rewrite/`)
- `rule.hpp` - Rewrite rules
  - `RewriteRule` - Pattern(s) + apply function
  - `RewriteContext` - Combined reusable buffers
- `rule_set.hpp` - Rule collections
  - `RuleSet` - Immutable collection of rules
- `rule_context.hpp` - Rule application context
  - `RuleContext` - Safe e-graph modifications in apply functions
- `rewriter.hpp` - Equality saturation engine
  - `Rewriter` - Orchestrates rule application until saturation
  - `RewriteConfig` - Limits (iterations, e-nodes, timeout)
  - `RewriteResult` - Statistics and stop reason

### Extraction (`extract/`)
- `extractor.hpp` - Cost-based expression extraction
  - Public stages: `ComputeFrontiers`, `CollectDependencies`, `TopologicalSort`
  - `ExtractionContext` - reusable per-extraction storage (frontier map, selection, in-degree, order)
  - `Extract(ctx, ...)` - one-shot sugar over the four stages
- `pareto_frontier.hpp` - `ParetoFrontier<Alt, Dominance>`
  - Used by cost models that propagate demand sets (or any per-alt context)
  - Dominance pruning, merge, and `mutate_pruning_invariant_preserving` for
    in-place edits that preserve the Pareto invariant

## Testing

```bash
# Build planner library and tests
cmake --build build --preset conan-relwithdebinfo --target memgraph__unit__planner -j 15

# Run all unit tests
./build/src/planner/test/planner

# Run specific test suites
./build/src/planner/test/planner --gtest_filter="EGraph*"
./build/src/planner/test/planner --gtest_filter="MatcherIndex*"
./build/src/planner/test/planner --gtest_filter="Rewrite*"

# Run fuzzers
./build/src/planner/fuzz/fuzz_egraph       # E-graph correctness
./build/src/planner/fuzz/fuzz_pattern_vm   # Pattern VM correctness
```

## Benchmarking

```bash
# Build benchmarks
cmake --build build --preset conan-relwithdebinfo --target memgraph__benchmark__new_planner -j 15

# Run all benchmarks
./build/src/planner/bench/memgraph__benchmark__new_planner

# Run specific benchmarks
./build/src/planner/bench/memgraph__benchmark__new_planner --benchmark_filter="SimplePattern"
./build/src/planner/bench/memgraph__benchmark__new_planner --benchmark_filter="VMSimplePattern"
./build/src/planner/bench/memgraph__benchmark__new_planner --benchmark_filter="Rewrite"
./build/src/planner/bench/memgraph__benchmark__new_planner --benchmark_filter="Join"
```

## Usage Example

```cpp
import memgraph.planner.core.egraph;
#include "planner/pattern/pattern.hpp"
#include "planner/rewrite/rule.hpp"
#include "planner/rewrite/rule_set.hpp"
#include "planner/rewrite/rewriter.hpp"

using namespace memgraph::planner::core;

// Define symbol type (must satisfy ENodeSymbol concept) and analysis
enum class Op { Add, Neg, Var, Const };
struct NoAnalysis {};

// Create e-graph and add expressions
EGraph<Op, NoAnalysis> egraph;
auto x = egraph.emplace(Op::Var, 1);
auto neg_x = egraph.emplace(Op::Neg, {x.eclass_id});
auto neg_neg_x = egraph.emplace(Op::Neg, {neg_x.eclass_id});

// Create double-negation elimination rule: Neg(Neg(?x)) -> ?x
constexpr PatternVar kRoot{10};
auto pattern = Pattern<Op>::build(Op::Neg, {
    Pattern<Op>::build(Op::Neg, {Var{0}})
}, kRoot);

auto rule = RewriteRule<Op, NoAnalysis>::Builder{"double_negation"}
    .pattern(std::move(pattern))
    .apply([](RuleContext<Op, NoAnalysis> &ctx, Match const &match) {
        ctx.merge(match[kRoot], match[PatternVar{0}]);
    });

// Run equality saturation
auto rules = RuleSet<Op, NoAnalysis>::Build(std::move(rule));
Rewriter<Op, NoAnalysis> rewriter(egraph, rules);
auto result = rewriter.saturate(RewriteConfig::Default());
// result.saturated() == true when fixed point reached
```

## VM Optimisations

### Eclass-Level Join Hoisting

Multi-pattern rules like `F(?x), Mul(?r, ?y)` match an anchor pattern then
traverse parents to find joined patterns. When the joined pattern's parent
traversal depends only on the anchor's **eclass** (not the specific enode),
the traversal can be hoisted above the enode loop so it runs once per eclass
instead of once per enode. Both orderings produce the same matches; hoisting
eliminates the redundant repetitions.

The benefit scales with enodes-per-eclass — the redundancy factor.

### O(1) Parent Iteration

`ParentsIter` stores iterator pairs into the parent set (a
`boost::unordered_flat_set`) rather than an index that would require
`begin()` + `std::advance(index)` on each `NextParent` call. The parent set
is immutable during matching, so iterators stay valid. Each `NextParent`
becomes constant-time, eliminating quadratic cost on long parent traversals.

## Related Documentation

- `TODO.md` - Task tracking and known issues
- `src/query/plan_v2/README.md` - Query integration status
