# Planner V2 Status Document

> **Status: EXPERIMENTAL - Not for production use**
>
> V2 is gated behind `--experimental-enabled=planner-v2`. V1 is default and unaffected.

---

## V1 Safety Confirmation

V1 planner is **safe and unchanged**:
- Experimental flag defaults to `NONE` (disabled)
- V2 code path requires explicit `--experimental-enabled=planner-v2`

---

## What Has Been Done

### Core E-Graph Infrastructure (`src/planner/`)
- **UnionFind** - Disjoint-set with path compression and union-by-rank
- **ENode** - Expression nodes with structural equality
- **EClass** - Equivalence classes grouping equivalent expressions
- **EGraph** - Core equality saturation engine with merge/rebuild
- **Extractor** - Cost-based plan extraction: `ComputeFrontiers` (bottom-up
  Pareto propagation) and `DfsPostOrder` (resolver scaffolding), with
  `FrontierContext` for reusable per-extraction storage.
- **ParetoFrontier** - Dominance-pruned alternative set used by cost models
  that propagate per-alt context (e.g. demand sets)

### Pattern Matching & Rewrite System (`src/planner/include/planner/`)
- **Pattern** (`pattern/pattern.hpp`)
  - Pattern variables (`PatternVar`), symbols, and wildcards
  - Fluent `Pattern::build()` DSL for constructing patterns
  - Explicit root bindings for accessing matched e-classes

- **E-Matching** (`pattern/match_index.hpp`, `pattern/match.hpp`)
  - `MatcherIndex` - Finds all pattern matches in an e-graph
  - Symbol index for O(1) candidate lookup
  - Incremental index rebuild for new e-classes
  - `Match` - Variable bindings from successful matches
  - `MatchArena` - Pool for match storage
  - VM bytecode executor (`pattern/vm/`) for compiled pattern matching

- **Rewrite Engine** (`rewrite/rule.hpp`, `rewrite/join.hpp`, `rewrite/rewriter.hpp`)
  - `RewriteRule` - Multi-pattern rules with hash-join
  - `RuleSet` - Immutable collection of rules
  - Greedy join ordering to minimize Cartesian products
  - `Rewriter` - Equality saturation loop with configurable limits
  - `RuleContext` - Safe wrapper for e-graph modifications
  - Buffer reuse via `JoinContext`, `RewriteContext`, `EMatchContext`

### Integration Layer (`src/query/plan_v2/`)
- **AST Converter** (`ast_converter.cpp`)
  - Converts Cypher AST to e-graph representation
  - Supports: `RETURN`, `WITH`, `Identifier`, `NamedExpression`, `PrimitiveLiteral`, `ParameterLookup`

- **EGraph Converter** (`egraph_converter.cpp`, `.hpp`)
  - `ConvertToLogicalOperator` drives extraction and produces `LogicalOperator`s
  - Per-Interpreter `QueryPlannerContext` owns the extraction buffers
    (`FrontierContext`, resolver scratch) so high-QPS workloads don't
    reallocate them per query
  - Creates `Produce` operators with proper bindings; preserves symbol names
    through position-based lookup

- **Cost Model** (`expression_cost.hpp`, `egraph_converter.cpp`)
  - `PlanCostModel` emits `CostFrontier = ParetoFrontier<Alternative, AlternativeDominance>`
    per eclass, where each `Alternative` carries `cost`, the `required` demand
    set (symbols this plan needs from its environment), and `is_alive` for Bind
    pairings.
  - `MapAlts` adjusts cost / e-node id in place for pass-through operators;
    `CombineAltsFn` does the cartesian product of child frontiers with
    dominance pruning.
  - `PlanResolver` selects, for each eclass, the min-cost alt whose `required`
    is compatible with the `provided` context, cascading into the chosen alt's
    children.

- **Rewrites** (`rewrites.cpp`)
  - `ApplyInlineRewrite` - Inlines `Identifier(?sym)` with `Bind(_, ?sym, ?expr)`
  - `ApplyAllRewrites` - Full saturation with all rules

- **Experimental Flag** (`src/flags/experimental.cpp`)
  - `PLANNER_V2` flag gates V2 activation
  - Plan cache disabled when V2 active

### Architecture
- **Trait-based storage** (`symbol_make_traits.hpp`)
  - Each symbol type defines its own storage via `symbol_make_traits<S>`
  - Combined storage using inheritance pattern
  - Clean public API with named methods (`MakeOnce`, `MakeSymbol`, etc.)

### Testing & Benchmarks
- **Unit tests** (`tests/unit/query_plan_v2_*.cpp`)
  - Pipeline tests: AST → EGraph → Rewrites → LogicalOperator
  - Rewrite tests: Pattern matching, multi-pattern rules
- **Planner benchmarks** (`src/planner/bench/`)
  - E-matching performance benchmarks
  - Rewrite system benchmarks

---

## What Is Missing (Gaps)

### Implementation Gaps

| Category | Missing Items |
|----------|---------------|
| **Query Types** | MATCH, CREATE, MERGE, DELETE, SET, REMOVE, CALL, LOAD CSV, FOREACH, UNWIND |
| **Expressions** | WHERE, EXISTS, CASE/WHEN, pattern comprehension, list comprehension |
| **Operators** | Comparison (=, <>, <, >), Boolean (AND, OR, NOT, XOR), Arithmetic (+, -, *, /, %) |
| **Functions** | All function calls, aggregations (COUNT, SUM, AVG, etc.) |
| **Types** | ListLiteral, MapLiteral, PropertyLookup, LabelsTest, subscript, slicing |
| **Pattern** | NodeAtom, EdgeAtom, Pattern matching |

All missing items have `MG_ASSERT(false, "not implemented yet")` stubs in `ast_converter.cpp`.

### Rewrite Rules Needed

#### Expression Simplification
| Rule | Example | Optimization |
|------|---------|--------------|
| Constant folding | `RETURN 1+1+1+1` | → `RETURN 4` |
| Independent expression hoisting | `UNWIND range(1,100) AS i RETURN 1+1+1+1` | Hoist `1+1+1+1` outside loop |
| Count simplification | `count(coalesce(x.thing, 1))` | → `count(*)` when expression is non-null |
| Range coalescing | `n.p < 10 AND n.p < 5` | → `n.p < 5` |
| Type conflict detection | `n.prop < 42 AND n.prop = true` | Detect impossible type comparisons |

#### Dead Code Elimination
| Rule | Example | Optimization |
|------|---------|--------------|
| Unused binding removal | `WITH 10 AS a, 5 AS b RETURN a` | Don't compute `b` |
| Redundant DISTINCT | `... UNION ...` with `DISTINCT` in branches | Remove inner DISTINCT when outer exists |
| Semantic distinctness | `MATCH (n) RETURN DISTINCT n` | Remove DISTINCT (nodes already distinct) |
| Duplicate literal removal | `UNWIND [1,1,1] AS val RETURN DISTINCT val` | Pre-process to `[1]` |

#### Predicate Optimization
| Rule | Example | Optimization |
|------|---------|--------------|
| Contradiction detection | `n.age > 10 AND n.age < 10` | Detect unsatisfiable predicates |
| Expression cost reordering | `expensiveCheck(n) AND n.prop = 42` | → `n.prop = 42 AND expensiveCheck(n)` |
| Redundant label filter | `Filter (n :L1:L2)` after `ScanAllByLabel (n :L1)` | Remove `:L1` from filter |

#### Operator Optimization
| Rule | Example | Optimization |
|------|---------|--------------|
| Produce barrier removal | `MATCH (n) WITH n WHERE n:L1 RETURN n` | Allow ScanAllByLabel through WITH |
| CALL subquery flattening | `CALL { WITH a MATCH (a)-->(b) RETURN b }` | Inline trivial subqueries |
| WITH DISTINCT to aggregation | `MATCH(n) WITH DISTINCT n RETURN count(n)` | → `MATCH(n) RETURN count(DISTINCT n)` |
| DISTINCT pushdown | `UNWIND [1,1,2,2] AS v1 UNWIND [...] AS v2 RETURN DISTINCT *` | Push to smaller tuples |
| SKIP elimination | `... SKIP 100 LIMIT 10` with non-deterministic order | Remove SKIP when cardinality > skip+limit |

#### Index & Scan Optimization
| Rule | Example | Optimization |
|------|---------|--------------|
| IN with L+P index | `WHERE n.prop IN [...]` | Use ScanAllByLabelPropertyValue |
| Index for ORDER BY | `MATCH (n:L) RETURN n ORDER BY n.prop` | Use L+P index to avoid sort |
| Heuristic cardinality | `MATCH (n1)--(n2 {k2:42}) WHERE n1.k1 < 100` | Expand from more selective side |

#### Pattern & Join Optimization
| Rule | Example | Optimization |
|------|---------|--------------|
| Common node expansion | `MATCH (b)<-[r1]-(a), (a)-[r2]->(c)` | Start from common node `a` |
| Edge uniqueness simplification | `WHERE e2 IN [e1,e3]` with `e2 != e3` | → `WHERE e2 = e1` |
| Join reordering | Multiple MATCH clauses | Order by estimated cardinality |

#### Schema-Aware Optimization
| Rule | Example | Optimization |
|------|---------|--------------|
| Unique constraint usage | `count(DISTINCT n.property)` with UNIQUE constraint | → `count(n.property)` |
| Existence constraint usage | `ASSERT EXISTS (n.property)` | Skip null checks |

#### Loop Optimization
| Rule | Example | Optimization |
|------|---------|--------------|
| UNWIND unrolling | `UNWIND RANGE(1, 2) AS i RETURN i` | → `RETURN 1 UNION ALL RETURN 2` |

### Infrastructure Gaps

| Item | Status |
|------|--------|
| Cost model | Pareto-frontier with demand sets and alive/dead Bind handling; no cardinality input yet |
| Cardinality estimation | Not started |
| Migration strategy | Not documented |

---

## Known Issues

### Medium (Technical Debt)

1. **Lost metadata** - `user_declared_`, `type_`, and `token_position_` not fully preserved during Symbol conversion
2. **No cardinality** - cost model uses structural cost only; needs index/statistics input to pick between alternative scan strategies once MATCH lands

### Low (Cleanup)

3. SymbolTable contract for `ConvertToLogicalOperator` is provisional — see `src/planner/TODO.md`

---

## Usage

```bash
# Run with V2 planner enabled
./build/memgraph --experimental-enabled=planner-v2

# Run pipeline tests
cmake --build build --preset conan-relwithdebinfo --target memgraph__unit__query_plan_v2_pipeline -j 15
./build/tests/unit/query_plan_v2_pipeline

# Run rewrite tests
cmake --build build --preset conan-relwithdebinfo --target memgraph__unit__query_plan_v2_rewrites -j 15
./build/tests/unit/query_plan_v2_rewrites
```
