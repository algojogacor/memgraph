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

namespace memgraph::query::plan::v2 {

/// Default per-call cardinality when nothing better is known.
///
/// Placeholder for storage-stats-backed estimates; deliberately not 1.0
/// because that would silently make every unknown look free and every
/// plan-shape decision tip toward "evaluate per row".  1000.0 is large
/// enough that an unknown row pipe out-costs a known scalar.
inline constexpr double kDefaultRowEstimate = 1000.0;

/// Per-output-row overhead for the UNWIND operator.  Paid once per row the
/// Unwind produces, on top of the list-expression evaluation cost.
/// Structural placeholder until measured data justifies a value -
/// deliberately small (1.0) so it doesn't dominate other per-row terms in
/// the cost model today.
inline constexpr double kUnwindPerRowOverhead = 1.0;

namespace leaf {

/// Cost of an Once leaf alternative.  Structural placeholder, 1.0 until
/// measured data justifies a value.
inline constexpr double kOnce = 1.0;

/// Cost of a Symbol leaf alternative.  Structural placeholder, 1.0 until
/// measured data justifies a value.
inline constexpr double kSymbol = 1.0;

/// Cost of a Literal leaf alternative.  Structural placeholder, 1.0 until
/// measured data justifies a value.
inline constexpr double kLiteral = 1.0;

/// Cost of a ParamLookup leaf alternative.  Structural placeholder, 1.0
/// until measured data justifies a value.
inline constexpr double kParamLookup = 1.0;

}  // namespace leaf

}  // namespace memgraph::query::plan::v2
