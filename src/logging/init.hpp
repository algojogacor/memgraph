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

#include <spdlog/common.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "spdlog/sinks/sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace memgraph::logging {

namespace detail {
// Inlined so the wrapper template can read/write without a link-time symbol
// from mg-logging.
inline std::atomic<spdlog::level::level_enum> g_global_level{spdlog::level::warn};
}  // namespace detail

inline std::shared_ptr<spdlog::sinks::sink> &stderr_sink() {
  static std::shared_ptr<spdlog::sinks::sink> sink = std::make_shared<spdlog::sinks::stderr_color_sink_st>();
  return sink;
}

const std::string &GetAllowedLogLevels();

constexpr const char *GetLogLevelHelpString() {
  return "Minimum log level. Allowed values: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL";
}

bool ValidLogLevel(std::string_view value);
std::optional<spdlog::level::level_enum> LogLevelToEnum(std::string_view value);

// Global log level. Wrapper consults this (or the per-session override if a
// session log context is active on the current thread) before deciding to log.
// Spdlog's logger itself is pinned to trace so it never short-circuits the
// wrapper.
inline void SetGlobalLevel(spdlog::level::level_enum lvl) {
  detail::g_global_level.store(lvl, std::memory_order_relaxed);
}

inline spdlog::level::level_enum GetGlobalLevel() { return detail::g_global_level.load(std::memory_order_relaxed); }

void InitializeLogger();
void AddLoggerSink(spdlog::sink_ptr new_sink);
// Sets stderr log level to off
void TurnOffStdErr();
// Sets logging level to the global logging level
void TurnOnStdErr();
void CleanLogsDir();

}  // namespace memgraph::logging
