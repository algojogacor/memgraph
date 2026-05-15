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
#include <spdlog/spdlog.h>
#include <utility>

// Wrapper API mirroring spdlog's overload set. Behaves identically to a
// direct spdlog::* call today; a follow-up commit adds per-session level
// gating and prefix tagging.
namespace memgraph::logging {

template <typename... Args>
void Trace(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::default_logger_raw()->trace(fmt, std::forward<Args>(args)...);
}

template <typename T>
void Trace(const T &msg) {
  spdlog::default_logger_raw()->trace(msg);
}

template <typename... Args>
void Debug(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::default_logger_raw()->debug(fmt, std::forward<Args>(args)...);
}

template <typename T>
void Debug(const T &msg) {
  spdlog::default_logger_raw()->debug(msg);
}

template <typename... Args>
void Info(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::default_logger_raw()->info(fmt, std::forward<Args>(args)...);
}

template <typename T>
void Info(const T &msg) {
  spdlog::default_logger_raw()->info(msg);
}

template <typename... Args>
void Warn(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::default_logger_raw()->warn(fmt, std::forward<Args>(args)...);
}

template <typename T>
void Warn(const T &msg) {
  spdlog::default_logger_raw()->warn(msg);
}

template <typename... Args>
void Error(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::default_logger_raw()->error(fmt, std::forward<Args>(args)...);
}

template <typename T>
void Error(const T &msg) {
  spdlog::default_logger_raw()->error(msg);
}

template <typename... Args>
void Critical(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  spdlog::default_logger_raw()->critical(fmt, std::forward<Args>(args)...);
}

template <typename T>
void Critical(const T &msg) {
  spdlog::default_logger_raw()->critical(msg);
}

}  // namespace memgraph::logging
