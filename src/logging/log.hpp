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

#include "logging/init.hpp"
#include "logging/session_context.hpp"

namespace memgraph::logging {

namespace detail {

inline spdlog::level::level_enum EffectiveLevel() noexcept {
  if (auto *ctx = ScopedSessionLog::Current()) {
    return ctx->level.load(std::memory_order_relaxed);
  }
  return GetGlobalLevel();
}

template <typename... Args>
inline void LogWithCtx(spdlog::level::level_enum lvl, spdlog::format_string_t<Args...> fmt, Args &&...args) {
  if (lvl < EffectiveLevel()) return;
  auto *ctx = ScopedSessionLog::Current();
  if (ctx != nullptr) {
    auto prefix = ctx->Prefix();
    if (!prefix.empty()) {
      spdlog::default_logger_raw()->log(lvl, "{} {}", prefix, fmt::format(fmt, std::forward<Args>(args)...));
      return;
    }
  }
  spdlog::default_logger_raw()->log(lvl, fmt, std::forward<Args>(args)...);
}

template <typename T>
inline void LogMsgWithCtx(spdlog::level::level_enum lvl, const T &msg) {
  if (lvl < EffectiveLevel()) return;
  auto *ctx = ScopedSessionLog::Current();
  if (ctx != nullptr) {
    auto prefix = ctx->Prefix();
    if (!prefix.empty()) {
      spdlog::default_logger_raw()->log(lvl, "{} {}", prefix, msg);
      return;
    }
  }
  spdlog::default_logger_raw()->log(lvl, msg);
}

}  // namespace detail

template <typename... Args>
void Trace(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  detail::LogWithCtx(spdlog::level::trace, fmt, std::forward<Args>(args)...);
}

template <typename T>
void Trace(const T &msg) {
  detail::LogMsgWithCtx(spdlog::level::trace, msg);
}

template <typename... Args>
void Debug(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  detail::LogWithCtx(spdlog::level::debug, fmt, std::forward<Args>(args)...);
}

template <typename T>
void Debug(const T &msg) {
  detail::LogMsgWithCtx(spdlog::level::debug, msg);
}

template <typename... Args>
void Info(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  detail::LogWithCtx(spdlog::level::info, fmt, std::forward<Args>(args)...);
}

template <typename T>
void Info(const T &msg) {
  detail::LogMsgWithCtx(spdlog::level::info, msg);
}

template <typename... Args>
void Warn(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  detail::LogWithCtx(spdlog::level::warn, fmt, std::forward<Args>(args)...);
}

template <typename T>
void Warn(const T &msg) {
  detail::LogMsgWithCtx(spdlog::level::warn, msg);
}

template <typename... Args>
void Error(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  detail::LogWithCtx(spdlog::level::err, fmt, std::forward<Args>(args)...);
}

template <typename T>
void Error(const T &msg) {
  detail::LogMsgWithCtx(spdlog::level::err, msg);
}

template <typename... Args>
void Critical(spdlog::format_string_t<Args...> fmt, Args &&...args) {
  detail::LogWithCtx(spdlog::level::critical, fmt, std::forward<Args>(args)...);
}

template <typename T>
void Critical(const T &msg) {
  detail::LogMsgWithCtx(spdlog::level::critical, msg);
}

}  // namespace memgraph::logging
