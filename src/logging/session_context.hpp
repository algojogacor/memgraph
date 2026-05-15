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

#include <fmt/format.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <string>

namespace memgraph::logging {

// Per-session log state. Lives on whatever owns the session (today: the
// query::Interpreter). Mutated only by the thread currently executing the
// session's work; read by the wrapper under the RAII guard installed by
// Session::Execute().
struct SessionLogContext {
  // Per-session level override. Defaults to off, meaning "follow the global
  // level". When SET SESSION LOG LEVEL TO '<level>' is run, this is set and
  // takes precedence over the global level for any log call on a thread that
  // has installed this context via ScopedSessionLog.
  std::atomic<spdlog::level::level_enum> level{spdlog::level::off};

  // Independent of level. Controls whether structured query-trace events
  // (parse/plan/exec/commit markers in the interpreter) are emitted at all.
  // When true, EmitSessionTraceEvent() emits unconditionally regardless of
  // the level gate above. SET SESSION TRACE ON toggles this; the old
  // QueryLogger semantics — trace as a debugging stream, not a level — are
  // preserved.
  std::atomic<bool> trace_enabled{false};

  std::string session_uuid;
  std::string user;
  std::string tx_id;

  // Render the prefix that gets prepended to log lines made under this
  // context: "[session=<uuid>] [user=<u>] [tx=<id>]" — empty fields skipped.
  std::string Prefix() const {
    std::string out;
    if (!session_uuid.empty()) {
      out += fmt::format("[session={}]", session_uuid);
    }
    if (!user.empty()) {
      if (!out.empty()) out += ' ';
      out += fmt::format("[user={}]", user);
    }
    if (!tx_id.empty()) {
      if (!out.empty()) out += ' ';
      out += fmt::format("[tx={}]", tx_id);
    }
    return out;
  }
};

namespace detail {
// Inlined thread_local so every translation unit that includes this header
// shares the same storage without needing a link-time symbol from mg-logging.
inline thread_local SessionLogContext *current_session_log = nullptr;
}  // namespace detail

// RAII guard. Construct at the top of Session::Execute() with a pointer to
// the session's SessionLogContext (or nullptr if there isn't one yet, e.g.
// pre-auth). Destructor restores the prior context — supports nested guards
// for correctness, although in normal flow there is only one level.
class ScopedSessionLog {
 public:
  explicit ScopedSessionLog(SessionLogContext *ctx) noexcept : prev_(detail::current_session_log) {
    detail::current_session_log = ctx;
  }

  ~ScopedSessionLog() noexcept { detail::current_session_log = prev_; }

  ScopedSessionLog(const ScopedSessionLog &) = delete;
  ScopedSessionLog &operator=(const ScopedSessionLog &) = delete;
  ScopedSessionLog(ScopedSessionLog &&) = delete;
  ScopedSessionLog &operator=(ScopedSessionLog &&) = delete;

  // Returns the current thread's session context, or nullptr if none active.
  static SessionLogContext *Current() noexcept { return detail::current_session_log; }

 private:
  SessionLogContext *prev_;
};

// Emit a structured query-trace event. Distinct from the level-gated wrapper:
// these events are a debugging tool (SET SESSION TRACE ON), not regular log
// output, so they bypass the per-session level filter and always emit when
// the session has trace enabled. Spdlog's logger is pinned to trace so the
// underlying write is unconditional from spdlog's perspective.
inline void EmitSessionTraceEvent(std::string_view msg) {
  auto *ctx = ScopedSessionLog::Current();
  if (ctx == nullptr || !ctx->trace_enabled.load(std::memory_order_relaxed)) return;
  auto prefix = ctx->Prefix();
  if (prefix.empty()) {
    spdlog::default_logger_raw()->log(spdlog::level::trace, msg);
  } else {
    spdlog::default_logger_raw()->log(spdlog::level::trace, "{} {}", prefix, msg);
  }
}

}  // namespace memgraph::logging
