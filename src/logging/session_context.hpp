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
#include <atomic>
#include <string>

namespace memgraph::logging {

// Per-session log state. Lives on whatever owns the session (today: the
// query::Interpreter). Mutated only by the thread currently executing the
// session's work; read by the wrapper under the RAII guard installed by
// Session::Execute().
struct SessionLogContext {
  // Per-session level override. Initialized at session create to the global
  // level. SET SESSION TRACE ON stores trace; OFF restores the global level.
  std::atomic<spdlog::level::level_enum> level{spdlog::level::off};

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

}  // namespace memgraph::logging
