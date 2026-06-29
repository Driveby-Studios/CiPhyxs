//==================================================================================================
/// @file  Assert.hpp
/// @brief  Configurable assertion macro for CiPhyxs production error handling.
///
/// Provides CIPHYXS_ASSERT(cond, msg) which replaces raw assert() calls throughout the
/// engine.  The default implementation prints a diagnostic message to stderr and calls
/// std::abort().  Users can override the macro before including CiPhyxs.hpp to install
/// a custom handler (e.g., throw an exception, log to a file, or trigger a breakpoint).
///
/// ## Usage
///
/// ```cpp
///   CIPHYXS_ASSERT(ptr != nullptr, "Unexpected null pointer in solver hot loop");
///   CIPHYXS_ASSERT(index < size(), "Body index out of range");
/// ```
///
/// ## Customisation
///
/// Define CIPHYXS_ASSERT before including CiPhyxs.hpp to replace the default.
/// If CIPHYXS_ASSERT is already defined when this header is included, the existing
/// definition is respected and no default is provided.
///
/// Example custom override:
/// ```cpp
///   #define CIPHYXS_ASSERT(cond, msg) do { if (!(cond)) { throw std::runtime_error(msg); } } while(false)
///   #include <ciphyxs/CiPhyxs.hpp>
/// ```
//==================================================================================================
#pragma once

#include <cstdio>
#include <cstdlib>

namespace ciphyxs {

/// @brief  Default assertion failure handler.
///
/// Prints a detailed diagnostic to stderr and terminates the process.
/// Users can override CIPHYXS_ASSERT to customise this behaviour.
inline void defaultAssertHandler(const char* condition, const char* message,
                                  const char* file, int line) noexcept {
    std::fprintf(stderr,
        "[CiPhyxs] ASSERTION FAILURE\n"
        "  Condition: %s\n"
        "  Message:   %s\n"
        "  File:      %s\n"
        "  Line:      %d\n",
        condition, message ? message : "(no message)",
        file ? file : "(unknown)", line);
    std::fflush(stderr);
    std::abort();
}

} // namespace ciphyxs

// ─── Default assertion macro ────────────────────────────────────────────────────────────────────
// If the user has not defined CIPHYXS_ASSERT, define it to call the default handler.
// This allows users to override the macro before including any CiPhyxs header.

#ifndef CIPHYXS_ASSERT
#define CIPHYXS_ASSERT(cond, msg)                                                          \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            ::ciphyxs::defaultAssertHandler(                                               \
                #cond, msg, __FILE__, static_cast<int>(__LINE__));                         \
        }                                                                                  \
    } while (false)
#endif
