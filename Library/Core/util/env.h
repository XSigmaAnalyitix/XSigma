#pragma once

#include <optional>
#include <string>

#include "common/export.h"

namespace xsigma::utils
{

// Set an environment variable.
XSIGMA_API void set_env(const char* name, const char* value, bool overwrite = true);

// Checks an environment variable is set.
XSIGMA_API bool has_env(const char* name) noexcept;

// Reads an environment variable and returns
// - std::optional<true>,              if set equal to "1"
// - std::optional<false>,             if set equal to "0"
// - nullopt,   otherwise
//
// NB:
// Issues a warning if the value of the environment variable is not 0 or 1.
XSIGMA_API std::optional<bool> check_env(const char* name);

// Reads the value of an environment variable if it is set.
// However, check_env should be used if the value is assumed to be a flag.
XSIGMA_API std::optional<std::string> get_env(const char* name) noexcept;

}  // namespace xsigma::utils
