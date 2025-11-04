#pragma once

#include <string>

#include "common/export.h"

namespace xsigma
{

XSIGMA_API void set_thread_name(std::string name);

XSIGMA_API std::string get_thread_name();

}  // namespace xsigma
