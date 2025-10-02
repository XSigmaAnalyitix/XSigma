#pragma once

#include "common/macros.h"

namespace xsigma
{
class XSIGMA_API cpu_info
{
    XSIGMA_DELETE_CLASS(cpu_info);

public:
    static void info();

    static void cpuinfo_cach(
        std::ptrdiff_t& l1, std::ptrdiff_t& l2, std::ptrdiff_t& l3, std::ptrdiff_t& l3_count);
};
}  // namespace xsigma