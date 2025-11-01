/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <gtest/gtest.h>  // for Test, TestInfo (ptr only)

#include <vector>

#include "logging/logger.h"  // for END_LOG_TO_FILE_NAME, START_LOG_TO_FILE_NAME
#include "util/cpu_info.h"   // for cpu_info
#include "xsigmaTest.h"      // for END_TEST, XSIGMATEST

XSIGMATEST(CPUinfo, CPUinfo)
{
    START_LOG_TO_FILE_NAME(CPUinfo);

    xsigma::cpu_info::info();

    END_LOG_TO_FILE_NAME(CPUinfo);
    END_TEST();
}

// ============================================================================
// CPU Cache Information Tests
// ============================================================================

XSIGMATEST(CPUinfo, cpuinfo_cache_retrieval)
{
    // Test cpuinfo_cach function to retrieve cache sizes
    std::ptrdiff_t l1       = 0;
    std::ptrdiff_t l2       = 0;
    std::ptrdiff_t l3       = 0;
    std::ptrdiff_t l3_count = 0;

    // Call the function to retrieve cache information
    xsigma::cpu_info::cpuinfo_cach(l1, l2, l3, l3_count);

    // Verify that cache sizes are non-negative
    EXPECT_GE(l1, 0);
    EXPECT_GE(l2, 0);
    EXPECT_GE(l3, 0);
    EXPECT_GE(l3_count, 0);

    // On most modern systems, L1 cache should be present
    // L1 cache is typically 32KB or 64KB per core
    if (l1 > 0)
    {
        EXPECT_GT(l1, 0);
        EXPECT_LT(l1, 1024 * 1024);  // L1 should be less than 1MB
    }

    // L2 cache is typically 256KB to 512KB per core
    if (l2 > 0)
    {
        EXPECT_GT(l2, 0);
        EXPECT_LT(l2, 10 * 1024 * 1024);  // L2 should be less than 10MB
    }

    // L3 cache is typically 8MB to 32MB
    if (l3 > 0)
    {
        EXPECT_GT(l3, 0);
        EXPECT_LT(l3, 100 * 1024 * 1024);  // L3 should be less than 100MB
    }

    END_TEST();
}

XSIGMATEST(CPUinfo, cpuinfo_cache_consistency)
{
    // Test that multiple calls return consistent results
    std::ptrdiff_t l1_first       = 0;
    std::ptrdiff_t l2_first       = 0;
    std::ptrdiff_t l3_first       = 0;
    std::ptrdiff_t l3_count_first = 0;

    xsigma::cpu_info::cpuinfo_cach(l1_first, l2_first, l3_first, l3_count_first);

    std::ptrdiff_t l1_second       = 0;
    std::ptrdiff_t l2_second       = 0;
    std::ptrdiff_t l3_second       = 0;
    std::ptrdiff_t l3_count_second = 0;

    xsigma::cpu_info::cpuinfo_cach(l1_second, l2_second, l3_second, l3_count_second);

    // Results should be consistent across multiple calls
    EXPECT_EQ(l1_first, l1_second);
    EXPECT_EQ(l2_first, l2_second);
    EXPECT_EQ(l3_first, l3_second);
    EXPECT_EQ(l3_count_first, l3_count_second);

    END_TEST();
}

XSIGMATEST(CPUinfo, cpuinfo_cache_hierarchy)
{
    // Test cache hierarchy: L1 < L2 < L3
    std::ptrdiff_t l1       = 0;
    std::ptrdiff_t l2       = 0;
    std::ptrdiff_t l3       = 0;
    std::ptrdiff_t l3_count = 0;

    xsigma::cpu_info::cpuinfo_cach(l1, l2, l3, l3_count);

    // If all caches are present, verify hierarchy
    if (l1 > 0 && l2 > 0)
    {
        EXPECT_LE(l1, l2);  // L1 should be <= L2
    }

    if (l2 > 0 && l3 > 0)
    {
        EXPECT_LE(l2, l3);  // L2 should be <= L3
    }

    if (l1 > 0 && l3 > 0)
    {
        EXPECT_LE(l1, l3);  // L1 should be <= L3
    }

    END_TEST();
}
