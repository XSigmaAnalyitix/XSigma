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

// A simple test to test XSIGMA::fmt is working as expected.

#include <vector>

#include "xsigmaTest.h"

// clang-format off
#include <fmt/core.h>
#include <fmt/ranges.h>// IWYU pragma: keep
// clang-format on

XSIGMATEST(Core, FMT)
{
    XSIGMA_UNUSED int    arg     = 0;
    XSIGMA_UNUSED char** arg_str = nullptr;

    fmt::print("Hello, {}!\n", "World");

    std::vector<int> v = {1, 2, 3};
    fmt::print("vector: {}\n", v);
    END_TEST();
}
