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

#include <string>

#include "common/macros.h"
#include "util/exception.h"
#include "xsigmaTest.h"

using namespace xsigma;
// int  max_iter = 10;

XSIGMATEST(Core, Exception)
{
#ifndef NDEBUG
    ASSERT_ANY_THROW({ XSIGMA_CHECK_DEBUG(false, "XSIGMA_CHECK_DEBUG: Should throw"); });
#endif

    ASSERT_ANY_THROW({ XSIGMA_CHECK(false, "XSIGMA_CHECK: Should throw"); });

#ifdef XSIGMA_ENABLE_GTEST

    //fixme: ASSERT_ANY_THROW({ XSIGMA_THROW("XSIGMA_THROW: should throw"); });

    auto e = xsigma::Error({__func__, __FILE__, static_cast<int>(__LINE__)}, "error 1");
    ASSERT_ANY_THROW({ XSIGMA_RETHROW(e, "re error 1"); });
#endif

    XSIGMA_WARN_ONCE("this is a warning once false!");

    xsigma::Warning::set_warnAlways(true);
    XSIGMA_WARN_ONCE("this is a warning once true!");

    END_TEST();
}
