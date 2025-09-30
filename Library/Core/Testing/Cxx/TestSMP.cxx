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

#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "common/configure.h"  // IWYU pragma: keep
#include "common/pointer.h"
#include "logging/logger.h"
#include "smp/Common/thread_local_api.h"
#include "smp/Common/tools_api.h"
#include "smp/tools.h"
#include "smp/xsigma_thread_local.h"
#include "xsigmaTest.h"

XSIGMATEST(Core, SMP)
{
    END_TEST();
}
