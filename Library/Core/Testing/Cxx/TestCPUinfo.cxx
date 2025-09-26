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

#include <vector>

#include "util/cpu_info.h"
#include "xsigmaTest.h"

XSIGMATEST(Core, CPUinfo)
{
    START_LOG_TO_FILE_NAME(CPUinfo);

    xsigma::cpu_info::info();

    END_LOG_TO_FILE_NAME(CPUinfo);
    END_TEST();
}
