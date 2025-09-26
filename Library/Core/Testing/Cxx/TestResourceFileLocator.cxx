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

#include <cstdio>  // for printf
#include <string>  // for string, allocator

#include "util/logger.h"                 // for logger,  logger_verbosity_enum::VERBOSITY_INFO
#include "util/resource_file_locator.h"  // for LIBRARY_PATH_FOR_SYMBOL, res...
#include "util/version.h"                // for GetXSIGMAVersion
#include "xsigmaTest.h"                  // for Test
#include "xsigma_build.h"

XSIGMATEST(Core, ResourceFileLocator)
{
    printf("xsigma version %s \n", GetXSIGMAVersion());

#if defined(XSIGMA_BUILD_SHARED_LIBS)
    const std::string xsigmalib = LIBRARY_PATH_FOR_SYMBOL(GetXSIGMAVersion);

    EXPECT_FALSE(xsigmalib.empty());

    const std::string xsigmadir = xsigma::resource_file_locator::GetFilenamePath(xsigmalib);

    const std::string path = xsigma::resource_file_locator::Locate(
        xsigmadir, "Testing/Temporary", xsigma::logger_verbosity_enum::VERBOSITY_INFO);

    EXPECT_FALSE(path.empty());

    xsigma::resource_file_locator::library_path_for_symbol_unix("cob");
#endif
    END_TEST();
}
