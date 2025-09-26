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

#include <iostream>  // for operator<<, ifstream, ostream
#include <string>    // for allocator

#include "util/resource_file_locator.h"
#include "xsigmaTest.h"  // for Test, SuiteApiResolver, TestInfo...

using namespace std;

void xsigmaSystemInformationPrintFile(const char* name, ostream& os)
{
    os << "================================================================\n";
    if (xsigma::resource_file_locator::file_exist(name) != 0)
    {
        os << "The file \"" << name << "\" does not exist.\n";
        return;
    }

#ifdef _WIN32
    fstream fin(name, ios::in | ios::binary);
#else
    fstream fin(name, ios::in);
#endif

    if (fin)
    {
        os << "Contents of \"" << name << "\":\n";
        os << "----------------------------------------------------------------\n";
        const int bufferSize = 4096;
        char      buffer[bufferSize];  // NOLINT
        // This copy loop is very sensitive on certain platforms with
        // slightly broken stream libraries (like HPUX).  Normally, it is
        // incorrect to not check the error condition on the fin.read()
        // before using the data, but the fin.gcount() will be zero if an
        // error occurred.  Therefore, the loop should be safe everywhere.
        while (fin)
        {
            fin.read(buffer, bufferSize);
            if (fin.gcount())  // NOLINT
            {
                os.write(buffer, fin.gcount());
            }
        }
        os.flush();
    }
    else
    {
        os << "Error opening \"" << name << "\" for reading.\n";
    }
}

XSIGMATEST(Core, SystemInformation)
{
#ifndef XSIGMA_GOOGLE_TEST
    if (argc != 2)
    {
        cerr << "Usage: TestSystemInformation <top-of-build-tree>\n";
        return 1;
    }
    std::string build_dir = argv[1];
    build_dir += "/";

    const char* files[] = {
        "CMakeCache.txt",
        "CMakeFiles/CMakeError.log",
        "configure.h",
        "XSIGMAConfig.cmake",
        "Testing/Temporary/ConfigSummary.txt",
        nullptr};

    std::cout << "CTEST_FULL_OUTPUT (Avoid ctest truncation of output)" << std::endl;

    for (const char** f = files; *f != nullptr; ++f)
    {
        std::string fname = build_dir + *f;
        xsigmaSystemInformationPrintFile(fname.c_str(), cout);
    }

    END_TEST();
#endif
}
