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

#include <cstdint>
#include <string_view>
#include <vector>

#include "util/multi_process_stream.h"
#include "xsigmaTest.h"

namespace
{
template <typename T>
void pop_push(xsigma::multi_process_stream* archive)
{
    unsigned int n = 5;

    T x[5];
    if (!archive->Empty())
        archive->Reset();

    archive->Push(&x[0], n);

    XSIGMA_UNUSED unsigned int size     = archive->Size();
    XSIGMA_UNUSED unsigned int row_size = archive->RawSize();

    auto data = archive->GetRawData();

    {
        auto* ret = &x[0];
        archive->Pop(ret, n);
    }
    {
        archive->Push(&x[0], n);
        T* ret = nullptr;
        archive->Pop(ret, n);
        delete[] ret;
    }

    data[0] = 1 - data[0];
    archive->SetRawData(data);

    EXPECT_EQ(archive->endianness(), archive->endianness());
}
}  // namespace

XSIGMATEST(Core, MultiProcessStream)
{
    START_LOG_TO_FILE_NAME(MultiProcessStream);

    xsigma::multi_process_stream* archive = new xsigma::multi_process_stream();

    {
        if (!archive->Empty())
            archive->Reset();
        char v = 1;
        (*archive) << v;
        (*archive) >> v;
    }
    {
        if (!archive->Empty())
            archive->Reset();
        unsigned char v = 1;
        (*archive) << v;
        (*archive) >> v;
    }
    {
        xsigma::multi_process_stream s;
        s = *archive;
        if (!s.Empty())
            s.Reset();
        const char* v = "mem";
        s << v;
    }
    {
        xsigma::multi_process_stream s = *archive;
        if (!archive->Empty())
            archive->Reset();
        std::string_view v("abcdef");
        (*archive) << v;
        (*archive) >> v;
    }

    pop_push<double>(archive);
    pop_push<float>(archive);
    pop_push<char>(archive);
    pop_push<int>(archive);
    pop_push<unsigned int>(archive);
    pop_push<unsigned char>(archive);
    pop_push<int64_t>(archive);
    pop_push<size_t>(archive);

    END_LOG_TO_FILE_NAME(MultiProcessStream);

    delete archive;

    END_TEST();
}
