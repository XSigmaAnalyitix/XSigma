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

// SPDX-FileCopyrightText: Copyright (c) T. Bellaj
// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>  // for Test, EXPECT_EQ, Message, TestPartResult, TestInfo (ptr ...

#include <atomic>   // for atomic, _Atomic_integral, _Atomic_integral_facade
#include <cstdint>  // for int32_t, int64_t
#include <memory>   // for unique_ptr

#include "common/pointer.h"      // for make_ptr_unique_mutable
#include "smp/multi_threader.h"  // for multi_threader
#include "xsigmaTest.h"          // for END_TEST, XSIGMATEST
#include "xsigma_threads.h"      // for __XSIGMA_THREAD_RETURN_TYPE__, __XSIGMA_THREAD_RETURN_VA...

using namespace std;

static std::atomic<int32_t> TotalAtomic(0);
static std::atomic<int64_t> TotalAtomic64(0);
static const int            Target = 1000000;
static int                  Values32[Target + 1];
static int                  Values64[Target + 1];
static int                  NumThreads = 5;

// uncomment the following line if you want to see
// the difference between using atomics and not
//#define SHOW_DIFFERENCE
#ifdef SHOW_DIFFERENCE
static int             Total   = 0;
static xsigmaTypeInt64 Total64 = 0;
#endif

__XSIGMA_THREAD_RETURN_TYPE__ MyFunction(void*)
{
    for (int i = 0; i < Target / NumThreads; i++)
    {
#ifdef SHOW_DIFFERENCE
        Total++;
        Total64++;
#endif

        int idx       = ++TotalAtomic;
        Values32[idx] = 1;

        idx           = ++TotalAtomic64;
        Values64[idx] = 1;
    }

    return __XSIGMA_THREAD_RETURN_VALUE__;
}

__XSIGMA_THREAD_RETURN_TYPE__ MyFunction2(void*)
{
    for (int i = 0; i < Target / NumThreads; i++)
    {
        --TotalAtomic;

        --TotalAtomic64;
    }

    return __XSIGMA_THREAD_RETURN_VALUE__;
}

__XSIGMA_THREAD_RETURN_TYPE__ MyFunction3(void*)
{
    for (int i = 0; i < Target / NumThreads; i++)
    {
        int idx = TotalAtomic += 1;
        Values32[idx]++;

        idx = TotalAtomic64 += 1;
        Values64[idx]++;
    }

    return __XSIGMA_THREAD_RETURN_VALUE__;
}

__XSIGMA_THREAD_RETURN_TYPE__ MyFunction4(void*)
{
    for (int i = 0; i < Target / NumThreads; i++)
    {
        TotalAtomic++;
        TotalAtomic += 1;
        TotalAtomic--;
        TotalAtomic -= 1;

        TotalAtomic64++;
        TotalAtomic64 += 1;
        TotalAtomic64--;
        TotalAtomic64 -= 1;
    }

    return __XSIGMA_THREAD_RETURN_VALUE__;
}

XSIGMATEST(Core, Atomic)
{
#ifdef SHOW_DIFFERENCE
    Total   = 0;
    Total64 = 0;
#endif

    TotalAtomic   = 0;
    TotalAtomic64 = 0;

    for (int i = 0; i <= Target; i++)
    {
        Values32[i] = 0;
        Values64[i] = 0;
    }

    auto mt = xsigma::util::make_ptr_unique_mutable<xsigma::multi_threader>();
    mt->SetSingleMethod(MyFunction, nullptr);
    mt->SetNumberOfThreads(NumThreads);
    mt->SingleMethodExecute();

    mt->SetSingleMethod(MyFunction2, nullptr);
    mt->SingleMethodExecute();

    mt->SetSingleMethod(MyFunction3, nullptr);
    mt->SingleMethodExecute();

    // Making sure that atomic incr returned unique
    // values each time. We expect all numbers from
    // 1 to Target to be 2.
    EXPECT_EQ(Values32[0], 0);
    EXPECT_EQ(Values64[0], 0);
    for (int i = 1; i <= Target; i++)
    {
        EXPECT_EQ(Values32[i], 2);
        EXPECT_EQ(Values64[i], 2);
    }

    /*xsigmaMTimeType *from = MTimeValues, *to = MTimeValues + Target;
  std::sort(from, to);
  if (std::unique(from, to) != to)
  {
    cout << "Found duplicate MTime Values" << endl;
    return 1;
  }*/

    mt->SetSingleMethod(MyFunction4, nullptr);
    mt->SingleMethodExecute();

#ifdef SHOW_DIFFERENCE
    cout << Total << " " << TotalAtomic.load() << endl;
    cout << Total64 << " " << TotalAtomic64.load() << endl;
#endif

    EXPECT_EQ(TotalAtomic.load(), Target);

    EXPECT_EQ(TotalAtomic64.load(), Target);

    END_TEST();
}
