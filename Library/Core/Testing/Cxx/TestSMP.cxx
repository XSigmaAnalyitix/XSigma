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
#include "smp/Common/thread_local_api.h"
#include "smp/Common/tools_api.h"
#include "smp/tools.h"
#include "smp/xsigma_thread_local.h"
#include "util/logger.h"
#include "xsigmaTest.h"
#include "xsigma_smp.h"  // IWYU pragma: keep

static const int Target = 10000;
using namespace xsigma;

class ARangeFunctor
{
public:
    xsigma_thread_local<int> Counter;

    ARangeFunctor() : Counter(0) {}

    void operator()(int begin, int end)
    {
        for (int i = begin; i < end; i++)
            this->Counter.Local()++;
    }
};

template <typename Iterator>
class ForRangeFunctor
{
public:
    xsigma_thread_local<double> Counter;

    ForRangeFunctor() : Counter(0) {}

    void operator()(Iterator begin, Iterator end)
    {
        for (auto it = begin; it != end; ++it)
        {
            this->Counter.Local() += *it;
        }
    }
};

template <typename Iterator>
class NestedFunctor
{
public:
    xsigma_thread_local<int> Counter;
    const int                Factor;

    NestedFunctor() : Counter(0), Factor(100) {}

    void operator()(Iterator begin, Iterator end)
    {
        for (auto it = begin; it != end; ++it)
        {
            for (int i = 0; i < *it; ++i)
            {
                xsigma_thread_local<int> nestedCounter(0);
                tools::For(
                    0,
                    this->Factor,
                    [&](int start, int stop)
                    {
                        for (int j = start; j < stop; ++j)
                        {
                            nestedCounter.Local()++;
                        }
                    });
                for (const auto& el : nestedCounter)
                {
                    this->Counter.Local() += el;
                }
            }
        }
    }
};

class NestedSingleFunctor
{
public:
    xsigma_thread_local<int> Counter;

    NestedSingleFunctor() : Counter(0) {}

    void operator()(int begin, int end)
    {
        bool isSingleOuter = tools::GetSingleThread();
        if (!isSingleOuter)
        {
            return;
        }
        for (int i = begin; i < end; i++)
        {
            xsigma_thread_local<int> nestedCounter(0);
            tools::For(
                0,
                100,
                [&](int start, int stop)
                {
                    bool isSingleInner = tools::GetSingleThread();
                    if (!isSingleInner)
                    {
                        return;
                    }
                    for (int j = start; j < stop; ++j)
                    {
                        nestedCounter.Local()++;
                    }
                });

            for (const auto& el : nestedCounter)
            {
                this->Counter.Local() += el;
            }
        }
    }
};

// For sorting comparison
bool myComp(double a, double b)
{
    return (a < b);
}

void doTestSMP()
{
    XSIGMA_LOG_INFO("Testing smp Tools with " << xsigma::tools::GetBackend() << " backend.");
    ARangeFunctor functor1;

    tools::For(0, Target, functor1);

    xsigma_thread_local<int>::iterator itr1 = functor1.Counter.begin();
    xsigma_thread_local<int>::iterator end1 = functor1.Counter.end();

    int total = 0;
    while (itr1 != end1)
    {
        total += *itr1;
        ++itr1;
    }

    EXPECT_EQ(total, Target);

    // Test For with range
    std::set<double> forData0;
    for (int i = 0; i < 1000; ++i)
    {
        forData0.emplace(i * 2);
    }
    ForRangeFunctor<std::set<double>::const_reverse_iterator> functor3;
    tools::For(forData0.crbegin(), forData0.crend(), functor3);
    total         = 0;
    int sumTarget = std::accumulate(forData0.begin(), forData0.end(), 0);
    for (const auto& el : functor3.Counter)
    {
        total += (int)el;
    }

    EXPECT_EQ(total, sumTarget);

    // Test IsParallelScope
    /*if (std::string(tools::GetBackend()) != "Sequential")
    {
        xsigma_thread_local<int> isParallel(0);
        int                   target = 20;
        tools::For(
            0,
            target,
            1,
            [&](int start, int end)
            {
                for (int i = start; i < end; ++i)
                {
                    isParallel.Local() += static_cast<int>(tools::IsParallelScope());
                }
            });
        total = 0;
        for (const auto& it : isParallel)
        {
            total += it;
        }

        EXPECT_EQ(total, target);
    }*/

    // Test nested parallelism
    for (const bool enabled : {true, false})
    {
        std::vector<int>                                nestedData0 = {5, 3, 8, 1, 10};
        NestedFunctor<std::vector<int>::const_iterator> functor4;
        tools::LocalScope(
            tools::Config{enabled},
            [&]() { tools::For(nestedData0.cbegin(), nestedData0.cend(), functor4); });

        sumTarget = functor4.Factor * std::accumulate(nestedData0.begin(), nestedData0.end(), 0);
        total     = 0;
        for (const auto& el : functor4.Counter)
        {
            total += el;
        }

        EXPECT_EQ(total, sumTarget);
    }

    /* This Test is faulty, see: https://gitlab.kitware.com/xsigma/xsigma/-/issues/19338
  // Test GetSingleThread
  if (std::string(tools::GetBackend()) != "Sequential")
  {
    NestedSingleFunctor functor5;
    tools::LocalScope(
      tools::Config{ true }, [&]() { tools::For(0, 100, functor5); });

    xsigma_thread_local<int>::iterator itr5 = functor5.Counter.begin();
    xsigma_thread_local<int>::iterator end5 = functor5.Counter.end();

    total = 0;
    while (itr5 != end5)
    {
      total += *itr5;
      ++itr5;
    }

     EXPECT_FALSE(total >= Target);
  }*/

    // Test LocalScope
    const int targetThreadNb = 2;
    int       scopeThreadNb  = 0;

    auto lambdaScope0 = [&]() { scopeThreadNb = tools::GetEstimatedNumberOfThreads(); };
    tools::LocalScope(tools::Config{targetThreadNb}, lambdaScope0);
    EXPECT_FALSE(scopeThreadNb <= 0 || scopeThreadNb > targetThreadNb);

    const bool isNestedTarget = true;
    bool       isNested       = false;

    auto lambdaScope1 = [&]() { isNested = tools::GetNestedParallelism(); };
    tools::LocalScope(tools::Config{isNestedTarget}, lambdaScope1);
    EXPECT_EQ(isNested, isNestedTarget);

    // Test sorting
    std::array<double, 11> data0 = {2, 1, 0, 3, 9, 6, 7, 3, 8, 4, 5};
    std::vector<double>    myvector(data0.begin(), data0.begin() + 11);
    std::array<double, 11> data1 = {2, 1, 0, 3, 9, 6, 7, 3, 8, 4, 5};
    std::array<double, 11> sdata = {0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9};

    // using default comparison (operator <):
    tools::Sort(myvector.begin(), myvector.begin() + 11);
    for (int i = 0; i < 11; ++i)
    {
        EXPECT_EQ(myvector[i], sdata[i]);
    }

    tools::Sort(data1.begin(), data1.begin() + 11, myComp);
    for (int i = 0; i < 11; ++i)
    {
        EXPECT_EQ(data1[i], sdata[i]);
    }

    // Test transform
    std::vector<double> transformData0 = {51, 9, 3, -10, 27, 1, -5, 82, 31, 9, 21};
    std::vector<double> transformData1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::set<double>    transformData2 = {7, 24, 98, 256, 72, 19, 3, 21, 2, 12};
    std::vector<double> transformData3 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    std::vector<double> transformData4 = {8, 23, 123, 9, 23, 1, 4, 20, 1, 7, 38, 21};
    std::vector<double> transformData5 = {0, 0, 0, 0};

    xsigma::tools::Transform(
        transformData0.cbegin(),
        transformData0.cend(),
        transformData1.cbegin(),
        transformData1.begin(),
        [](double x, double y) { return x * y; });
    auto it0 = transformData0.begin();
    auto it1 = transformData1.begin();
    for (size_t i = 0; i < 11; ++i, it0++, it1++)
    {
        EXPECT_EQ(*it1, *it0 * i);
    }

    tools::Transform(
        transformData2.cbegin(),
        transformData2.cend(),
        transformData3.begin(),
        [](double x) { return x - 1; });
    auto it2 = transformData2.begin();
    for (const auto& it3 : transformData3)
    {
        EXPECT_EQ(it3, *it2 - 1);
        it2++;
    }

    // Test fill
    std::vector<double> fillData0 = {51, 9, 3, -10, 27, 1, -5, 82, 31, 9};
    std::deque<double>  fillData1 = {0, 0, 0, 0, 0};

    const auto fillValue0 = 0.5;
    xsigma::tools::Fill(fillData0.begin(), fillData0.end(), fillValue0);
    for (auto it : fillData0)
    {
        EXPECT_EQ(it, fillValue0);
    }

    const double fillValue1 = 42;
    tools::Fill(fillData1.begin(), fillData1.end(), fillValue1);
    for (auto& it : fillData1)
    {
        EXPECT_EQ(it, fillValue1);
    }
}

XSIGMATEST(Core, SMP)
{
#ifdef XSIGMA_GOOGLE_TEST
    xsigma::tools::SetBackend("STDTHREAD");
    doTestSMP();

    xsigma::tools::SetBackend("SEQUENTIAL");
    doTestSMP();
#else
    for (int i = 1; i < argc; i++)
    {
        std::string argument(argv[i] + 2);
        std::size_t separator = argument.find('=');
        std::string backend   = argument.substr(0, separator);
        int         value     = std::atoi(argument.substr(separator + 1, argument.size()).c_str());
        if (value != 0)
        {
            tools::SetBackend(backend.c_str());
            doTestSMP();
        }
    }

    END_TEST();
#endif
}
