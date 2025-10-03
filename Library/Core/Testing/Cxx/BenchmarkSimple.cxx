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

#include <benchmark/benchmark.h>

#include <cmath>
#include <vector>
#include <string>

// Simple benchmarks to verify Google Benchmark integration works
// This is a minimal example following the cpuinfo integration pattern

// Benchmark: Vector push_back operation
static void BM_VectorPushBack(benchmark::State& state)
{
    for (auto _ : state)
    {
        std::vector<int> v;
        v.reserve(state.range(0));
        for (int i = 0; i < state.range(0); ++i)
        {
            v.push_back(i);
        }
        benchmark::DoNotOptimize(v.data());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_VectorPushBack)->Range(8, 8 << 10);

// Benchmark: String operations
static void BM_StringCopy(benchmark::State& state)
{
    std::string x = "hello world";
    for (auto _ : state)
    {
        std::string copy(x);
        benchmark::DoNotOptimize(copy);
    }
}
BENCHMARK(BM_StringCopy);

// Benchmark: Math operations
static void BM_MathOperations(benchmark::State& state)
{
    double x = 1.0;
    for (auto _ : state)
    {
        x = std::sqrt(x + 1.0);
        benchmark::DoNotOptimize(x);
    }
}
BENCHMARK(BM_MathOperations);

// Benchmark: Vector reserve and resize
static void BM_VectorReserve(benchmark::State& state)
{
    for (auto _ : state)
    {
        std::vector<int> v;
        v.reserve(1000);
        benchmark::DoNotOptimize(v.data());
    }
}
BENCHMARK(BM_VectorReserve);

// Main function
BENCHMARK_MAIN();

