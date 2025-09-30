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

#include <thread>
#include <vector>

#include "util/logger.h"

// Benchmark: Simple log message (INFO level)
static void BM_LogSimpleInfo(benchmark::State& state)
{
    for (auto _ : state)
    {
        xsigma::logger::Log(xsigma::logger::Severity::INFO, "Simple info message");
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogSimpleInfo);

// Benchmark: Simple log message (WARNING level)
static void BM_LogSimpleWarning(benchmark::State& state)
{
    for (auto _ : state)
    {
        xsigma::logger::Log(xsigma::logger::Severity::WARNING, "Simple warning message");
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogSimpleWarning);

// Benchmark: Simple log message (ERROR level)
static void BM_LogSimpleError(benchmark::State& state)
{
    for (auto _ : state)
    {
        xsigma::logger::Log(xsigma::logger::Severity::ERROR, "Simple error message");
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogSimpleError);

// Benchmark: Formatted log message with single integer
static void BM_LogFormattedSingleInt(benchmark::State& state)
{
    int value = 42;
    for (auto _ : state)
    {
        xsigma::logger::LogF(xsigma::logger::Severity::INFO, "Value: %d", value);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogFormattedSingleInt);

// Benchmark: Formatted log message with multiple arguments
static void BM_LogFormattedMultipleArgs(benchmark::State& state)
{
    int    int_val    = 42;
    double double_val = 3.14159;
    const char* str_val = "test";
    
    for (auto _ : state)
    {
        xsigma::logger::LogF(
            xsigma::logger::Severity::INFO,
            "Int: %d, Double: %.2f, String: %s",
            int_val,
            double_val,
            str_val);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogFormattedMultipleArgs);

// Benchmark: Formatted log message with long string
static void BM_LogFormattedLongString(benchmark::State& state)
{
    const char* long_string = "This is a relatively long string that might be used in a log message "
                              "to test the performance impact of longer messages";
    
    for (auto _ : state)
    {
        xsigma::logger::LogF(xsigma::logger::Severity::INFO, "Message: %s", long_string);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogFormattedLongString);

// Benchmark: Scope creation and destruction
static void BM_LogScopeCreation(benchmark::State& state)
{
    for (auto _ : state)
    {
        auto scope = xsigma::logger::StartScope("BenchmarkScope");
        benchmark::DoNotOptimize(scope);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogScopeCreation);

// Benchmark: Formatted scope creation
static void BM_LogScopeFormattedCreation(benchmark::State& state)
{
    int iteration = 0;
    for (auto _ : state)
    {
        auto scope = xsigma::logger::StartScopeF("BenchmarkScope_%d", iteration++);
        benchmark::DoNotOptimize(scope);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogScopeFormattedCreation);

// Benchmark: Nested scopes
static void BM_LogNestedScopes(benchmark::State& state)
{
    for (auto _ : state)
    {
        auto scope1 = xsigma::logger::StartScope("OuterScope");
        {
            auto scope2 = xsigma::logger::StartScope("MiddleScope");
            {
                auto scope3 = xsigma::logger::StartScope("InnerScope");
                benchmark::DoNotOptimize(scope3);
            }
            benchmark::DoNotOptimize(scope2);
        }
        benchmark::DoNotOptimize(scope1);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogNestedScopes);

// Benchmark: Thread-safe logging with multiple threads
static void BM_LogMultiThreaded(benchmark::State& state)
{
    if (state.thread_index() == 0)
    {
        // Setup code (runs once per benchmark)
    }

    for (auto _ : state)
    {
        xsigma::logger::Log(
            xsigma::logger::Severity::INFO,
            "Multi-threaded log message");
    }

    if (state.thread_index() == 0)
    {
        // Teardown code (runs once per benchmark)
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogMultiThreaded)->Threads(1);
BENCHMARK(BM_LogMultiThreaded)->Threads(2);
BENCHMARK(BM_LogMultiThreaded)->Threads(4);
BENCHMARK(BM_LogMultiThreaded)->Threads(8);

// Benchmark: Thread-safe formatted logging with multiple threads
static void BM_LogFormattedMultiThreaded(benchmark::State& state)
{
    int thread_id = state.thread_index();

    for (auto _ : state)
    {
        xsigma::logger::LogF(
            xsigma::logger::Severity::INFO,
            "Thread %d: iteration %lld",
            thread_id,
            state.iterations());
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogFormattedMultiThreaded)->Threads(1);
BENCHMARK(BM_LogFormattedMultiThreaded)->Threads(2);
BENCHMARK(BM_LogFormattedMultiThreaded)->Threads(4);
BENCHMARK(BM_LogFormattedMultiThreaded)->Threads(8);

// Benchmark: Scope overhead in multi-threaded environment
static void BM_LogScopeMultiThreaded(benchmark::State& state)
{
    int thread_id = state.thread_index();

    for (auto _ : state)
    {
        auto scope = xsigma::logger::StartScopeF("Thread_%d_Scope", thread_id);
        benchmark::DoNotOptimize(scope);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogScopeMultiThreaded)->Threads(1);
BENCHMARK(BM_LogScopeMultiThreaded)->Threads(2);
BENCHMARK(BM_LogScopeMultiThreaded)->Threads(4);
BENCHMARK(BM_LogScopeMultiThreaded)->Threads(8);

// Benchmark: Mixed operations (realistic workload)
static void BM_LogMixedOperations(benchmark::State& state)
{
    int counter = 0;
    
    for (auto _ : state)
    {
        // Simulate a typical function with logging
        auto scope = xsigma::logger::StartScopeF("Operation_%d", counter);
        
        xsigma::logger::Log(xsigma::logger::Severity::INFO, "Starting operation");
        
        xsigma::logger::LogF(
            xsigma::logger::Severity::INFO,
            "Processing item %d",
            counter);
        
        if (counter % 10 == 0)
        {
            xsigma::logger::Log(xsigma::logger::Severity::WARNING, "Checkpoint reached");
        }
        
        xsigma::logger::Log(xsigma::logger::Severity::INFO, "Operation completed");
        
        counter++;
        benchmark::DoNotOptimize(scope);
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogMixedOperations);

// Benchmark: Memory allocation overhead
static void BM_LogMemoryAllocation(benchmark::State& state)
{
    std::vector<std::string> messages;
    messages.reserve(100);
    
    for (int i = 0; i < 100; ++i)
    {
        messages.push_back("Message " + std::to_string(i));
    }
    
    size_t index = 0;
    for (auto _ : state)
    {
        xsigma::logger::Log(
            xsigma::logger::Severity::INFO,
            messages[index % messages.size()].c_str());
        index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_LogMemoryAllocation);

// Benchmark: SetThreadName performance
static void BM_SetThreadName(benchmark::State& state)
{
    for (auto _ : state)
    {
        xsigma::logger::SetThreadName("BenchmarkThread");
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SetThreadName);

// Main function for benchmark
BENCHMARK_MAIN();

