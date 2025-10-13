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

#include <thread>
#include <vector>

#include "common/pointer.h"
#include "logging/logger.h"
#include "memory/cpu/allocator.h"
#include "memory/cpu/allocator_pool.h"
#include "memory/cpu/allocator_report_generator.h"
#include "memory/cpu/allocator_tracking.h"
#include "xsigmaTest.h"

using namespace xsigma;

// ============================================================================
// Report Generation Tests
// ============================================================================

XSIGMATEST(AllocatorReportGeneration, BasicReport)
{
    XSIGMA_LOG_INFO("Testing basic report generation...");

    // Create tracking allocator
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        20,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "test_report_basic");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Perform some allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i)
    {
        size_t size = 128 * (i + 1);
        void*  ptr  = tracking->allocate_raw(64, size);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Generate report
    allocator_report_generator generator;
    report_config              config;
    config.include_leak_detection      = true;
    config.include_memory_graphs       = true;
    config.include_allocation_patterns = true;
    config.include_timing_stats        = true;
    config.include_recommendations     = true;

    std::string report = generator.generate_comprehensive_report(*tracking, config);

    // Verify report contains expected sections
    EXPECT_TRUE(report.find("Summary Statistics") != std::string::npos);
    EXPECT_TRUE(report.find("Memory Leak Detection") != std::string::npos);
    EXPECT_TRUE(report.find("Allocation Patterns") != std::string::npos);
    EXPECT_TRUE(report.find("Performance Statistics") != std::string::npos);

    // Display report
    XSIGMA_LOG_INFO("\n{}", report);

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    XSIGMA_LOG_INFO("Basic report generation test completed successfully");
}

XSIGMATEST(AllocatorReportGeneration, LeakDetection)
{
    XSIGMA_LOG_INFO("Testing leak detection in reports...");

    // Create tracking allocator
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        20,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "test_report_leaks");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Allocate some memory and intentionally "leak" it
    std::vector<void*> leaked_ptrs;
    for (int i = 0; i < 10; ++i)
    {
        void* ptr = tracking->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            leaked_ptrs.push_back(ptr);
        }
    }

    // Wait a bit to make allocations "old"
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Generate report with leak detection
    allocator_report_generator generator;
    auto                       leaks = generator.detect_leaks(*tracking, 50);  // 50ms threshold

    XSIGMA_LOG_INFO("Detected {} potential leaks", leaks.size());
    EXPECT_GT(leaks.size(), 0);

    // Generate full report
    report_config config;
    config.include_leak_detection = true;
    config.max_leak_reports       = 5;

    std::string report = generator.generate_comprehensive_report(*tracking, config);

    //EXPECT_TRUE(report.find("Memory Leak Detection") != std::string::npos);
    //EXPECT_TRUE(report.find("potential memory leak") != std::string::npos);

    XSIGMA_LOG_INFO("\n{}", report);

    // Cleanup
    for (void* ptr : leaked_ptrs)
    {
        tracking->deallocate_raw(ptr);
    }
    XSIGMA_LOG_INFO("\n{}", tracking->GenerateReport(true));

    XSIGMA_LOG_INFO("Leak detection test completed successfully");
}

XSIGMATEST(AllocatorReportGeneration, PerformanceAnalysis)
{
    XSIGMA_LOG_INFO("Testing performance analysis in reports...");

    // Create tracking allocator
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        50,
        true,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "test_report_perf");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Perform many allocations to get timing statistics
    std::vector<void*> ptrs;
    for (int i = 0; i < 1000; ++i)
    {
        void* ptr = tracking->allocate_raw(64, 512);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Deallocate half
    for (size_t i = 0; i < ptrs.size() / 2; ++i)
    {
        tracking->deallocate_raw(ptrs[i]);
    }

    // Generate performance report
    allocator_report_generator generator;
    auto                       timing_stats = tracking->GetTimingStats();

    std::string perf_report = generator.generate_performance_report(timing_stats);

    EXPECT_TRUE(perf_report.find("Total Allocations") != std::string::npos);
    EXPECT_TRUE(perf_report.find("Avg Allocation Time") != std::string::npos);
    EXPECT_TRUE(perf_report.find("Performance Assessment") != std::string::npos);

    XSIGMA_LOG_INFO("\n{}", perf_report);

    // Cleanup
    for (size_t i = ptrs.size() / 2; i < ptrs.size(); ++i)
    {
        tracking->deallocate_raw(ptrs[i]);
    }

    XSIGMA_LOG_INFO("Performance analysis test completed successfully");
}

XSIGMATEST(AllocatorReportGeneration, ReportBuilder)
{
    XSIGMA_LOG_INFO("Testing report builder pattern...");

    // Create tracking allocator
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        30,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "test_report_builder");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Perform allocations
    std::vector<void*> ptrs;
    for (int i = 0; i < 100; ++i)
    {
        void* ptr = tracking->allocate_raw(64, 256 * (i % 10 + 1));
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Use builder pattern to create report
    memory_report_builder builder;
    std::string           report = builder.with_allocator(tracking)
                             .with_leak_detection(true)
                             .with_memory_graphs(true)
                             .with_performance_analysis(true)
                             .with_fragmentation_analysis(true)
                             .with_recommendations(true)
                             .with_detailed_allocations(false)
                             .with_leak_threshold_ms(100)
                             .with_max_leak_reports(5)
                             .build();

    EXPECT_FALSE(report.empty());
    EXPECT_TRUE(report.find("XSigma Memory Allocation Tracking Report") != std::string::npos);

    XSIGMA_LOG_INFO("\n{}", report);

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    XSIGMA_LOG_INFO("Report builder test completed successfully");
}

XSIGMATEST(AllocatorReportGeneration, SizeDistribution)
{
    XSIGMA_LOG_INFO("Testing size distribution analysis...");

    // Create tracking allocator
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        40,
        false,
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "test_report_distribution");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Allocate various sizes
    std::vector<void*> ptrs;
    const size_t       sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

    for (int round = 0; round < 10; ++round)
    {
        for (size_t size : sizes)
        {
            void* ptr = tracking->allocate_raw(64, size);
            if (ptr != nullptr)
            {
                ptrs.push_back(ptr);
            }
        }
    }

    // Generate size distribution
    allocator_report_generator generator;
    auto                       records = tracking->GetEnhancedRecords();

    std::string distribution = generator.generate_size_distribution(records);

    EXPECT_FALSE(distribution.empty());
    EXPECT_TRUE(distribution.find("Allocation Size Distribution") != std::string::npos);

    XSIGMA_LOG_INFO("\n{}", distribution);

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    XSIGMA_LOG_INFO("Size distribution test completed successfully");
}

XSIGMATEST(AllocatorReportGeneration, Recommendations)
{
    XSIGMA_LOG_INFO("Testing optimization recommendations...");

    // Create tracking allocator with suboptimal configuration
    auto base_allocator = util::make_ptr_unique_mutable<basic_cpu_allocator>(
        0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

    auto pool = std::make_unique<allocator_pool>(
        5,      // Small pool to create inefficiency
        false,  // No auto-resize
        std::move(base_allocator),
        util::make_ptr_unique_mutable<NoopRounder>(),
        "test_report_recommendations");

    auto tracking = new allocator_tracking(pool.get(), true, true);

    // Perform many allocations to stress the allocator
    std::vector<void*> ptrs;
    for (int i = 0; i < 200; ++i)
    {
        void* ptr = tracking->allocate_raw(64, 1024);
        if (ptr != nullptr)
        {
            ptrs.push_back(ptr);
        }
    }

    // Generate recommendations
    allocator_report_generator generator;
    std::string                recommendations = generator.generate_recommendations(*tracking);

    EXPECT_FALSE(recommendations.empty());

    XSIGMA_LOG_INFO("\n{}", recommendations);

    // Cleanup
    for (void* ptr : ptrs)
    {
        tracking->deallocate_raw(ptr);
    }

    XSIGMA_LOG_INFO("Recommendations test completed successfully");
}
