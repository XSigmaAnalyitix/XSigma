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

#include "memory/cpu/allocator_report_generator.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include "logging/logger.h"
#include "memory/cpu/allocator.h"
#include "memory/cpu/allocator_tracking.h"

namespace xsigma
{

// ============================================================================
// Helper Functions
// ============================================================================

std::string allocator_report_generator::format_bytes(size_t bytes) 
{
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int         unit    = 0;
    auto      size    = static_cast<double>(bytes);

    while (size >= 1024.0 && unit < 4)
    {
        size /= 1024.0;
        unit++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return oss.str();
}

std::string allocator_report_generator::format_duration(int64_t microseconds) 
{
    if (microseconds < 1000)
    {
        return std::to_string(microseconds) + " μs";
    }
    if (microseconds < 1000000)
    {
        return std::to_string(microseconds / 1000) + " ms";
    }
    
            return std::to_string(microseconds / 1000000) + " s";
   
}

std::string allocator_report_generator::generate_header(const std::string& title) 
{
    std::ostringstream header;
    std::string        const separator(80, '=');

    header << "\n" << separator << "\n";
    header << title << "\n";
    header << separator << "\n";

    return header.str();
}

std::string allocator_report_generator::generate_separator() 
{
    return std::string(80, '-') + "\n";
}

// ============================================================================
// Main Report Generation
// ============================================================================

std::string allocator_report_generator::generate_comprehensive_report(
    const allocator_tracking& allocator, const report_config& config) const
{
    std::ostringstream report;

    // Report header
    report << generate_header("XSigma Memory Allocation Tracking Report");

    auto now        = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    report << "Generated: " << std::ctime(&time_t_now);
    report << "Allocator: " << allocator.Name() << "\n\n";

    // Summary statistics
    report << generate_summary_statistics(allocator);

    // Memory leak detection
    if (config.include_leak_detection)
    {
        report << generate_header("Memory Leak Detection");
        auto leaks = detect_leaks(allocator, 60000);  // 60 second threshold
        report << generate_leak_report(leaks, config.max_leak_reports);
    }

    // Memory usage graphs
    if (config.include_memory_graphs)
    {
        report << generate_header("Memory Usage Visualization");
        auto stats_opt = allocator.GetStats();
        if (stats_opt.has_value())
        {
            auto&       stats      = stats_opt.value();
            std::string const usage_bars = visualizer_.create_usage_bars(
                stats.bytes_in_use.load(),
                stats.peak_bytes_in_use.load(),
                stats.bytes_limit.load());
            report << usage_bars << "\n";
        }
    }

    // Allocation patterns
    if (config.include_allocation_patterns)
    {
        report << generate_header("Allocation Patterns");
        auto records = allocator.GetEnhancedRecords();
        report << analyze_allocation_patterns(records);
        report << "\n" << generate_size_distribution(records);
    }

    // Performance timing statistics
    if (config.include_timing_stats)
    {
        report << generate_header("Performance Statistics");
        auto timing_stats = allocator.GetTimingStats();
        report << generate_performance_report(timing_stats);
    }

    // Fragmentation analysis
    if (config.include_fragmentation)
    {
        report << generate_header("Fragmentation Analysis");
        auto metrics = allocator.GetFragmentationMetrics();
        report << generate_fragmentation_report(metrics);
    }

    // Optimization recommendations
    if (config.include_recommendations)
    {
        report << generate_header("Optimization Recommendations");
        report << generate_recommendations(allocator);
    }

    // Detailed allocations
    if (config.include_detailed_allocations)
    {
        report << generate_header("Detailed Allocation Records");
        auto   records = allocator.GetEnhancedRecords();
        size_t const count   = std::min(records.size(), config.max_allocation_details);

        for (size_t i = 0; i < count; ++i)
        {
            const auto& rec = records[i];
            report << "  [" << rec.allocation_id << "] " << format_bytes(rec.requested_bytes)
                   << " / " << format_bytes(rec.alloc_bytes) << " bytes, "
                   << "align=" << rec.alignment << ", "
                   << "time=" << rec.alloc_duration_us << "μs";

            if (!rec.tag.empty())
            {
                report << ", tag=" << rec.tag;
            }

            if (rec.source_file != nullptr)
            {
                report << ", " << rec.source_file << ":" << rec.source_line;
            }

            report << "\n";
        }

        if (records.size() > count)
        {
            report << "  ... and " << (records.size() - count) << " more allocations\n";
        }
    }

    // Report footer
    report << "\n" << generate_separator();
    report << "End of Report\n";
    report << generate_separator();

    // Export to file if requested
    if (config.export_to_file && !config.export_filename.empty())
    {
        if (export_report(report.str(), config.export_filename))
        {
            report << "\nReport exported to: " << config.export_filename << "\n";
        }
        else
        {
            report << "\nFailed to export report to: " << config.export_filename << "\n";
        }
    }

    return report.str();
}

// ============================================================================
// Leak Detection
// ============================================================================

std::vector<memory_leak_info> allocator_report_generator::detect_leaks(
    const allocator_tracking& allocator, int64_t leak_threshold_ms) 
{
    std::vector<memory_leak_info> leaks;

    auto records = allocator.GetEnhancedRecords();
    auto now     = std::chrono::high_resolution_clock::now();

    for (const auto& record : records)
    {
        // Calculate age of allocation
        auto alloc_time      = std::chrono::microseconds(record.alloc_micros);
        auto alloc_timepoint = std::chrono::high_resolution_clock::time_point(alloc_time);
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - alloc_timepoint);

        if (age.count() > leak_threshold_ms)
        {
            memory_leak_info leak;
            leak.size               = record.requested_bytes;
            leak.allocation_id      = record.allocation_id;
            leak.allocation_time_us = record.alloc_micros;
            leak.age_ms             = age.count();
            leak.tag                = record.tag;
            leak.source_file        = record.source_file;
            leak.source_line        = record.source_line;
            leak.function_name      = record.function_name;

            leaks.push_back(leak);
        }
    }

    // Sort by age (oldest first)
    std::sort(
        leaks.begin(),
        leaks.end(),
        [](const memory_leak_info& a, const memory_leak_info& b) { return a.age_ms > b.age_ms; });

    return leaks;
}

std::string allocator_report_generator::generate_leak_report(
    const std::vector<memory_leak_info>& leaks, size_t max_reports) 
{
    std::ostringstream report;

    if (leaks.empty())
    {
        report << "No memory leaks detected.\n\n";
        return report.str();
    }

    report << "Detected " << leaks.size() << " potential memory leak(s):\n\n";

    size_t const count = std::min(leaks.size(), max_reports);
    for (size_t i = 0; i < count; ++i)
    {
        const auto& leak = leaks[i];

        report << "Leak #" << (i + 1) << ":\n";
        report << "  Size: " << format_bytes(leak.size) << "\n";
        report << "  Age: " << format_duration(leak.age_ms * 1000) << "\n";
        report << "  Allocation ID: " << leak.allocation_id << "\n";

        if (!leak.tag.empty())
        {
            report << "  Tag: " << leak.tag << "\n";
        }

        if (leak.source_file != nullptr)
        {
            report << "  Location: " << leak.source_file << ":" << leak.source_line;
            if (leak.function_name != nullptr)
            {
                report << " (" << leak.function_name << ")";
            }
            report << "\n";
        }

        report << "\n";
    }

    if (leaks.size() > count)
    {
        report << "... and " << (leaks.size() - count) << " more leak(s)\n\n";
    }

    return report.str();
}

// ============================================================================
// Summary Statistics
// ============================================================================

std::string allocator_report_generator::generate_summary_statistics(
    const allocator_tracking& allocator) 
{
    std::ostringstream summary;

    summary << generate_header("Summary Statistics");

    auto stats_opt = allocator.GetStats();
    if (!stats_opt.has_value())
    {
        summary << "Statistics not available.\n\n";
        return summary.str();
    }

    const auto& stats = stats_opt.value();

    summary << "Total Allocations:     " << stats.num_allocs.load() << "\n";
    summary << "Total Deallocations:   " << stats.num_deallocs.load() << "\n";
    summary << "Active Allocations:    " << stats.active_allocations.load() << "\n";
    summary << "Current Memory Usage:  " << format_bytes(stats.bytes_in_use.load()) << "\n";
    summary << "Peak Memory Usage:     " << format_bytes(stats.peak_bytes_in_use.load()) << "\n";
    summary << "Total Allocated:       " << format_bytes(stats.total_bytes_allocated.load())
            << "\n";
    summary << "Total Deallocated:     " << format_bytes(stats.total_bytes_deallocated.load())
            << "\n";
    summary << "Largest Allocation:    " << format_bytes(stats.largest_alloc_size.load()) << "\n";
    summary << "Failed Allocations:    " << stats.failed_allocations.load() << "\n";

    if (stats.num_allocs.load() > 0)
    {
        double const avg_size = stats.average_allocation_size();
        summary << "Average Allocation:    " << format_bytes(static_cast<size_t>(avg_size)) << "\n";
    }

    summary << "Memory Efficiency:     " << std::fixed << std::setprecision(2)
            << (stats.memory_efficiency() * 100.0) << "%\n";

    auto [utilization, overhead, efficiency] = allocator.GetEfficiencyMetrics();
    summary << "Utilization Ratio:     " << std::fixed << std::setprecision(2)
            << (utilization * 100.0) << "%\n";
    summary << "Overhead Ratio:        " << std::fixed << std::setprecision(2) << (overhead * 100.0)
            << "%\n";
    summary << "Efficiency Score:      " << std::fixed << std::setprecision(2)
            << (efficiency * 100.0) << "%\n";

    summary << "\n";

    return summary.str();
}

// ============================================================================
// Performance Report
// ============================================================================

std::string allocator_report_generator::generate_performance_report(
    const atomic_timing_stats& timing_stats) 
{
    std::ostringstream perf;

    perf << "Total Allocations:      " << timing_stats.total_allocations.load() << "\n";
    perf << "Total Deallocations:    " << timing_stats.total_deallocations.load() << "\n";
    perf << "Avg Allocation Time:    " << std::fixed << std::setprecision(2)
         << timing_stats.average_alloc_time_us() << " μs\n";
    perf << "Avg Deallocation Time:  " << std::fixed << std::setprecision(2)
         << timing_stats.average_dealloc_time_us() << " μs\n";

    uint64_t const min_alloc = timing_stats.min_alloc_time_us.load();
    uint64_t const max_alloc = timing_stats.max_alloc_time_us.load();
    if (min_alloc != UINT64_MAX)
    {
        perf << "Min Allocation Time:    " << min_alloc << " μs\n";
        perf << "Max Allocation Time:    " << max_alloc << " μs\n";
    }

    uint64_t const min_dealloc = timing_stats.min_dealloc_time_us.load();
    uint64_t const max_dealloc = timing_stats.max_dealloc_time_us.load();
    if (min_dealloc != UINT64_MAX)
    {
        perf << "Min Deallocation Time:  " << min_dealloc << " μs\n";
        perf << "Max Deallocation Time:  " << max_dealloc << " μs\n";
    }

    // Performance assessment
    double const avg_alloc = timing_stats.average_alloc_time_us();
    perf << "\nPerformance Assessment:\n";
    if (avg_alloc < 1.0)
    {
        perf << "  Allocation Speed: EXCELLENT (< 1 μs)\n";
    }
    else if (avg_alloc < 10.0)
    {
        perf << "  Allocation Speed: GOOD (< 10 μs)\n";
    }
    else if (avg_alloc < 100.0)
    {
        perf << "  Allocation Speed: FAIR (< 100 μs)\n";
    }
    else
    {
        perf << "  Allocation Speed: POOR (> 100 μs)\n";
    }

    perf << "\n";

    return perf.str();
}

// ============================================================================
// Allocation Pattern Analysis
// ============================================================================

std::string allocator_report_generator::analyze_allocation_patterns(
    const std::vector<enhanced_alloc_record>& records) 
{
    std::ostringstream analysis;

    if (records.empty())
    {
        analysis << "No allocation records available.\n\n";
        return analysis.str();
    }

    // Analyze size patterns
    size_t total_requested = 0;
    size_t total_allocated = 0;
    size_t min_size        = SIZE_MAX;
    size_t max_size        = 0;

    for (const auto& record : records)
    {
        total_requested += record.requested_bytes;
        total_allocated += record.alloc_bytes;
        min_size = std::min(min_size, record.requested_bytes);
        max_size = std::max(max_size, record.requested_bytes);
    }

    double const avg_requested = static_cast<double>(total_requested) / records.size();
    double const avg_allocated = static_cast<double>(total_allocated) / records.size();
    double const overhead_ratio =
        total_allocated > 0
            ? static_cast<double>(total_allocated - total_requested) / total_allocated
            : 0.0;

    analysis << "Allocation Count:       " << records.size() << "\n";
    analysis << "Size Range:             " << format_bytes(min_size) << " - "
             << format_bytes(max_size) << "\n";
    analysis << "Average Requested:      " << format_bytes(static_cast<size_t>(avg_requested))
             << "\n";
    analysis << "Average Allocated:      " << format_bytes(static_cast<size_t>(avg_allocated))
             << "\n";
    analysis << "Overhead Ratio:         " << std::fixed << std::setprecision(2)
             << (overhead_ratio * 100.0) << "%\n";

    // Detect patterns
    analysis << "\nPattern Detection:\n";

    // Check for size predictability
    double size_variance = 0.0;
    for (const auto& record : records)
    {
        double const diff = static_cast<double>(record.requested_bytes) - avg_requested;
        size_variance += diff * diff;
    }
    size_variance /= records.size();
    double const size_stddev = std::sqrt(size_variance);

    if (size_stddev < avg_requested * 0.1)
    {
        analysis << "  Size Pattern: HIGHLY PREDICTABLE (low variance)\n";
    }
    else if (size_stddev < avg_requested * 0.5)
    {
        analysis << "  Size Pattern: MODERATELY PREDICTABLE\n";
    }
    else
    {
        analysis << "  Size Pattern: UNPREDICTABLE (high variance)\n";
    }

    analysis << "\n";

    return analysis.str();
}

std::string allocator_report_generator::generate_size_distribution(
    const std::vector<enhanced_alloc_record>& records) const
{
    if (records.empty())
    {
        return "No allocation records for size distribution.\n\n";
    }

    // Extract allocation sizes
    std::vector<size_t> allocation_sizes;
    allocation_sizes.reserve(records.size());

    for (const auto& record : records)
    {
        allocation_sizes.push_back(record.requested_bytes);
    }

    return visualizer_.create_histogram(allocation_sizes);
}

// ============================================================================
// Fragmentation Analysis
// ============================================================================

std::string allocator_report_generator::generate_fragmentation_report(
    const memory_fragmentation_metrics& metrics) 
{
    std::ostringstream frag;

    frag << "Total Free Blocks:      " << metrics.total_free_blocks << "\n";
    frag << "Largest Free Block:     " << format_bytes(metrics.largest_free_block) << "\n";
    frag << "Smallest Free Block:    " << format_bytes(metrics.smallest_free_block) << "\n";
    frag << "Average Free Block:     " << format_bytes(metrics.average_free_block_size) << "\n";
    frag << "Fragmentation Ratio:    " << std::fixed << std::setprecision(2)
         << (metrics.fragmentation_ratio * 100.0) << "%\n";
    frag << "External Fragmentation: " << std::fixed << std::setprecision(2)
         << (metrics.external_fragmentation * 100.0) << "%\n";

    frag << "\nFragmentation Assessment:\n";
    if (metrics.fragmentation_ratio < 0.1)
    {
        frag << "  Status: EXCELLENT (< 10% fragmentation)\n";
    }
    else if (metrics.fragmentation_ratio < 0.3)
    {
        frag << "  Status: GOOD (< 30% fragmentation)\n";
    }
    else if (metrics.fragmentation_ratio < 0.5)
    {
        frag << "  Status: FAIR (< 50% fragmentation)\n";
    }
    else
    {
        frag << "  Status: POOR (> 50% fragmentation)\n";
        frag << "  Recommendation: Consider defragmentation or allocator tuning\n";
    }

    frag << "\n";

    return frag.str();
}

// ============================================================================
// Recommendations
// ============================================================================

std::string allocator_report_generator::generate_recommendations(
    const allocator_tracking& allocator) 
{
    std::ostringstream recommendations;

    auto stats_opt = allocator.GetStats();
    if (!stats_opt.has_value())
    {
        recommendations << "Statistics not available for recommendations.\n\n";
        return recommendations.str();
    }

    const auto& stats                        = stats_opt.value();
    auto        timing_stats                 = allocator.GetTimingStats();
    auto [utilization, overhead, efficiency] = allocator.GetEfficiencyMetrics();

    bool has_recommendations = false;

    // Check for low efficiency
    if (efficiency < 0.7)
    {
        recommendations << "• Low efficiency detected (" << std::fixed << std::setprecision(1)
                        << (efficiency * 100.0) << "%):\n";
        recommendations << "  - Consider using a pool allocator for predictable sizes\n";
        recommendations << "  - Review allocation patterns for optimization opportunities\n\n";
        has_recommendations = true;
    }

    // Check for high overhead
    if (overhead > 0.3)
    {
        recommendations << "• High overhead detected (" << std::fixed << std::setprecision(1)
                        << (overhead * 100.0) << "%):\n";
        recommendations << "  - Consider aligning allocation sizes to reduce padding\n";
        recommendations << "  - Use power-of-2 sizes when possible\n\n";
        has_recommendations = true;
    }

    // Check for slow allocations
    double const avg_alloc_time = timing_stats.average_alloc_time_us();
    if (avg_alloc_time > 50.0)
    {
        recommendations << "• Slow allocation performance detected (" << std::fixed
                        << std::setprecision(2) << avg_alloc_time << " μs average):\n";
        recommendations << "  - Consider using a pool allocator for frequently allocated sizes\n";
        recommendations << "  - Reduce allocation frequency through object pooling\n\n";
        has_recommendations = true;
    }

    // Check for memory leaks
    int64_t const active = stats.active_allocations.load();
    int64_t const total  = stats.num_allocs.load();
    if (active > 0 && total > 0)
    {
        double const leak_ratio = static_cast<double>(active) / total;
        if (leak_ratio > 0.5)
        {
            recommendations << "• High number of active allocations (" << active << " / " << total
                            << "):\n";
            recommendations << "  - Review deallocation patterns for potential leaks\n";
            recommendations << "  - Use RAII patterns to ensure proper cleanup\n\n";
            has_recommendations = true;
        }
    }

    if (!has_recommendations)
    {
        recommendations << "No significant issues detected. Allocator performance is good.\n\n";
    }

    return recommendations.str();
}

// ============================================================================
// File Export
// ============================================================================

bool allocator_report_generator::export_report(
    const std::string& report, const std::string& filename) 
{
    try
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            XSIGMA_LOG_ERROR("Failed to open file for export: {}", filename);
            return false;
        }

        file << report;
        file.close();

        XSIGMA_LOG_INFO("Report exported successfully to: {}", filename);
        return true;
    }
    catch (const std::exception& e)
    {
        XSIGMA_LOG_ERROR("Exception during report export: {}", e.what());
        return false;
    }
}

// ============================================================================
// memory_report_builder Implementation
// ============================================================================

memory_report_builder::memory_report_builder() = default;

memory_report_builder& memory_report_builder::with_allocator(const allocator_tracking* allocator)
{
    allocator_ = allocator;
    return *this;
}

memory_report_builder& memory_report_builder::with_leak_detection(bool enable)
{
    config_.include_leak_detection = enable;
    return *this;
}

memory_report_builder& memory_report_builder::with_memory_graphs(bool enable)
{
    config_.include_memory_graphs = enable;
    return *this;
}

memory_report_builder& memory_report_builder::with_performance_analysis(bool enable)
{
    config_.include_timing_stats = enable;
    return *this;
}

memory_report_builder& memory_report_builder::with_fragmentation_analysis(bool enable)
{
    config_.include_fragmentation = enable;
    return *this;
}

memory_report_builder& memory_report_builder::with_recommendations(bool enable)
{
    config_.include_recommendations = enable;
    return *this;
}

memory_report_builder& memory_report_builder::with_detailed_allocations(bool enable)
{
    config_.include_detailed_allocations = enable;
    return *this;
}

memory_report_builder& memory_report_builder::with_leak_threshold_ms(int64_t threshold_ms)
{
    leak_threshold_ms_ = threshold_ms;
    return *this;
}

memory_report_builder& memory_report_builder::with_max_leak_reports(size_t max_reports)
{
    config_.max_leak_reports = max_reports;
    return *this;
}

memory_report_builder& memory_report_builder::with_file_export(const std::string& filename)
{
    config_.export_to_file  = true;
    config_.export_filename = filename;
    return *this;
}

std::string memory_report_builder::build() const
{
    if (allocator_ == nullptr)
    {
        return "Error: No allocator specified for report generation.\n";
    }

    allocator_report_generator const generator;
    return generator.generate_comprehensive_report(*allocator_, config_);
}

}  // namespace xsigma
