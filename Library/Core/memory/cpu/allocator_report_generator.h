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

#pragma once

#include <string>
#include <vector>

#include "common/macros.h"
#include "memory/cpu/allocator.h"
#include "memory/cpu/allocator_tracking.h"
#include "memory/unified_memory_stats.h"
#include "memory/visualization/ascii_visualizer.h"

namespace xsigma
{

/**
 * @brief Configuration for memory tracking report generation
 */
struct XSIGMA_VISIBILITY report_config
{
    bool        include_leak_detection{true};         ///< Include memory leak analysis
    bool        include_memory_graphs{true};          ///< Include ASCII memory usage graphs
    bool        include_peak_analysis{true};          ///< Include peak memory usage analysis
    bool        include_allocation_patterns{true};    ///< Include allocation/deallocation patterns
    bool        include_fragmentation{true};          ///< Include fragmentation analysis
    bool        include_timing_stats{true};           ///< Include performance timing statistics
    bool        include_recommendations{true};        ///< Include optimization recommendations
    bool        include_detailed_allocations{false};  ///< Include individual allocation details
    size_t      max_leak_reports{10};                 ///< Maximum number of leaks to report
    size_t      max_allocation_details{20};           ///< Maximum allocation details to show
    bool        export_to_file{false};                ///< Whether to export report to file
    std::string export_filename;                      ///< Filename for export (if enabled)
};

/**
 * @brief Memory leak information
 */
struct XSIGMA_VISIBILITY memory_leak_info
{
    void*       address{nullptr};        ///< Address of leaked memory
    size_t      size{0};                 ///< Size of leaked allocation
    int64_t     allocation_id{0};        ///< Unique allocation identifier
    int64_t     allocation_time_us{0};   ///< Time when allocated (microseconds)
    int64_t     age_ms{0};               ///< Age of allocation (milliseconds)
    std::string tag;                     ///< Optional allocation tag
    const char* source_file{nullptr};    ///< Source file where allocated
    int         source_line{0};          ///< Source line number
    const char* function_name{nullptr};  ///< Function name where allocated
};

/**
 * @brief Comprehensive memory tracking report generator
 *
 * Generates detailed, human-readable reports for memory allocator tracking
 * systems. Provides leak detection, usage analysis, performance metrics,
 * and optimization recommendations.
 *
 * **Thread Safety**: Not thread-safe - external synchronization required
 * **Performance**: O(n) where n is number of allocations tracked
 *
 * **Example Usage**:
 * ```cpp
 * allocator_report_generator generator;
 * report_config config;
 * config.include_leak_detection = true;
 * config.include_recommendations = true;
 *
 * std::string report = generator.generate_comprehensive_report(
 *     tracking_allocator, config);
 * std::cout << report << std::endl;
 * ```
 */
class XSIGMA_VISIBILITY allocator_report_generator
{
public:
    /**
     * @brief Generate comprehensive memory tracking report
     *
     * @param allocator Tracking allocator to analyze
     * @param config Report configuration options
     * @return Formatted report string
     *
     * **Performance**: O(n) where n is number of tracked allocations
     * **Thread Safety**: Not thread-safe
     */
    XSIGMA_API std::string generate_comprehensive_report(
        const allocator_tracking& allocator, const report_config& config = report_config{}) const;

    /**
     * @brief Detect memory leaks in tracked allocations
     *
     * @param allocator Tracking allocator to analyze
     * @param leak_threshold_ms Age threshold for leak detection (milliseconds)
     * @return Vector of detected memory leaks
     *
     * **Algorithm**: Identifies allocations older than threshold
     * **Performance**: O(n) where n is number of active allocations
     */
    static XSIGMA_API std::vector<memory_leak_info> detect_leaks(
        const allocator_tracking& allocator, int64_t leak_threshold_ms = 60000) ;

    /**
     * @brief Generate memory usage timeline visualization
     *
     * @param records Allocation records to visualize
     * @return ASCII timeline visualization
     *
     * **Performance**: O(n) where n is number of records
     */
    XSIGMA_API std::string generate_memory_timeline(
        const std::vector<enhanced_alloc_record>& records) const;

    /**
     * @brief Generate allocation size distribution histogram
     *
     * @param records Allocation records to analyze
     * @return ASCII histogram visualization
     *
     * **Performance**: O(n) where n is number of records
     */
    XSIGMA_API std::string generate_size_distribution(
        const std::vector<enhanced_alloc_record>& records) const;

    /**
     * @brief Generate performance analysis report
     *
     * @param timing_stats Timing statistics to analyze
     * @return Formatted performance report
     *
     * **Performance**: O(1) - simple statistics formatting
     */
    static XSIGMA_API std::string generate_performance_report(
        const atomic_timing_stats& timing_stats) ;

    /**
     * @brief Generate optimization recommendations
     *
     * @param allocator Tracking allocator to analyze
     * @return Formatted recommendations
     *
     * **Performance**: O(1) - heuristic-based analysis
     */
    static XSIGMA_API std::string generate_recommendations(const allocator_tracking& allocator) ;

    /**
     * @brief Generate fragmentation analysis report
     *
     * @param metrics Fragmentation metrics to analyze
     * @return Formatted fragmentation report
     *
     * **Performance**: O(1) - metrics formatting
     */
    static XSIGMA_API std::string generate_fragmentation_report(
        const memory_fragmentation_metrics& metrics) ;

    /**
     * @brief Export report to file
     *
     * @param report Report content to export
     * @param filename Output filename
     * @return true if export successful, false otherwise
     *
     * **Performance**: O(n) where n is report size
     * **Thread Safety**: Not thread-safe
     */
    static XSIGMA_API bool export_report(const std::string& report, const std::string& filename) ;

private:
    /**
     * @brief Format bytes in human-readable format
     */
    static std::string format_bytes(size_t bytes) ;

    /**
     * @brief Format duration in human-readable format
     */
    static std::string format_duration(int64_t microseconds) ;

    /**
     * @brief Generate report header
     */
    static std::string generate_header(const std::string& title) ;

    /**
     * @brief Generate report section separator
     */
    static std::string generate_separator() ;

    /**
     * @brief Analyze allocation patterns
     */
    static std::string analyze_allocation_patterns(
        const std::vector<enhanced_alloc_record>& records) ;

    /**
     * @brief Calculate memory efficiency score
     */
    double calculate_efficiency_score(const allocator_tracking& allocator) const;

    /**
     * @brief Detect allocation hotspots
     */
    std::string detect_hotspots(const std::vector<enhanced_alloc_record>& records) const;

    /**
     * @brief Generate leak report section
     */
    static std::string generate_leak_report(
        const std::vector<memory_leak_info>& leaks, size_t max_reports) ;

    /**
     * @brief Generate summary statistics
     */
    static std::string generate_summary_statistics(const allocator_tracking& allocator) ;

    /**
     * @brief ASCII visualizer for graphs
     */
    mutable ascii_visualizer visualizer_;
};

/**
 * @brief Memory tracking report builder with fluent interface
 *
 * Provides a convenient builder pattern for constructing memory tracking
 * reports with customizable sections and formatting.
 *
 * **Example Usage**:
 * ```cpp
 * memory_report_builder builder;
 * std::string report = builder
 *     .with_allocator(tracking_alloc)
 *     .with_leak_detection(true)
 *     .with_performance_analysis(true)
 *     .with_recommendations(true)
 *     .build();
 * ```
 */
class XSIGMA_VISIBILITY memory_report_builder
{
public:
    /**
     * @brief Construct report builder
     */
    XSIGMA_API memory_report_builder();

    /**
     * @brief Set allocator to analyze
     */
    XSIGMA_API memory_report_builder& with_allocator(const allocator_tracking* allocator);

    /**
     * @brief Enable leak detection
     */
    XSIGMA_API memory_report_builder& with_leak_detection(bool enable = true);

    /**
     * @brief Enable memory graphs
     */
    XSIGMA_API memory_report_builder& with_memory_graphs(bool enable = true);

    /**
     * @brief Enable performance analysis
     */
    XSIGMA_API memory_report_builder& with_performance_analysis(bool enable = true);

    /**
     * @brief Enable fragmentation analysis
     */
    XSIGMA_API memory_report_builder& with_fragmentation_analysis(bool enable = true);

    /**
     * @brief Enable recommendations
     */
    XSIGMA_API memory_report_builder& with_recommendations(bool enable = true);

    /**
     * @brief Enable detailed allocation listing
     */
    XSIGMA_API memory_report_builder& with_detailed_allocations(bool enable = true);

    /**
     * @brief Set leak detection threshold
     */
    XSIGMA_API memory_report_builder& with_leak_threshold_ms(int64_t threshold_ms);

    /**
     * @brief Set maximum number of leaks to report
     */
    XSIGMA_API memory_report_builder& with_max_leak_reports(size_t max_reports);

    /**
     * @brief Enable file export
     */
    XSIGMA_API memory_report_builder& with_file_export(const std::string& filename);

    /**
     * @brief Build and return report
     */
    XSIGMA_API std::string build() const;

private:
    const allocator_tracking* allocator_{nullptr};
    report_config             config_;
    int64_t                   leak_threshold_ms_{60000};
};

}  // namespace xsigma
