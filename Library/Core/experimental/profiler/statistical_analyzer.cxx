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

#include "statistical_analyzer.h"

// Include hash compatibility layer for libc++ versions that don't export __hash_memory

#include "util/flat_hash.h"

// Prevent Windows min/max macros from interfering
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace xsigma
{

//=============================================================================
// statistical_metrics Implementation
//=============================================================================

void statistical_metrics::reset()
{
    min_value     = (std::numeric_limits<double>::max)();
    max_value     = (std::numeric_limits<double>::lowest)();
    mean          = 0.0;
    median        = 0.0;
    std_deviation = 0.0;
    variance      = 0.0;
    sum           = 0.0;
    count         = 0;
    percentiles.clear();
    outliers.clear();
}

//=============================================================================
// statistical_analyzer Implementation
//=============================================================================

statistical_analyzer::statistical_analyzer() = default;

statistical_analyzer::~statistical_analyzer()
{
    if (analyzing_.load())
    {
        stop_analysis();
    }
}

void statistical_analyzer::start_analysis()
{
    if (analyzing_.exchange(true))
    {
        return;  // Already analyzing
    }

    // Clear existing data
    clear_data();
}

void statistical_analyzer::stop_analysis()
{
    if (!analyzing_.exchange(false))
    {
        return;  // Not analyzing
    }
}

void statistical_analyzer::add_timing_sample(const std::string& name, double time_ms)
{
    if (!analyzing_.load())
        return;

    std::lock_guard<std::mutex> lock(timing_mutex_);
    auto&                       series = timing_data_[name];
    series.push_back(time_ms);
    trim_series_if_needed(series);
}

void statistical_analyzer::add_memory_sample(const std::string& name, size_t memory_bytes)
{
    if (!analyzing_.load())
        return;

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto&                       series = memory_data_[name];
    series.push_back(static_cast<double>(memory_bytes));
    trim_series_if_needed(series);
}

void statistical_analyzer::add_custom_sample(const std::string& name, double value)
{
    if (!analyzing_.load())
        return;

    std::lock_guard<std::mutex> lock(custom_mutex_);
    auto&                       series = custom_data_[name];
    series.push_back(value);
    trim_series_if_needed(series);
}

void statistical_analyzer::add_time_series_point(
    const std::string& series_name, double value, const std::string& label)
{
    if (!analyzing_.load())
        return;

    xsigma::time_series_point point;
    point.timestamp_ = std::chrono::high_resolution_clock::now();
    point.value_     = value;
    point.label_     = label;
    point.thread_id_ = std::this_thread::get_id();

    std::lock_guard<std::mutex> lock(time_series_mutex_);
    auto&                       series = time_series_data_[series_name];
    series.push_back(point);
    trim_time_series_if_needed(series);
}

xsigma::statistical_metrics statistical_analyzer::calculate_timing_stats(
    const std::string& name) const
{
    std::lock_guard<std::mutex> lock(timing_mutex_);
    auto                        it = timing_data_.find(name);
    if (it == timing_data_.end())
    {
        return xsigma::statistical_metrics{};
    }
    return calculate_metrics(it->second);
}

xsigma::statistical_metrics statistical_analyzer::calculate_memory_stats(
    const std::string& name) const
{
    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto                        it = memory_data_.find(name);
    if (it == memory_data_.end())
    {
        return xsigma::statistical_metrics{};
    }
    return calculate_metrics(it->second);
}

xsigma::statistical_metrics statistical_analyzer::calculate_custom_stats(
    const std::string& name) const
{
    std::lock_guard<std::mutex> lock(custom_mutex_);
    auto                        it = custom_data_.find(name);
    if (it == custom_data_.end())
    {
        return xsigma::statistical_metrics{};
    }
    return calculate_metrics(it->second);
}

xsigma_map<std::string, xsigma::statistical_metrics>
statistical_analyzer::calculate_all_timing_stats() const
{
    std::lock_guard<std::mutex>                          lock(timing_mutex_);
    xsigma_map<std::string, xsigma::statistical_metrics> results;

    for (const auto& pair : timing_data_)
    {
        results[pair.first] = calculate_metrics(pair.second);
    }

    return results;
}

xsigma_map<std::string, xsigma::statistical_metrics>
statistical_analyzer::calculate_all_memory_stats() const
{
    std::lock_guard<std::mutex>                          lock(memory_mutex_);
    xsigma_map<std::string, xsigma::statistical_metrics> results;

    for (const auto& pair : memory_data_)
    {
        results[pair.first] = calculate_metrics(pair.second);
    }

    return results;
}

xsigma_map<std::string, xsigma::statistical_metrics>
statistical_analyzer::calculate_all_custom_stats() const
{
    std::lock_guard<std::mutex>                          lock(custom_mutex_);
    xsigma_map<std::string, xsigma::statistical_metrics> results;

    for (const auto& pair : custom_data_)
    {
        results[pair.first] = calculate_metrics(pair.second);
    }

    return results;
}

std::vector<xsigma::time_series_point> statistical_analyzer::get_time_series(
    const std::string& series_name) const
{
    std::lock_guard<std::mutex> lock(time_series_mutex_);
    auto                        it = time_series_data_.find(series_name);

    if (it == time_series_data_.end())
    {
        return {};
    }
    return it->second;
}

xsigma::statistical_metrics statistical_analyzer::analyze_time_series(
    const std::string& series_name) const
{
    std::lock_guard<std::mutex> lock(time_series_mutex_);
    auto                        it = time_series_data_.find(series_name);
    if (it == time_series_data_.end())
    {
        return xsigma::statistical_metrics{};
    }

    std::vector<double> values;
    values.reserve(it->second.size());
    for (const auto& point : it->second)
    {
        values.push_back(point.value_);
    }

    return calculate_metrics(values);
}

double statistical_analyzer::calculate_trend_slope(const std::string& series_name) const
{
    std::lock_guard<std::mutex> lock(time_series_mutex_);
    auto                        it = time_series_data_.find(series_name);
    if (it == time_series_data_.end() || it->second.size() < 2)
    {
        return 0.0;
    }

    const auto& series = it->second;
    size_t      n      = series.size();

    // Calculate linear regression slope using least squares
    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        double x = static_cast<double>(i);
        double y = series[i].value_;

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

bool statistical_analyzer::is_trending_up(const std::string& series_name, double threshold) const
{
    double slope = calculate_trend_slope(series_name);
    return slope > threshold;
}

bool statistical_analyzer::is_trending_down(const std::string& series_name, double threshold) const
{
    double slope = calculate_trend_slope(series_name);
    return slope < -threshold;
}

double statistical_analyzer::calculate_correlation(
    const std::string& series1, const std::string& series2) const
{
    std::lock_guard<std::mutex> lock(time_series_mutex_);

    auto it1 = time_series_data_.find(series1);
    auto it2 = time_series_data_.find(series2);

    if (it1 == time_series_data_.end() || it2 == time_series_data_.end())
    {
        return 0.0;
    }

    const auto& s1 = it1->second;
    const auto& s2 = it2->second;

    size_t min_size = std::min(s1.size(), s2.size());
    if (min_size < 2)
        return 0.0;

    // Calculate Pearson correlation coefficient
    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;

    for (size_t i = 0; i < min_size; ++i)
    {
        double x = s1[i].value_;
        double y = s2[i].value_;

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    double n           = static_cast<double>(min_size);
    double numerator   = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

    return (denominator != 0.0) ? numerator / denominator : 0.0;
}

bool statistical_analyzer::detect_performance_regression(
    const std::string& name, double baseline_mean, double threshold) const
{
    auto stats = calculate_timing_stats(name);
    if (!stats.is_valid())
        return false;

    double performance_change = (stats.mean - baseline_mean) / baseline_mean;
    return performance_change > threshold;
}

void statistical_analyzer::clear_data()
{
    {
        std::lock_guard<std::mutex> lock(timing_mutex_);
        timing_data_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        memory_data_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(custom_mutex_);
        custom_data_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(time_series_mutex_);
        time_series_data_.clear();
    }
}

void statistical_analyzer::clear_series(const std::string& name)
{
    {
        std::lock_guard<std::mutex> lock(timing_mutex_);
        timing_data_.erase(name);
    }
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        memory_data_.erase(name);
    }
    {
        std::lock_guard<std::mutex> lock(custom_mutex_);
        custom_data_.erase(name);
    }
    {
        std::lock_guard<std::mutex> lock(time_series_mutex_);
        time_series_data_.erase(name);
    }
}

size_t statistical_analyzer::get_sample_count(const std::string& name) const
{
    {
        std::lock_guard<std::mutex> lock(timing_mutex_);
        auto                        it = timing_data_.find(name);

        if (it != timing_data_.end())
        {
            return it->second.size();
        }
    }
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        auto                        it = memory_data_.find(name);

        if (it != memory_data_.end())
        {
            return it->second.size();
        }
    }
    {
        std::lock_guard<std::mutex> lock(custom_mutex_);
        auto                        it = custom_data_.find(name);

        if (it != custom_data_.end())
        {
            return it->second.size();
        }
    }
    return 0;
}

void statistical_analyzer::set_max_samples_per_series(size_t max_samples)
{
    max_samples_per_series_ = max_samples;
}

void statistical_analyzer::set_outlier_threshold(double threshold)
{
    outlier_threshold_ = threshold;
}

void statistical_analyzer::set_percentiles(const std::vector<double>& percentiles)
{
    percentiles_ = percentiles;
}

xsigma::statistical_metrics statistical_analyzer::calculate_metrics(
    const std::vector<double>& data) const
{
    xsigma::statistical_metrics metrics;

    if (data.empty())
    {
        return metrics;
    }

    metrics.count = data.size();

    // Calculate basic statistics
    metrics.sum  = std::accumulate(data.begin(), data.end(), 0.0);
    metrics.mean = metrics.sum / metrics.count;

    metrics.min_value = *std::min_element(data.begin(), data.end());
    metrics.max_value = *std::max_element(data.begin(), data.end());

    // Calculate variance and standard deviation
    double variance_sum = 0.0;

    for (double value : data)
    {
        double diff = value - metrics.mean;
        variance_sum += diff * diff;
    }
    metrics.variance      = variance_sum / metrics.count;
    metrics.std_deviation = std::sqrt(metrics.variance);

    // Calculate median and percentiles
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    size_t mid = sorted_data.size() / 2;

    if (sorted_data.size() % 2 == 0)
    {
        metrics.median = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0;
    }
    else
    {
        metrics.median = sorted_data[mid];
    }

    // Calculate percentiles
    metrics.percentiles = calculate_percentiles(sorted_data, percentiles_);

    // Detect outliers
    metrics.outliers          = detect_outliers(data, outlier_threshold_);
    metrics.outlier_threshold = outlier_threshold_;

    return metrics;
}

std::vector<double> statistical_analyzer::calculate_percentiles(
    std::vector<double> data, const std::vector<double>& percentiles) const
{
    if (data.empty() || percentiles.empty())
    {
        return {};
    }

    std::sort(data.begin(), data.end());
    std::vector<double> results;
    results.reserve(percentiles.size());

    for (double p : percentiles)
    {
        if (p < 0.0 || p > 100.0)
            continue;

        double index = (p / 100.0) * (data.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));

        if (lower == upper)
        {
            results.push_back(data[lower]);
        }
        else
        {
            double weight = index - lower;
            results.push_back(data[lower] * (1.0 - weight) + data[upper] * weight);
        }
    }

    return results;
}

std::vector<double> statistical_analyzer::detect_outliers(
    const std::vector<double>& data, double threshold) const
{
    if (data.size() < 3)
    {
        return {};  // Need at least 3 points for meaningful outlier detection
    }

    // Calculate mean and standard deviation
    double sum  = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();

    double variance_sum = 0.0;
    for (double value : data)
    {
        double diff = value - mean;
        variance_sum += diff * diff;
    }
    double std_dev = std::sqrt(variance_sum / data.size());

    // Find outliers using z-score method
    std::vector<double> outliers;
    for (double value : data)
    {
        double z_score = std::abs(value - mean) / std_dev;
        if (z_score > threshold)
        {
            outliers.push_back(value);
        }
    }

    return outliers;
}

void statistical_analyzer::trim_series_if_needed(std::vector<double>& series) const
{
    if (series.size() > max_samples_per_series_)
    {
        // Remove oldest samples (from the beginning)
        size_t excess = series.size() - max_samples_per_series_;
        series.erase(series.begin(), series.begin() + excess);
    }
}

void statistical_analyzer::trim_time_series_if_needed(
    std::vector<xsigma::time_series_point>& series) const
{
    if (series.size() > max_samples_per_series_)
    {
        // Remove oldest samples (from the beginning)
        size_t excess = series.size() - max_samples_per_series_;
        series.erase(series.begin(), series.begin() + excess);
    }
}

//=============================================================================
// StatisticalAnalysisScope Implementation
//=============================================================================

statistical_analysis_scope::statistical_analysis_scope(
    xsigma::statistical_analyzer& analyzer, const std::string& name)
    : analyzer_(analyzer), name_(name)
{
    start_time_ = std::chrono::high_resolution_clock::now();
}

statistical_analysis_scope::~statistical_analysis_scope()
{
    if (active_)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        double duration_ms = duration.count() / 1000.0;

        analyzer_.add_timing_sample(name_, duration_ms);
    }
}

void statistical_analysis_scope::add_checkpoint(const std::string& label)
{
    auto checkpoint_time = std::chrono::high_resolution_clock::now();
    checkpoints_.emplace_back(label, checkpoint_time);

    // Add timing sample for this checkpoint
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(checkpoint_time - start_time_);
    double duration_ms = duration.count() / 1000.0;
    analyzer_.add_timing_sample(name_ + "_" + label, duration_ms);
}

xsigma::statistical_metrics statistical_analysis_scope::get_checkpoint_stats() const
{
    if (checkpoints_.empty())
    {
        return xsigma::statistical_metrics{};
    }

    std::vector<double> checkpoint_times;
    checkpoint_times.reserve(checkpoints_.size());

    for (const auto& checkpoint : checkpoints_)
    {
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(checkpoint.second - start_time_);
        double duration_ms = duration.count() / 1000.0;
        checkpoint_times.push_back(duration_ms);
    }

    return analyzer_.calculate_metrics(checkpoint_times);
}

}  // namespace xsigma
