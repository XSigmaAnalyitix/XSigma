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

#include "memory/cpu/allocator_selector.h"

#include <algorithm>
#include <mutex>
#include <sstream>

#include "common/pointer.h"
#include "logging/logger.h"
#include "memory/cpu/allocator_bfc.h"
#include "memory/cpu/allocator_cpu.h"
#include "memory/cpu/allocator_pool.h"
#include "memory/cpu/allocator_tracking.h"

namespace xsigma
{

// ============================================================================
// allocator_selector Implementation
// ============================================================================

allocator_selector::recommendation allocator_selector::recommend(const allocation_context& ctx)
{
    recommendation rec;

    // Priority 1: Tracking for development/debugging
    if (ctx.require_tracking)
    {
        rec.allocator_type = "allocator_tracking";
        rec.rationale      = "Tracking required for debugging and profiling";
        rec.configuration  = generate_configuration("allocator_tracking", ctx);
        rec.confidence     = 1.0;
        return rec;
    }

    // Priority 2: Pool allocator for high-frequency, predictable sizes
    if (is_pool_suitable(ctx))
    {
        rec.allocator_type = "allocator_pool";
        rec.rationale = "High-frequency allocations with predictable sizes benefit from pooling";
        rec.configuration = generate_configuration("allocator_pool", ctx);
        rec.confidence    = calculate_confidence("allocator_pool", ctx);
        return rec;
    }

    // Priority 3: BFC allocator for large allocations or memory constraints
    if (is_bfc_suitable(ctx))
    {
        rec.allocator_type = "allocator_bfc";
        rec.rationale      = "Large allocations or memory constraints benefit from BFC coalescing";
        rec.configuration  = generate_configuration("allocator_bfc", ctx);
        rec.confidence     = calculate_confidence("allocator_bfc", ctx);
        return rec;
    }

    // Default: CPU allocator for general-purpose use
    rec.allocator_type = "allocator_cpu";
    rec.rationale      = "General-purpose allocator suitable for unpredictable patterns";
    rec.configuration  = generate_configuration("allocator_cpu", ctx);
    rec.confidence     = calculate_confidence("allocator_cpu", ctx);
    return rec;
}

std::string allocator_selector::analyze_context(const allocation_context& ctx)
{
    std::ostringstream analysis;

    analysis << "=== Allocation Context Analysis ===\n";
    analysis << "Allocation Size: " << ctx.allocation_size << " bytes\n";
    analysis << "Size Range: " << ctx.min_size << " - " << ctx.max_size << " bytes\n";
    analysis << "Estimated Frequency: " << ctx.estimated_frequency << " allocs/sec\n";
    analysis << "Estimated Lifetime: " << ctx.estimated_lifetime_ms << " ms\n";
    analysis << "Thread Count: " << ctx.thread_count << "\n";
    analysis << "Size Predictable: " << (ctx.size_predictable ? "Yes" : "No") << "\n";
    analysis << "Lifetime Predictable: " << (ctx.lifetime_predictable ? "Yes" : "No") << "\n";
    analysis << "Memory Constrained: " << (ctx.memory_constrained ? "Yes" : "No") << "\n";
    analysis << "Require Tracking: " << (ctx.require_tracking ? "Yes" : "No") << "\n";

    // Categorize allocation size
    analysis << "\nSize Category: ";
    if (ctx.allocation_size < 64)
    {
        analysis << "Very Small (<64 bytes)\n";
    }
    else if (ctx.allocation_size < 1024ULL)
    {
        analysis << "Small (64 bytes - 1 KB)\n";
    }
    else if (ctx.allocation_size < 4096)
    {
        analysis << "Medium (1 KB - 4 KB)\n";
    }
    else if (ctx.allocation_size < 1024ULL * 1024ULL)
    {
        analysis << "Large (4 KB - 1 MB)\n";
    }
    else
    {
        analysis << "Very Large (>1 MB)\n";
    }

    // Categorize frequency
    analysis << "Frequency Category: ";
    if (ctx.estimated_frequency > 10000)
    {
        analysis << "Very High (>10K/sec)\n";
    }
    else if (ctx.estimated_frequency > 1000)
    {
        analysis << "High (1K-10K/sec)\n";
    }
    else if (ctx.estimated_frequency > 100)
    {
        analysis << "Moderate (100-1K/sec)\n";
    }
    else
    {
        analysis << "Low (<100/sec)\n";
    }

    // Get recommendation
    auto rec = recommend(ctx);
    analysis << "\n=== Recommendation ===\n";
    analysis << "Allocator: " << rec.allocator_type << "\n";
    analysis << "Rationale: " << rec.rationale << "\n";
    analysis << "Confidence: " << (rec.confidence * 100.0) << "%\n";
    analysis << "Configuration:\n" << rec.configuration << "\n";

    return analysis.str();
}

std::vector<allocator_selector::recommendation> allocator_selector::compare_allocators(
    const allocation_context& ctx)
{
    std::vector<recommendation> recommendations;

    // Evaluate each allocator type
    const std::string allocator_types[] = {
        "allocator_cpu", "allocator_bfc", "allocator_pool", "allocator_tracking"};

    for (const auto& type : allocator_types)
    {
        recommendation rec;
        rec.allocator_type = type;
        rec.confidence     = calculate_confidence(type, ctx);
        rec.configuration  = generate_configuration(type, ctx);

        // Generate rationale based on type
        if (type == "allocator_cpu")
        {
            rec.rationale = "General-purpose, minimal overhead, excellent thread scaling";
        }
        else if (type == "allocator_bfc")
        {
            rec.rationale = "Best-fit coalescing, good for large allocations and fragmentation";
        }
        else if (type == "allocator_pool")
        {
            rec.rationale = "LRU pooling, excellent for repeated similar-sized allocations";
        }
        else if (type == "allocator_tracking")
        {
            rec.rationale = "Comprehensive tracking and debugging capabilities";
        }

        recommendations.push_back(rec);
    }

    // Sort by confidence (descending)
    std::sort(
        recommendations.begin(),
        recommendations.end(),
        [](const recommendation& a, const recommendation& b)
        { return a.confidence > b.confidence; });

    return recommendations;
}

std::string allocator_selector::validate_choice(
    const std::string& allocator_type, const allocation_context& ctx)
{
    std::ostringstream validation;

    validation << "=== Allocator Choice Validation ===\n";
    validation << "Chosen Allocator: " << allocator_type << "\n";

    auto rec = recommend(ctx);
    validation << "Recommended Allocator: " << rec.allocator_type << "\n";

    if (allocator_type == rec.allocator_type)
    {
        validation << "Status: OPTIMAL - Choice matches recommendation\n";
    }
    else
    {
        double const chosen_confidence = calculate_confidence(allocator_type, ctx);
        validation << "Status: SUBOPTIMAL - Consider " << rec.allocator_type << "\n";
        validation << "Chosen Confidence: " << (chosen_confidence * 100.0) << "%\n";
        validation << "Recommended Confidence: " << (rec.confidence * 100.0) << "%\n";
    }

    // Specific warnings
    if (allocator_type == "allocator_pool" && !ctx.size_predictable)
    {
        validation << "WARNING: Pool allocator with unpredictable sizes may cause fragmentation\n";
    }

    if (allocator_type == "allocator_bfc" && ctx.estimated_frequency > 10000)
    {
        validation << "WARNING: BFC allocator may have high overhead for very high frequency\n";
    }

    if (allocator_type == "allocator_tracking" && !ctx.require_tracking)
    {
        validation << "WARNING: Tracking allocator adds overhead when not needed\n";
    }

    if (allocator_type == "allocator_cpu" && ctx.memory_constrained)
    {
        validation << "WARNING: CPU allocator may not handle memory pressure optimally\n";
    }

    return validation.str();
}

double allocator_selector::calculate_confidence(
    const std::string& allocator_type, const allocation_context& ctx)
{
    double confidence = 0.5;  // Base confidence

    if (allocator_type == "allocator_pool")
    {
        // High confidence for predictable, high-frequency, small allocations
        if (ctx.size_predictable)
        {
            confidence += 0.2;
        }
        if (ctx.estimated_frequency > 5000)
        {
            confidence += 0.2;
        }
        if (ctx.allocation_size < 4096)
        {
            confidence += 0.1;
        }
    }
    else if (allocator_type == "allocator_bfc")
    {
        // High confidence for large allocations or memory constraints
        if (ctx.allocation_size > 4096)
        {
            confidence += 0.2;
        }
        if (ctx.memory_constrained)
        {
            confidence += 0.2;
        }
        if (ctx.lifetime_predictable)
        {
            confidence += 0.1;
        }
    }
    else if (allocator_type == "allocator_cpu")
    {
        // High confidence for unpredictable patterns or high thread count
        if (!ctx.size_predictable || !ctx.lifetime_predictable)
        {
            confidence += 0.2;
        }
        if (ctx.thread_count > 8)
        {
            confidence += 0.2;
        }
        if (ctx.allocation_size < 64)
        {
            confidence += 0.1;
        }
    }
    else if (allocator_type == "allocator_tracking")
    {
        // High confidence only when tracking is required
        if (ctx.require_tracking)
        {
            confidence = 1.0;
        }
        else
        {
            confidence = 0.3;  // Low confidence when not needed
        }
    }

    return std::min(1.0, confidence);
}

std::string allocator_selector::generate_configuration(
    const std::string& allocator_type, const allocation_context& ctx)
{
    std::ostringstream config;

    if (allocator_type == "allocator_pool")
    {
        size_t pool_size = std::max(size_t(10), ctx.estimated_frequency / 100);
        pool_size        = std::min(pool_size, size_t(500));

        config << "pool_size_limit: " << pool_size << "\n";
        config << "auto_resize: " << (ctx.size_predictable ? "false" : "true") << "\n";
        config << "size_rounder: power_of_2_rounder\n";
    }
    else if (allocator_type == "allocator_bfc")
    {
        size_t const pool_size = std::max<size_t>(ctx.allocation_size * 100, 10ULL * 1024ULL * 1024ULL);

        config << "total_memory: " << pool_size << " bytes\n";
        config << "allow_growth: " << (!ctx.memory_constrained ? "true" : "false") << "\n";
        config << "garbage_collection: " << (ctx.memory_constrained ? "true" : "false") << "\n";
        config << "allow_retry_on_failure: true\n";
    }
    else if (allocator_type == "allocator_cpu")
    {
        config << "numa_node: NUMANOAFFINITY\n";
        config << "enable_stats: true\n";
    }
    else if (allocator_type == "allocator_tracking")
    {
        config << "track_sizes_locally: true\n";
        config << "enable_enhanced_tracking: true\n";
        config << "logging_level: INFO\n";
    }

    return config.str();
}

bool allocator_selector::is_pool_suitable(const allocation_context& ctx)
{
    // Pool is suitable for high-frequency, predictable-size allocations
    return ctx.size_predictable && ctx.estimated_frequency > 1000 && ctx.allocation_size < 65536;
}

bool allocator_selector::is_bfc_suitable(const allocation_context& ctx)
{
    // BFC is suitable for large allocations or memory-constrained environments
    return ctx.allocation_size > 4096 || ctx.memory_constrained;
}

bool allocator_selector::is_cpu_suitable(XSIGMA_UNUSED const allocation_context& ctx)
{
    // CPU allocator is always suitable as a fallback
    return true;
}

// ============================================================================
// adaptive_allocator_manager Implementation
// ============================================================================

struct adaptive_allocator_manager::impl
{
    std::mutex                      mutex_;
    Allocator*                      cpu_allocator_{nullptr};
    std::unique_ptr<allocator_pool> pool_allocator_;
    std::unique_ptr<allocator_bfc>  bfc_allocator_;
    bool                            initialized_{false};
};

adaptive_allocator_manager::adaptive_allocator_manager() : pimpl_(std::make_unique<impl>()) {}

adaptive_allocator_manager::~adaptive_allocator_manager() = default;

void adaptive_allocator_manager::initialize(
    bool enable_pool, bool enable_bfc, XSIGMA_UNUSED bool enable_tracking)
{
    std::lock_guard<std::mutex> const lock(pimpl_->mutex_);

    if (pimpl_->initialized_)
    {
        XSIGMA_LOG_WARNING("adaptive_allocator_manager already initialized");
        return;
    }

    // Initialize CPU allocator
    EnableCPUAllocatorStats();
    pimpl_->cpu_allocator_ = cpu_allocator(0);

    // Initialize pool allocator if requested
    if (enable_pool)
    {
        auto sub_alloc = util::make_ptr_unique_mutable<basic_cpu_allocator>(
            0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

        pimpl_->pool_allocator_ = std::make_unique<allocator_pool>(
            100,   // pool_size_limit
            true,  // auto_resize
            std::move(sub_alloc),
            util::make_ptr_unique_mutable<NoopRounder>(),
            "adaptive_pool");
    }

    // Initialize BFC allocator if requested
    if (enable_bfc)
    {
        auto sub_alloc = std::make_unique<basic_cpu_allocator>(
            0, std::vector<sub_allocator::Visitor>{}, std::vector<sub_allocator::Visitor>{});

        allocator_bfc::Options opts;
        opts.allow_growth       = true;
        opts.garbage_collection = true;

        pimpl_->bfc_allocator_ = std::make_unique<allocator_bfc>(
            std::move(sub_alloc), 100ULL * 1024ULL * 1024ULL, "adaptive_bfc", opts);
    }

    pimpl_->initialized_ = true;

    XSIGMA_LOG_INFO("adaptive_allocator_manager initialized successfully");
}

Allocator* adaptive_allocator_manager::get_allocator(const allocation_context& ctx)
{
    std::lock_guard<std::mutex> const lock(pimpl_->mutex_);

    if (!pimpl_->initialized_)
    {
        XSIGMA_LOG_ERROR("adaptive_allocator_manager not initialized");
        return nullptr;
    }

    auto rec = allocator_selector::recommend(ctx);

    if (rec.allocator_type == "allocator_pool" && pimpl_->pool_allocator_)
    {
        return pimpl_->pool_allocator_.get();
    }
    if (rec.allocator_type == "allocator_bfc" && pimpl_->bfc_allocator_)
    {
        return pimpl_->bfc_allocator_.get();
    }

    return pimpl_->cpu_allocator_;
}

std::string adaptive_allocator_manager::generate_report() const
{
    std::lock_guard<std::mutex> const lock(pimpl_->mutex_);

    std::ostringstream report;
    report << "=== Adaptive Allocator Manager Report ===\n";

    // CPU allocator stats
    if (pimpl_->cpu_allocator_ != nullptr)
    {
        auto stats = pimpl_->cpu_allocator_->GetStats();
        if (stats.has_value())
        {
            report << "\nCPU Allocator:\n";
            report << "  Allocations: " << stats->num_allocs.load() << "\n";
            report << "  Peak Memory: " << stats->peak_bytes_in_use.load() << " bytes\n";
        }
    }

    // Pool allocator stats
    if (pimpl_->pool_allocator_)
    {
        auto stats = pimpl_->pool_allocator_->GetStats();
        if (stats.has_value())
        {
            report << "\nPool Allocator:\n";
            report << "  Allocations: " << stats->num_allocs.load() << "\n";
            report << "  Peak Memory: " << stats->peak_bytes_in_use.load() << " bytes\n";
        }
    }

    // BFC allocator stats
    if (pimpl_->bfc_allocator_)
    {
        auto stats = pimpl_->bfc_allocator_->GetStats();
        if (stats.has_value())
        {
            report << "\nBFC Allocator:\n";
            report << "  Allocations: " << stats->num_allocs.load() << "\n";
            report << "  Peak Memory: " << stats->peak_bytes_in_use.load() << " bytes\n";
        }
    }

    return report.str();
}

std::optional<allocator_stats> adaptive_allocator_manager::get_stats(
    const std::string& allocator_type) const
{
    std::lock_guard<std::mutex> const lock(pimpl_->mutex_);

    if (allocator_type == "allocator_cpu" && (pimpl_->cpu_allocator_ != nullptr))
    {
        return pimpl_->cpu_allocator_->GetStats();
    }
    if (allocator_type == "allocator_pool" && pimpl_->pool_allocator_)
    {
        return pimpl_->pool_allocator_->GetStats();
    }
    if (allocator_type == "allocator_bfc" && pimpl_->bfc_allocator_)
    {
        return pimpl_->bfc_allocator_->GetStats();
    }

    return std::nullopt;
}

void adaptive_allocator_manager::reset_stats()
{
    std::lock_guard<std::mutex> const lock(pimpl_->mutex_);

    if (pimpl_->cpu_allocator_ != nullptr)
    {
        pimpl_->cpu_allocator_->ClearStats();
    }
    if (pimpl_->pool_allocator_)
    {
        pimpl_->pool_allocator_->ClearStats();
    }
    if (pimpl_->bfc_allocator_)
    {
        pimpl_->bfc_allocator_->ClearStats();
    }
}

}  // namespace xsigma
