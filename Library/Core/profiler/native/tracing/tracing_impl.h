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

#include "profiler/native/cpu/threadpool_listener_state.h"

namespace xsigma::tracing
{

inline bool event_collector::is_enabled()
{
    return profiler::threadpool_listener::IsEnabled();
}

}  // namespace xsigma::tracing

// Stub tracing macros for portability.
#define XSIGMA_TRACELITERAL(a) \
    do                         \
    {                          \
    } while (0)
#define XSIGMA_TRACESTRING(s) \
    do                        \
    {                         \
    } while (0)
#define XSIGMA_TRACEPRINTF(format, ...) \
    do                                  \
    {                                   \
    } while (0)
