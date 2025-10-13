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

/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef XSIGMA_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_
#define XSIGMA_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_

#include "experimental/profiler/exporters/xplane/xplane.h"

namespace xsigma
{

/**
 * @brief Interface for XSigma profiler plugins (REQUIRED COMPONENT)
 *
 * profile_session calls each of these methods at most once per instance, and
 * implementations can rely on that guarantee for simplicity.
 *
 * Thread-safety: Implementations are only required to be thread-compatible.
 * profile_session is thread-safe and synchronizes access to profiler_interface
 * instances.
 *
 * COMPONENT CLASSIFICATION: REQUIRED
 * This is a core component essential for basic profiling functionality.
 */
class XSIGMA_API profiler_interface
{
public:
    virtual ~profiler_interface() = default;

    /**
     * @brief Starts profiling
     * @return true if profiling started successfully, false otherwise
     */
    virtual bool start() = 0;

    /**
     * @brief Stops profiling
     * @return true if profiling stopped successfully, false otherwise
     */
    virtual bool stop() = 0;

    /**
     * @brief Saves collected profile data into XSpace
     * @param space Pointer to XSpace where profile data will be stored
     * @return true if data collection was successful, false otherwise
     */
    virtual bool collect_data(x_space* space) = 0;
};

}  // namespace xsigma

#endif  // XSIGMA_TSL_PROFILER_LIB_PROFILER_INTERFACE_H_
