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

/* Copyright 2022 The OpenXLA Authors.

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
#include <memory>  // for unique_ptr

#include "common/macros.h"                             // for XSIGMA_UNUSED
#include "experimental/profiler/profiler_factory.h"    // for register_profiler_factory
#include "experimental/profiler/profiler_interface.h"  // for profiler_interface
#include "experimental/profiler/profiler_options.h"    // for profile_options

namespace xsigma
{
namespace profiler
{
namespace
{

std::unique_ptr<profiler_interface> CreatePythonTracer(
    XSIGMA_UNUSED const profile_options& profile_options)
{
    /*PythonTracerOptions options;
    options.enable_trace_python_function = profile_options.python_tracer_level();
    options.enable_python_traceme        = profile_options.host_tracer_level();
    return CreatePythonTracer(options);*/
    return nullptr;
}

auto register_python_tracer_factory = []
{
    register_profiler_factory(&CreatePythonTracer);
    return 0;
}();

}  // namespace
}  // namespace profiler
}  // namespace xsigma
