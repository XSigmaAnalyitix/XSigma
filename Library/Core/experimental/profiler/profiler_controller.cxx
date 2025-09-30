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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "experimental/profiler/profiler_controller.h"

#include <memory>
#include <utility>

#include "experimental/profiler/profiler_interface.h"
#include "experimental/profiler/xplane/xplane.h"
#include "logging/logger.h"

namespace xsigma
{

profiler_controller::profiler_controller(std::unique_ptr<profiler_interface> profiler)
    : profiler_(std::move(profiler)), status_(true)
{
}

profiler_controller::~profiler_controller()
{
    // Ensure a successfully started profiler is stopped.
    if (state_ == profiler_state_enum::START && status_)
    {
        profiler_->stop();
    }
}

bool profiler_controller::start()
{
    bool status = false;
    if (state_ == profiler_state_enum::INIT)
    {
        state_ = profiler_state_enum::START;
        if (status_)
        {
            status = status_ = profiler_->start();
        }
        else
        {
            XSIGMA_LOG_ERROR("Previous call returned an error.");
        }
    }
    else
    {
        XSIGMA_LOG_ERROR("start called in the wrong order");
    }
    return status;
}

bool profiler_controller::stop()
{
    bool status = false;
    if (state_ == profiler_state_enum::START)
    {
        state_ = profiler_state_enum::STOP;
        if (status_)
        {
            status = status_ = profiler_->stop();
        }
        else
        {
            XSIGMA_LOG_ERROR("Previous call returned an error.");
        }
    }
    else
    {
        XSIGMA_LOG_ERROR("stop called in the wrong order");
    }
    return status;
}

bool profiler_controller::collect_data(x_space* space)
{
    bool status = false;
    if (state_ == profiler_state_enum::STOP)
    {
        state_ = profiler_state_enum::COLLECT_DATA;
        if (status_)
        {
            status = status_ = profiler_->collect_data(space);
        }
        else
        {
            XSIGMA_LOG_ERROR("Previous call returned an error.");
        }
    }
    else
    {
        XSIGMA_LOG_ERROR("collect_data called in the wrong order.");
    }
    return status;
}

}  // namespace xsigma
