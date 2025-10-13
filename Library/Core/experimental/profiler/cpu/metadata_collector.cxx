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

/* Copyright 2018 The OpenXLA Authors.

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

#include <memory>  // for make_unique, unique_ptr

#include "common/macros.h"                                  // for XSIGMA_UNUSED
#include "experimental/profiler/core/profiler_factory.h"    // for register_profiler_factory
#include "experimental/profiler/core/profiler_interface.h"  // for profiler_interface
#include "experimental/profiler/core/profiler_options.h"    // for profile_options
#include "experimental/profiler/exporters/xplane/xplane.h"  // for x_space

namespace xsigma::profiler
{
namespace
{

// MetadataCollector collect miscellaneous metadata for xprof, e.g. HLO protos
// from XLA runtime etc.
//
// Thread-safety: This class is go/thread-compatible.
class MetadataCollector : public profiler_interface
{
public:
    MetadataCollector() = default;

    bool start() override
    {
        if (!trace_active_)
        {
            //xla::XlaDebugInfoManager::Get()->StartTracing();
            trace_active_ = true;
        }
        return true;
    }

    bool stop() override
    {
        if (trace_active_)
        {
            //xla::XlaDebugInfoManager::Get()->StopTracing(&debug_info_);
            trace_active_ = false;
        }
        return true;
    }

    bool collect_data(XSIGMA_UNUSED x_space* space) override
    {
        /*if (!debug_info_.empty())
        {
            XPlane*               plane = FindOrAddMutablePlaneWithName(space, kMetadataPlaneName);
            MetadataXPlaneBuilder metadata_plane(plane);
            for (auto& hlo_proto : debug_info_)
            {
                metadata_plane.AddHloProto(hlo_proto->hlo_module().id(), *hlo_proto);
                hlo_proto.reset();
            }
            debug_info_.clear();
        }*/
        return true;
    }

    MetadataCollector(const MetadataCollector&) = delete;
    void operator=(const MetadataCollector&)    = delete;

private:
    //std::vector<std::unique_ptr<xla::HloProto>> debug_info_;
    bool trace_active_ = false;
};

std::unique_ptr<profiler_interface> CreatMetadataCollector(const profile_options& options)
{
    return options.enable_hlo_proto() ? std::make_unique<MetadataCollector>() : nullptr;
}

}  // namespace

static auto register_metadata_collector_factory = []
{
    register_profiler_factory(&CreatMetadataCollector);
    return 0;
}();
}  // namespace xsigma::profiler
