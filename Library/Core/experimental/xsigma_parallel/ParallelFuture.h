#pragma once

#include <functional>

#include "common/export.h"
#include "common/macros.h"

// TODO: ATen dependency - needs to be replaced or removed
// #include <ATen/core/ivalue.h>

namespace xsigma {

// Launches intra-op parallel task, returns a future
// TODO: Update return type when ivalue::Future is available
// XSIGMA_API xsigma::intrusive_ptr<xsigma::ivalue::Future> intraop_launch_future(
//     const std::function<void()>& func);

}  // namespace xsigma
