#pragma once

#include <XSigma/core/grad_mode.h>
#include <torch/csrc/Export.h>

namespace torch::autograd
{

using GradMode     = xsigma::GradMode;
using AutoGradMode = xsigma::AutoGradMode;

}  // namespace torch::autograd
