#pragma once

#include <torch/csrc/Export.h>
#include <xsigma/core/InferenceMode.h>

namespace torch::autograd
{

using InferenceMode = xsigma::InferenceMode;

}
