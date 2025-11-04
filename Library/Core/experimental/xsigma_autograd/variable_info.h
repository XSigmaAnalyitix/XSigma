#pragma once

#include <torch/csrc/autograd/variable.h>

namespace torch::autograd
{

struct TORCH_API VariableInfo
{
    explicit VariableInfo();
    explicit VariableInfo(const Variable& var, bool use_zeros_like = false);

    Variable zeros(xsigma::OptionalDeviceGuard& device_guard) const;

    xsigma::Layout              layout      = xsigma::Layout::Strided;
    xsigma::Device              device      = xsigma::kCPU;
    xsigma::ScalarType          scalar_type = xsigma::kFloat;
    std::vector<xsigma::SymInt> size;
    bool                        requires_grad;
    bool                        is_empty;
    // needed for e.g. NJTs since they only support zeros_like()
    std::optional<Variable> the_var;
};

}  // namespace torch::autograd
