#pragma once

#include <torch/csrc/jit/api/module.h>

namespace torch::jit
{

struct TORCH_API LinearBNParameters
{
    xsigma::Tensor linear_w;
    xsigma::Tensor linear_b;
    xsigma::Tensor bn_rm;
    xsigma::Tensor bn_rv;
    double         bn_eps = 0.0;
    xsigma::Tensor bn_w;
    xsigma::Tensor bn_b;
};

/**
 * Given the current weight and bias tensors of a Linear module and parameters
 * of the BatchNorm module we're folding with, compute the updated values
 * for the weight and bias.
 *
 * The function is basically copied from torch/nn/utils/fusion.py
 */
TORCH_API std::tuple<xsigma::Tensor, xsigma::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p);

}  // namespace torch::jit
