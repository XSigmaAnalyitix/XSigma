#include <XSigma/TensorOperators.h>
#include <torch/csrc/jit/passes/fold_linear_bn.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <XSigma/Functions.h>
#else
#include <XSigma/ops/rsqrt.h>
#endif

namespace torch::jit
{

std::tuple<xsigma::Tensor, xsigma::Tensor> computeUpdatedLinearWeightAndBias(
    const LinearBNParameters& p)
{
    xsigma::Tensor bn_scale = p.bn_w * xsigma::rsqrt(p.bn_rv + p.bn_eps);
    xsigma::Tensor fused_w  = p.linear_w * bn_scale.unsqueeze(-1);
    xsigma::Tensor fused_b  = (p.linear_b - p.bn_rm) * bn_scale + p.bn_b;

    auto linear_w_dtype = p.linear_w.dtype();
    auto linear_b_dtype = p.linear_b.dtype();

    return std::make_tuple(fused_w.to(linear_w_dtype), fused_b.to(linear_b_dtype));
}

}  // namespace torch::jit
