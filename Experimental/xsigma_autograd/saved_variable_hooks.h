#pragma once

#include <XSigma/core/Tensor.h>
#include <xsigma/core/SafePyObject.h>

namespace torch::autograd
{

struct TORCH_API SavedVariableHooks
{
    virtual void           call_pack_hook(const xsigma::Tensor& tensor) = 0;
    virtual xsigma::Tensor call_unpack_hook()                           = 0;
    virtual ~SavedVariableHooks()                                       = default;
    virtual std::optional<std::pair<xsigma::SafePyObject, xsigma::SafePyObject>>
    retrieve_unpack_hook_data() const
    {
        XSIGMA_CHECK(false, "Compiled Autograd only supports python saved tensor hooks ");
    }
};

}  // namespace torch::autograd
