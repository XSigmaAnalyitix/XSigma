#pragma once

#include <XSigma/ExpandUtils.h>
#include <XSigma/NestedTensorImpl.h>
#include <XSigma/core/Tensor.h>
#include <xsigma/core/Device.h>
#include <xsigma/core/DeviceType.h>
#include <xsigma/core/Stream.h>
#include <xsigma/core/SymIntArrayRef.h>
#include <xsigma/core/TensorImpl.h>
#include <xsigma/core/impl/DeviceGuardImplInterface.h>
#include <xsigma/util/DimVector.h>
#include <xsigma/util/SmallVector.h>

#include "util/exception.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <XSigma/Functions.h>
#else
#include <XSigma/ops/zeros.h>
#endif

namespace torch::autograd
{

using SymIntSmallVec = xsigma::SmallVector<xsigma::SymInt, xsigma::kDimVectorStaticSize>;
using MetadataShape  = std::variant<SymIntSmallVec, xsigma::Tensor>;

/**
 * Records TensorOptions, shape of the tensor, whether or not the Python
 * dispatch key is set (tensor subclass), and, where applicable, the stream the
 * corresponding operation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct TORCH_API InputMetadata
{
    InputMetadata() = default;
    InputMetadata(
        const xsigma::TensorOptions&      options,
        MetadataShape                     input_shape,
        bool                              is_tensor_subclass,
        bool                              is_nested,
        std::optional<xsigma::ScalarType> grad_dtype);
    InputMetadata(const xsigma::Tensor& t);

    const xsigma::TensorOptions& options() const { return options_; }

    caffe2::TypeMeta dtype() const { return options_.dtype(); }

    xsigma::Device device() const { return options_.device(); }

    xsigma::Layout layout() const { return options_.layout(); }

    xsigma::Stream stream() const { return stream_; }

    bool is_tensor_subclass() const { return is_tensor_subclass_; }

    xsigma::Tensor zeros_like() const;

    bool is_same_shape(const xsigma::Tensor& grad) const;

    bool is_expandable_to_shape(const xsigma::Tensor& grad) const;

    xsigma::Tensor reduce_grad(xsigma::Tensor& grad) const;

    xsigma::Tensor maybe_reduce(
        const size_t                                          index,
        xsigma::Tensor                                        grad,
        const std::function<std::string(const std::string&)>& format_error) const;

    std::stringstream incompatible_shape_error_message(
        const size_t index, const xsigma::Tensor& grad) const;

    bool was_default_constructed() const { return was_default_constructed_; }

    bool is_cpp_nested_tensor() const;

    bool is_nested_tensor() const { return is_nested_; }

    xsigma::SymIntArrayRef shape_as_dim_vector() const;

    // Danger: not thread safe, caller must protect with lock
    SymIntSmallVec& mutable_shape_as_dim_vector();

    std::optional<xsigma::ScalarType> grad_dtype() const
    {
        TORCH_INTERNAL_ASSERT(!was_default_constructed_);
        return grad_dtype_;
    }

    void set_grad_dtype(const std::optional<xsigma::ScalarType>& grad_dtype)
    {
        TORCH_INTERNAL_ASSERT(!was_default_constructed_);
        grad_dtype_ = grad_dtype;
    }

private:
    xsigma::Tensor shape_as_tensor() const;
    bool           is_nestedness_same(const xsigma::Tensor& grad) const;
    bool           maybe_expandable_to(const xsigma::Tensor& grad) const;

    // NB: The engine does not use the dtype from the options, but rather the
    //     grad_dtype_ field to validate grad_output dtype.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const xsigma::TensorOptions options_;
    MetadataShape               shape_;
    xsigma::Stream stream_             = xsigma::Stream(xsigma::Stream::Default::DEFAULT, device());
    bool           is_tensor_subclass_ = false;
    bool           is_nested_          = false;
    bool           was_default_constructed_ = true;

    // The grad_dtype_ field is the dtype that the engine expects the grad to be.
    // When nullopt, grad_dtype_ is allowed to be any dtype.
    // This field is mutated if THPVariable_set_grad_dtype is called
    // and the AccumulateGrad has already been created.
    std::optional<xsigma::ScalarType> grad_dtype_;
};
}  // namespace torch::autograd
