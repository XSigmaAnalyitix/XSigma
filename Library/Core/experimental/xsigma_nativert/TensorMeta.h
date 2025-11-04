#pragma once

#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/executor/Placement.h>
#include <xsigma/core/Device.h>
#include <xsigma/core/Layout.h>
#include <xsigma/core/MemoryFormat.h>
#include <xsigma/core/ScalarType.h>
#include <xsigma/core/TensorOptions.h>
#include <xsigma/util/ArrayRef.h>
#include <xsigma/util/Logging.h>

namespace torch::nativert
{

xsigma::ScalarType   convertJsonScalarType(const torch::_export::ScalarType& scalarType);
xsigma::MemoryFormat convertJsonMemoryFormat(const torch::_export::MemoryFormat& memoryFormat);
xsigma::Layout       convertJsonLayout(const torch::_export::Layout& layout);
xsigma::Device       convertJsonDevice(const torch::_export::Device& device);

class TensorMeta
{
public:
    explicit TensorMeta(const torch::_export::TensorMeta& tensorMeta);

    xsigma::IntArrayRef sizes() const
    {
        XSIGMA_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
        return sizes_;
    }

    xsigma::IntArrayRef strides() const
    {
        XSIGMA_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
        return strides_;
    }

    xsigma::Layout layout() const { return layout_; }

    xsigma::ScalarType dtype() const { return dtype_; }

    bool requires_grad() const { return requiresGrad_; }

    int64_t storage_offset() const { return storage_offset_; }

    int64_t dim() const { return sizes_.size(); }

    int64_t numel() const
    {
        XSIGMA_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
        return numel_;
    }

    xsigma::Device device() const { return device_; }

    // override device according to placement
    void setDevice(xsigma::Device device) { device_ = device; }

    xsigma::TensorOptions asTensorOptions() const
    {
        return xsigma::TensorOptions().dtype(dtype_).layout(layout_).requires_grad(requiresGrad_);
    }

    // override device according to placement
    void applyDevicePlacement(const Placement& placement)
    {
        device_ = placement.getMappedDevice(device_);
    }

    // NYI
    // xsigma::SymIntArrayRef sym_sizes() const {}
    // xsigma::SymIntArrayRef sym_strides() const {}
    // xsigma::SymInt sym_storage_offset() const {}
    // xsigma::SymInt sym_numel() const {}

private:
    bool hasSymbolicShape_ = false;

    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
    int64_t              storage_offset_ = 0;
    int64_t              numel_          = 1;

    xsigma::ScalarType dtype_;
    xsigma::Layout     layout_;
    bool               requiresGrad_;

    xsigma::Device device_;
};

}  // namespace torch::nativert
