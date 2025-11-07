#pragma once

#include <Python.h>
#include "collection.h"
#include "python/pybind.h"

namespace pybind11::detail
{
using xsigma::profiler::impl::TensorID;

#define STRONG_POINTER_TYPE_CASTER(T)                            \
    template <>                                                  \
    struct type_caster<T> : public strong_pointer_type_caster<T> \
    {                                                            \
    };

STRONG_POINTER_TYPE_CASTER(xsigma::profiler::impl::StorageImplData)
STRONG_POINTER_TYPE_CASTER(xsigma::profiler::impl::AllocationID)
STRONG_POINTER_TYPE_CASTER(xsigma::profiler::impl::TensorImplAddress)
STRONG_POINTER_TYPE_CASTER(xsigma::profiler::impl::PyModuleSelf)
STRONG_POINTER_TYPE_CASTER(xsigma::profiler::impl::PyModuleCls)
STRONG_POINTER_TYPE_CASTER(xsigma::profiler::impl::PyOptimizerSelf)
#undef STRONG_POINTER_TYPE_CASTER

template <>
struct type_caster<TensorID> : public strong_uint_type_caster<TensorID>
{
};
}  // namespace pybind11::detail

namespace xsigma::profiler
{

void initPythonBindings(PyObject* module);

}  // namespace xsigma::profiler
