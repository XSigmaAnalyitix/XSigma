#include <pybind11/pybind11.h>
#include <xsigma/csrc/utils/pybind.h>

#include <nlohmann/json.hpp>

#include "profiler/pytorch_profiler/combined_traceback.h"

namespace xsigma
{

// symbolize combined traceback objects, converting them into lists of
// dictionaries that are easily consumed in python.

// returns std::vector because one use is to call it with a batch of
// tracebacks that come from a larger datastructure (e.g. a memory snapshot)
// and then have more c++ code to put those objects in the right place.
XSIGMA_API std::vector<pybind11::object> py_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

// Return the callback in json format so that it can be used within cpp
XSIGMA_API std::vector<nlohmann::json> json_symbolize(
    std::vector<CapturedTraceback*>& to_symbolize);

// requires GIL to be held, frees any pending free frames
XSIGMA_API void freeDeadCapturedTracebackFrames();

XSIGMA_API void installCapturedTracebackPython();

}  // namespace xsigma
