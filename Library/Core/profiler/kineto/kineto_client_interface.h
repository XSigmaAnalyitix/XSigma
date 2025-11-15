#pragma once

// TODO: Missing XSigma dependency - original include was:
// #include <xsigma/csrc/jit/runtime/interpreter.h>
// This is a XSigma-specific header not available in XSigma

#include "profiler/common/unwind/unwind.h"

namespace xsigma
{

// declare global_kineto_init for libtorch_cpu.so to call
XSIGMA_API void global_kineto_init();

}  // namespace xsigma
