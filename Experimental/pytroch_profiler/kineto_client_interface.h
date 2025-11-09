#pragma once

//#include <xsigma/csrc/jit/runtime/interpreter.h>

#include "unwind/unwind.h"

namespace xsigma
{

// declare global_kineto_init for libtorch_cpu.so to call
XSIGMA_API void global_kineto_init();

}  // namespace xsigma
