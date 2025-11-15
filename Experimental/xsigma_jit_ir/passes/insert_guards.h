#pragma once

#include <XSigma/XSigma.h>
#include <XSigma/core/ivalue.h>
#include <XSigma/core/jit_type.h>
#include <XSigma/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

#include <list>
#include <vector>

namespace torch::jit
{

TORCH_API void InsertGuards(std::shared_ptr<Graph> graph);

TORCH_API void RemoveProfilingNodes(const std::shared_ptr<Graph>& graph);

}  // namespace torch::jit
