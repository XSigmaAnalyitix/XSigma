#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch::jit
{

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g, std::map<std::string, IValue>& paramsDict, bool is_train);

}  // namespace torch::jit
