#pragma once

#include <XSigma/XSigma.h>
#include <XSigma/core/ivalue.h>
#include <XSigma/core/jit_type.h>
#include <XSigma/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <xsigma/util/sparse_bitset.h>

#include <list>
#include <unordered_map>
#include <vector>

namespace torch::jit
{

using SparseBitVector = ::xsigma::SparseBitVector<256>;

// BuildLivenessSets computes "bailout" liveness which is equivalent to
// "{LIVE_IN} or {GEN}" or "{LIVE_OUT} - {KILL}"
TORCH_API std::unordered_map<Node*, std::vector<Value*>> BuildLivenessSets(
    std::shared_ptr<Graph> graph);
}  // namespace torch::jit
