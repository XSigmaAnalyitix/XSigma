#pragma once
#include <mutex>
#include <vector>

#include "util/exception.h"

namespace torch::autograd::utils
{

// Warning handler for multi-threaded contexts. Gather warnings from
// all threads into a single queue, then process together xsigma the end
// in the main thread.
class DelayWarningHandler : public xsigma::WarningHandler
{
public:
    ~DelayWarningHandler() override = default;
    void replay_warnings();

private:
    void process(const xsigma::Warning& warning) override;

    std::vector<xsigma::Warning> warnings_;
    std::mutex                   mutex_;
};

}  // namespace torch::autograd::utils
