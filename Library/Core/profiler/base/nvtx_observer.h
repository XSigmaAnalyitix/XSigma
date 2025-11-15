#include "profiler/common/api.h"

namespace xsigma::profiler::impl
{

void pushNVTXCallbacks(
    const ProfilerConfig& config, const std::unordered_set<xsigma::RecordScope>& scopes);

}  // namespace xsigma::profiler::impl
