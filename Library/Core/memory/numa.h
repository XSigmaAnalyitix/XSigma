#pragma once

#include "common/configure.h"  // IWYU pragma: keep

#ifdef XSIGMA_NUMA_ENABLED

#include <cstddef>

namespace xsigma
{
/**
 * Check whether NUMA is enabled
 */
XSIGMA_API bool IsNUMAEnabled();

/**
 * Bind to a given NUMA node
 */
XSIGMA_API void NUMABind(int numa_node_id);

/**
 * Get the NUMA id for a given pointer `ptr`
 */
XSIGMA_API int GetNUMANode(const void* ptr);

/**
 * Get number of NUMA nodes
 */
XSIGMA_API int GetNumNUMANodes();

/**
 * Move the memory pointed to by `ptr` of a given size to another NUMA node
 */
XSIGMA_API void NUMAMove(void* ptr, size_t size, int numa_node_id);

/**
 * Get the current NUMA node id
 */
XSIGMA_API int GetCurrentNUMANode();

}  // namespace xsigma
#endif  // XSIGMA_NUMA_ENABLED
