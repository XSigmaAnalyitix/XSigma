#include "memory/numa.h"

#ifdef XSIGMA_NUMA_ENABLED

#include <numa.h>
#include <numaif.h>
#include <unistd.h>

#include "common/configure.h"
#include "util/exception.h"

namespace xsigma
{
bool IsNUMAEnabled()
{
    return numa_available() >= 0;
}

void NUMABind(int numa_node_id)
{
    if (numa_node_id < 0)
    {
        return;
    }
    if (!IsNUMAEnabled())
    {
        return;
    }

    XSIGMA_CHECK(numa_node_id <= numa_max_node(), "NUMA node id ", numa_node_id, " is unavailable");

    auto* bm = numa_allocate_nodemask();
    numa_bitmask_setbit(bm, numa_node_id);
    numa_bind(bm);
    numa_bitmask_free(bm);
}

int GetNUMANode(const void* ptr)
{
    if (!IsNUMAEnabled())
    {
        return -1;
    }
    XSIGMA_CHECK(ptr != nullptr, "");

    int numa_node = -1;
    XSIGMA_CHECK(
        get_mempolicy(&numa_node, nullptr, 0, const_cast<void*>(ptr), MPOL_F_NODE | MPOL_F_ADDR) ==
            0,
        "Unable to get memory policy, errno:",
        errno);
    return numa_node;
}

int GetNumNUMANodes()
{
    if (!IsNUMAEnabled())
    {
        return -1;
    }

    return numa_num_configured_nodes();
}

void NUMAMove(void* ptr, size_t size, int numa_node_id)
{
    if (numa_node_id < 0)
    {
        return;
    }
    if (!IsNUMAEnabled())
    {
        return;
    }
    XSIGMA_CHECK(ptr != nullptr, "");

    uintptr_t page_start_ptr = ((reinterpret_cast<uintptr_t>(ptr)) & ~(getpagesize() - 1));
    ptrdiff_t offset         = reinterpret_cast<uintptr_t>(ptr) - page_start_ptr;
    // Avoid extra dynamic allocation and NUMA api calls
    XSIGMA_CHECK(static_cast<unsigned>(numa_node_id) < sizeof(unsigned long) * 8, "");
    unsigned long mask = 1UL << numa_node_id;
    XSIGMA_CHECK(
        mbind(
            reinterpret_cast<void*>(page_start_ptr),
            size + offset,
            MPOL_BIND,
            &mask,
            sizeof(mask) * 8,
            MPOL_MF_MOVE | MPOL_MF_STRICT) == 0,
        "Could not move memory to a NUMA node");
}

int GetCurrentNUMANode()
{
    if (!IsNUMAEnabled())
    {
        return -1;
    }

    auto n = numa_node_of_cpu(sched_getcpu());
    return n;
}

}  // namespace xsigma
#endif