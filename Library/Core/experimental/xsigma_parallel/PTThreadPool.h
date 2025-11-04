#pragma once

#include "common/export.h"
#include "experimental/xsigma_parallel/Parallel.h"
#include "experimental/xsigma_parallel/thread_pool.h"

namespace xsigma
{

class XSIGMA_VISIBILITY pt_thread_pool : public xsigma::thread_pool
{
public:
    explicit pt_thread_pool(int pool_size, int numa_node_id = -1)
        : xsigma::thread_pool(
              pool_size,
              numa_node_id,
              []()
              {
                  xsigma::set_thread_name("PTThreadPool");
                  xsigma::init_num_threads();
              })
    {
    }
};

}  // namespace xsigma
