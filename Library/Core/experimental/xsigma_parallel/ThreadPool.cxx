#include "experimental/xsigma_parallel/ThreadPool.h"

#include "experimental/xsigma_parallel/WorkersPool.h"
#include "logging/logger.h"

#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#else
#include <thread>
#endif

XSIGMA_DEFINE_bool(
    caffe2_threadpool_force_inline, false, "Force to always run jobs on the calling thread")

    // Whether or not threadpool caps apply to Android
    XSIGMA_DEFINE_int(caffe2_threadpool_android_cap, true, "")

    // Whether or not threadpool caps apply to iOS and MacOS
    XSIGMA_DEFINE_int(caffe2_threadpool_ios_cap, true, "")
        XSIGMA_DEFINE_int(caffe2_threadpool_macos_cap, true, "")

            XSIGMA_DEFINE_int(pthreadpool_size, 0, "Override the default thread pool size.")

                namespace caffe2
{
    namespace
    {
    class thread_pool_impl : public thread_pool
    {
    public:
        explicit thread_pool_impl(int num_threads);
        ~thread_pool_impl() override;

        // Returns the number of threads currently in use
        int  get_num_threads() const override;
        void set_num_threads(size_t num_threads) override;

        void run(const std::function<void(int, size_t)>& fn, size_t range) override;
        void with_pool(const std::function<void(workers_pool*)>& f) override;

    private:
        std::atomic_size_t                 num_threads_;
        std::shared_ptr<workers_pool>      workers_pool_;
        std::vector<std::shared_ptr<task>> tasks_;
    };
    }  // namespace

    size_t get_default_num_threads()
    {
#if !defined(__s390x__) && !defined(__powerpc__)
        auto num_threads = 1U;
        if (cpuinfo_initialize())
        {
            num_threads = std::max(cpuinfo_get_processors_count(), 1U);
        }
        else
        {
            XSIGMA_LOG_WARNING("cpuinfo initialization failed");
            num_threads = std::max(std::thread::hardware_concurrency(), 1U);
        }

        bool apply_cap = false;
#if defined(XSIGMA_ANDROID)
        apply_cap = FLAGS_caffe2_threadpool_android_cap;
#elif defined(XSIGMA_IOS)
        apply_cap = FLAGS_caffe2_threadpool_ios_cap;
#elif defined(TARGET_OS_MAC)
        apply_cap = FLAGS_caffe2_threadpool_macos_cap;
#endif

        if (apply_cap)
        {
            switch (num_threads)
            {
#if defined(XSIGMA_ANDROID) && (CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64)
            case 4:
                switch (cpuinfo_get_core(0)->midr & UINT32_C(0xFF00FFF0))
                {
                case UINT32_C(0x51002110): /* Snapdragon 820 Kryo Silver */
                case UINT32_C(0x51002010): /* Snapdragon 821 Kryo Silver */
                case UINT32_C(0x51002050): /* Snapdragon 820/821 Kryo Gold */
                    /* Kryo: 2+2 big.LITTLE */
                    num_threads = 2;
                    break;
                default:
                    /* Anything else: assume homogeneous architecture */
                    num_threads = 4;
                    break;
                }
                break;
#endif
            case 5:
                /* 4+1 big.LITTLE */
                num_threads = 4;
                break;
            case 6:
                /* 2+4 big.LITTLE */
                num_threads = 2;
                break;
            // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,bugprone-branch-clone)
            case 8:
                /* 4+4 big.LITTLE */
                num_threads = 4;
                break;
            case 10:
                /* 4+4+2 Min.Med.Max, running on Med cores */
                num_threads = 4;
                break;
            default:
                if (num_threads > 4)
                {
                    num_threads = num_threads / 2;
                }
                break;
            }
        }
#else
        auto num_threads = std::max(std::thread::hardware_concurrency(), 1U);
#endif

        if (FLAGS_pthreadpool_size)
        {
            // Always give precedence to explicit setting.
            num_threads = FLAGS_pthreadpool_size;
        }

        /*
   * For llvm-tsan, holding limit for the number of locks for a single thread
   * is 63 (because of comparison < 64 instead of <=). pthreadpool's worst
   * case is the number of threads in a pool. So we want to limit the threadpool
   * size to 64 when running with tsan. However, sometimes it is tricky to
   * detect if we are running under tsan, for now capping the default
   * threadcount to the tsan limit unconditionally.
   */
        auto tsan_thread_limit = 63U;
        num_threads            = std::min(num_threads, tsan_thread_limit);

        return num_threads;
    }

    // Default smallest amount of work that will be partitioned between
    // multiple threads; the runtime value is configurable
    constexpr size_t k_default_min_work_size = 1;

    size_t thread_pool::default_num_threads_ = 0;

    thread_pool* thread_pool::create_thread_pool(int num_threads)
    {
        return new thread_pool_impl(num_threads);
    }

    std::unique_ptr<thread_pool> thread_pool::default_thread_pool()
    {
        default_num_threads_ = get_default_num_threads();
        XSIGMA_LOG_INFO("Constructing thread pool with {} threads", default_num_threads_);
        return std::make_unique<thread_pool_impl>(default_num_threads_);
    }

    thread_pool_impl::thread_pool_impl(int num_threads)
        : num_threads_(num_threads), workers_pool_(std::make_shared<workers_pool>())
    {
        min_work_size_ = k_default_min_work_size;
    }

    // NOLINTNEXTLINE(modernize-use-equals-default)
    thread_pool_impl::~thread_pool_impl() {}

    int thread_pool_impl::get_num_threads() const
    {
        return num_threads_;
    }

    // Sets the number of threads
    // # of threads should not be bigger than the number of big cores
    void thread_pool_impl::set_num_threads(size_t num_threads)
    {
        if (default_num_threads_ == 0)
        {
            default_num_threads_ = get_default_num_threads();
        }
        num_threads_ = std::min(num_threads, default_num_threads_);
    }

    void thread_pool_impl::run(const std::function<void(int, size_t)>& fn, size_t range)
    {
        const auto num_threads = num_threads_.load(std::memory_order_relaxed);

        std::lock_guard<std::mutex> guard(execution_mutex_);
        // If there are no worker threads, or if the range is too small (too
        // little work), just run locally
        const bool run_locally =
            range < min_work_size_ || FLAGS_caffe2_threadpool_force_inline || (num_threads == 0);
        if (run_locally)
        {
            // Work is small enough to just run locally; multithread overhead
            // is too high
            for (size_t i = 0; i < range; ++i)
            {
                fn(0, i);
            }
            return;
        }

        struct fn_task : public task
        {
            const std::function<void(int, size_t)>* fn_{};
            int                                     idx_{};
            size_t                                  start_{};
            size_t                                  end_{};
            void                                    run() override
            {
                for (auto i = start_; i < end_; ++i)
                {
                    (*fn_)(idx_, i);
                }
            }
        };

        if (num_threads_ < 1)
        {
            XSIGMA_LOG_ERROR("Invalid number of threads: {}", num_threads_);
            return;
        }
        const size_t units_per_task = (range + num_threads - 1) / num_threads;
        tasks_.resize(num_threads);
        for (size_t i = 0; i < num_threads; ++i)
        {
            if (!tasks_[i])
            {
                // NOLINTNEXTLINE(modernize-make-shared)
                tasks_[i].reset(new fn_task());
            }
            auto* task   = (fn_task*)tasks_[i].get();
            task->fn_    = &fn;
            task->idx_   = i;
            task->start_ = std::min<size_t>(range, i * units_per_task);
            task->end_   = std::min<size_t>(range, (i + 1) * units_per_task);
            if (task->start_ >= task->end_)
            {
                tasks_.resize(i);
                break;
            }
            if (task->start_ > range || task->end_ > range)
            {
                XSIGMA_LOG_ERROR("Task range out of bounds");
                return;
            }
        }
        if (tasks_.size() > num_threads || tasks_.size() < 1)
        {
            XSIGMA_LOG_ERROR("Invalid task count: {}", tasks_.size());
            return;
        }
        workers_pool_->execute(tasks_);
    }

    void thread_pool_impl::with_pool(const std::function<void(workers_pool*)>& f)
    {
        std::lock_guard<std::mutex> guard(execution_mutex_);
        f(workers_pool_.get());
    }

}  // namespace caffe2
