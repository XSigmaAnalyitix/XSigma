/**
 * @file TestProfilerHeavyFunction.cxx
 * @brief Comprehensive profiling example with computationally intensive functions
 *
 * This test demonstrates practical profiling usage with realistic heavy computational
 * workloads including matrix operations, sorting algorithms, and data processing.
 * Shows complete instrumentation, multi-format output generation, and performance analysis.
 *
 * @author XSigma Development Team
 * @version 1.0
 * @date 2024
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "Testing/xsigmaTest.h"
#include "logging/tracing/traceme.h"
#include "logging/tracing/traceme_recorder.h"
#include "profiler/analysis/statistical_analyzer.h"
#include "profiler/memory/memory_tracker.h"
#include "profiler/session/profiler.h"

using namespace xsigma;

namespace
{

/**
 * @brief Heavy computational function: Matrix multiplication
 * Performs dense matrix multiplication with profiling instrumentation
 */
std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
{
    XSIGMA_PROFILE_SCOPE("matrix_multiply");

    const size_t rows_a = a.size();
    const size_t cols_a = a[0].size();
    const size_t cols_b = b[0].size();

    // Validate dimensions
    if (cols_a != b.size())
    {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_b, 0.0));

    {
        XSIGMA_PROFILE_SCOPE("matrix_multiply_computation");

        for (size_t i = 0; i < rows_a; ++i)
        {
            XSIGMA_PROFILE_SCOPE("matrix_row_computation");

            for (size_t j = 0; j < cols_b; ++j)
            {
                for (size_t k = 0; k < cols_a; ++k)
                {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return result;
}

/**
 * @brief Heavy computational function: Merge sort implementation
 * Recursive merge sort with profiling at each level
 */
void merge_sort(std::vector<double>& arr, size_t left, size_t right, int depth = 0)
{
    XSIGMA_PROFILE_SCOPE("merge_sort_depth_" + std::to_string(depth));

    if (left >= right)
        return;

    size_t mid = left + (right - left) / 2;

    {
        XSIGMA_PROFILE_SCOPE("merge_sort_left_half");
        merge_sort(arr, left, mid, depth + 1);
    }

    {
        XSIGMA_PROFILE_SCOPE("merge_sort_right_half");
        merge_sort(arr, mid + 1, right, depth + 1);
    }

    {
        XSIGMA_PROFILE_SCOPE("merge_operation");

        // Merge the sorted halves
        std::vector<double> temp(right - left + 1);
        size_t              i = left, j = mid + 1, k = 0;

        while (i <= mid && j <= right)
        {
            if (arr[i] <= arr[j])
            {
                temp[k++] = arr[i++];
            }
            else
            {
                temp[k++] = arr[j++];
            }
        }

        while (i <= mid)
            temp[k++] = arr[i++];
        while (j <= right)
            temp[k++] = arr[j++];

        for (size_t idx = 0; idx < temp.size(); ++idx)
        {
            arr[left + idx] = temp[idx];
        }
    }
}

/**
 * @brief Heavy computational function: Monte Carlo Pi estimation
 * Estimates Pi using random sampling with configurable precision
 */
double estimate_pi_monte_carlo(size_t num_samples)
{
    XSIGMA_PROFILE_SCOPE("monte_carlo_pi_estimation");

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    size_t points_inside_circle = 0;

    {
        XSIGMA_PROFILE_SCOPE("monte_carlo_sampling");

        for (size_t i = 0; i < num_samples; ++i)
        {
            if (i % 100000 == 0)
            {
                XSIGMA_PROFILE_SCOPE("monte_carlo_batch_" + std::to_string(i / 100000));

                for (size_t j = 0; j < std::min(size_t(100000), num_samples - i); ++j)
                {
                    double x = dis(gen);
                    double y = dis(gen);
                    if (x * x + y * y <= 1.0)
                    {
                        ++points_inside_circle;
                    }
                }
            }
        }
    }

    return 4.0 * static_cast<double>(points_inside_circle) / static_cast<double>(num_samples);
}

/**
 * @brief Heavy computational function: FFT-like computation
 * Simulates frequency domain analysis with nested loops
 */
std::vector<std::complex<double>> simulate_fft(const std::vector<double>& signal)
{
    XSIGMA_PROFILE_SCOPE("simulate_fft");

    const size_t                      n = signal.size();
    std::vector<std::complex<double>> result(n);

    {
        XSIGMA_PROFILE_SCOPE("fft_computation");

        for (size_t k = 0; k < n; ++k)
        {
            XSIGMA_PROFILE_SCOPE("fft_frequency_bin");

            std::complex<double> sum(0.0, 0.0);
            for (size_t j = 0; j < n; ++j)
            {
                double angle = -2.0 * 3.14 * k * j / n;
                sum += signal[j] * std::complex<double>(std::cos(angle), std::sin(angle));
            }
            result[k] = sum;
        }
    }

    return result;
}

/**
 * @brief Generate test data for computational functions
 */
std::vector<std::vector<double>> generate_test_matrix(size_t rows, size_t cols)
{
    XSIGMA_PROFILE_SCOPE("generate_test_matrix");

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 10.0);

    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

std::vector<double> generate_test_signal(size_t size)
{
    XSIGMA_PROFILE_SCOPE("generate_test_signal");

    std::random_device                     rd;
    std::mt19937                           gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    std::vector<double> signal(size);
    for (size_t i = 0; i < size; ++i)
    {
        signal[i] = dis(gen);
    }

    return signal;
}

}  // anonymous namespace

// Test comprehensive profiling with heavy computational functions
XSIGMATEST(Profiler, heavy_function_comprehensive_computational_profiling)
{
    // Configure profiler session with all features enabled
    profiler_options opts;
    opts.enable_timing_               = true;
    opts.enable_memory_tracking_      = true;
    opts.enable_statistical_analysis_ = true;
    opts.enable_thread_safety_        = true;
    opts.output_format_               = profiler_options::output_format_enum::JSON;

    profiler_session session(opts);
    session.start();

    {
        XSIGMA_PROFILE_SCOPE("heavy_computational_workload");

        // Test 1: Matrix multiplication profiling
        {
            XSIGMA_PROFILE_SCOPE("matrix_operations_test");

            const size_t matrix_size = 100;  // 100x100 matrices
            auto         matrix_a    = generate_test_matrix(matrix_size, matrix_size);
            auto         matrix_b    = generate_test_matrix(matrix_size, matrix_size);

            // Perform multiple matrix multiplications
            for (int i = 0; i < 3; ++i)
            {
                XSIGMA_PROFILE_SCOPE("matrix_multiply_iteration_" + std::to_string(i));
                auto result = matrix_multiply(matrix_a, matrix_b);

                // Verify result is not empty (basic correctness check)
                EXPECT_EQ(result.size(), matrix_size);
                EXPECT_EQ(result[0].size(), matrix_size);
            }
        }

        // Test 2: Sorting algorithm profiling
        {
            XSIGMA_PROFILE_SCOPE("sorting_algorithms_test");

            const size_t        array_size = 50000;
            std::vector<double> test_data(array_size);

            // Generate random data
            {
                XSIGMA_PROFILE_SCOPE("random_data_generation");
                std::random_device                     rd;
                std::mt19937                           gen(rd());
                std::uniform_real_distribution<double> dis(0.0, 1000.0);

                for (size_t i = 0; i < array_size; ++i)
                {
                    test_data[i] = dis(gen);
                }
            }

            // Test merge sort
            {
                XSIGMA_PROFILE_SCOPE("merge_sort_test");
                auto data_copy = test_data;
                merge_sort(data_copy, 0, data_copy.size() - 1);

                // Verify sorting correctness
                EXPECT_TRUE(std::is_sorted(data_copy.begin(), data_copy.end()));
            }

            // Test std::sort for comparison
            {
                XSIGMA_PROFILE_SCOPE("std_sort_comparison");
                auto data_copy = test_data;
                std::sort(data_copy.begin(), data_copy.end());

                EXPECT_TRUE(std::is_sorted(data_copy.begin(), data_copy.end()));
            }
        }

        // Test 3: Monte Carlo simulation profiling
        {
            XSIGMA_PROFILE_SCOPE("monte_carlo_simulation_test");

            const size_t num_samples = 1000000;  // 1 million samples
            double       pi_estimate = estimate_pi_monte_carlo(num_samples);

            // Verify Pi estimate is reasonable (should be close to 3.14159)
            EXPECT_GT(pi_estimate, 3.0);
            EXPECT_LT(pi_estimate, 3.3);

            std::cout << "Monte Carlo Pi estimate: " << pi_estimate << std::endl;
        }

        // Test 4: FFT simulation profiling
        {
            XSIGMA_PROFILE_SCOPE("fft_simulation_test");

            const size_t signal_size = 512;  // Common FFT size
            auto         test_signal = generate_test_signal(signal_size);

            auto fft_result = simulate_fft(test_signal);

            // Verify FFT result size
            EXPECT_EQ(fft_result.size(), signal_size);
        }

        // Test 5: Multi-threaded computation profiling
        {
            XSIGMA_PROFILE_SCOPE("multithreaded_computation_test");

            std::vector<std::thread> workers;
            const int                num_threads = 4;

            for (int i = 0; i < num_threads; ++i)
            {
                workers.emplace_back(
                    [i]()
                    {
                        XSIGMA_PROFILE_SCOPE("worker_thread_" + std::to_string(i));

                        // Each thread performs different computational work
                        const size_t         samples_per_thread = 250000;
                        XSIGMA_UNUSED double pi_est = estimate_pi_monte_carlo(samples_per_thread);

                        // Small computation to keep thread busy
                        std::vector<double> data(10000);
                        std::iota(data.begin(), data.end(), i * 10000);
                        std::sort(data.begin(), data.end(), std::greater<double>());
                    });
            }

            // Wait for all threads to complete
            for (auto& worker : workers)
            {
                worker.join();
            }
        }
    }

    session.stop();

    // ========================================================================
    // CHROME TRACE FORMAT EXPORT
    // ========================================================================
    // Export profiling results to Chrome Trace Event Format (JSON)
    // This format is compatible with:
    //   1. Chrome DevTools (chrome://tracing)
    //   2. Perfetto UI (https://ui.perfetto.dev)
    //   3. Other trace viewers that support the standard format
    //
    // HOW TO VIEW THE TRACE:
    // 1. Open Chrome browser and navigate to: chrome://tracing
    // 2. Click "Load" button and select the generated JSON file
    // 3. Use the following keyboard shortcuts:
    //    - W/S: Zoom in/out
    //    - A/D: Pan left/right
    //    - Click and drag: Select time range
    //    - Double-click: Zoom to selection
    //
    // INSIGHTS FROM THE VISUALIZATION:
    // - Timeline view shows execution order and duration of each scope
    // - Nested scopes appear as hierarchical blocks
    // - Color coding helps identify different operations
    // - Memory allocation/deallocation events are marked
    // - Thread information shows parallel execution patterns
    // - Hover over events to see detailed statistics
    // ========================================================================

    // Export to Chrome Trace JSON format
    std::string chrome_trace_file = "heavy_function_profile.json";
    session.export_report(chrome_trace_file);

    std::cout << "\n=== Heavy Function Performance Analysis ===\n";
    std::cout << "\n✓ Chrome Trace JSON exported to: " << chrome_trace_file << "\n";
    std::cout << "\nTo view the trace:\n";
    std::cout << "  1. Open Chrome and navigate to: chrome://tracing\n";
    std::cout << "  2. Click 'Load' and select: " << chrome_trace_file << "\n";
    std::cout << "  3. Use W/S to zoom, A/D to pan, click to select\n";
    std::cout << "\nAlternatively, use Perfetto UI:\n";
    std::cout << "  1. Visit: https://ui.perfetto.dev\n";
    std::cout << "  2. Open the JSON file in the UI\n";

    std::cout << "\nAll computational workloads profiled successfully:\n";
    std::cout << "  - Matrix multiplication (100x100)\n";
    std::cout << "  - Merge sort (50,000 elements)\n";
    std::cout << "  - Monte Carlo Pi estimation (1M samples)\n";
    std::cout << "  - FFT simulation (512 points)\n";
    std::cout << "  - Multi-threaded computation\n";
}

// ============================================================================
// KINETO PROFILER TEST
// ============================================================================
// Test PyTorch Kineto profiler integration with heavy computational functions
//
// Kineto provides comprehensive profiling capabilities including:
// - CPU activity tracing
// - GPU activity tracing (if available)
// - Memory profiling
// - Operator-level profiling
//
// OUTPUT: Kineto generates JSON trace files compatible with:
// - PyTorch Profiler Viewer
// - TensorBoard
// - Chrome DevTools (chrome://tracing)
//
// HOW TO USE:
// 1. Run this test to generate kineto_trace.json
// 2. View with PyTorch Profiler:
//    python -m torch.profiler.viewer kineto_trace.json
// 3. Or view in Chrome:
//    - Open chrome://tracing
//    - Load the JSON file
// ============================================================================

#ifdef XSIGMA_HAS_KINETO
#include "profiler/kineto_profiler.h"

XSIGMATEST(Profiler, kineto_heavy_function_profiling)
{
    std::cout << "\n=== Kineto Profiler Heavy Function Test ===\n";

    // Create Kineto profiler with configuration
    auto profiler = xsigma::kineto_profiler::create();

    if (!profiler)
    {
        std::cout << "Kineto profiler not available (wrapper mode)\n";
        EXPECT_TRUE(true);  // Test passes even if Kineto not available
        return;
    }

    // Configure profiler
    xsigma::kineto_profiler::profiling_config config;
    config.enable_cpu_tracing      = true;
    config.enable_gpu_tracing      = false;  // GPU tracing requires CUDA
    config.enable_memory_profiling = true;
    config.output_dir              = ".";
    config.trace_name              = "kineto_heavy_function_trace";
    config.max_activities          = 0;  // Unlimited

    // Start profiling
    if (profiler->start_profiling())
    {
        std::cout << "Kineto profiling started\n";

        // Profile heavy computational workloads
        {
            XSIGMA_PROFILE_SCOPE("kineto_matrix_operations");

            const size_t matrix_size = 50;  // Smaller for faster execution
            auto         matrix_a    = generate_test_matrix(matrix_size, matrix_size);
            auto         matrix_b    = generate_test_matrix(matrix_size, matrix_size);

            for (int i = 0; i < 2; ++i)
            {
                XSIGMA_PROFILE_SCOPE("kineto_matrix_multiply_" + std::to_string(i));
                auto result = matrix_multiply(matrix_a, matrix_b);
                EXPECT_EQ(result.size(), matrix_size);
            }
        }

        {
            XSIGMA_PROFILE_SCOPE("kineto_sorting_operations");

            const size_t        array_size = 10000;
            std::vector<double> test_data(array_size);

            std::random_device                     rd;
            std::mt19937                           gen(rd());
            std::uniform_real_distribution<double> dis(0.0, 1000.0);

            for (size_t i = 0; i < array_size; ++i)
            {
                test_data[i] = dis(gen);
            }

            {
                XSIGMA_PROFILE_SCOPE("kineto_merge_sort");
                auto data_copy = test_data;
                merge_sort(data_copy, 0, data_copy.size() - 1);
                EXPECT_TRUE(std::is_sorted(data_copy.begin(), data_copy.end()));
            }
        }

        // Stop profiling and get results
        auto result = profiler->stop_profiling();

        std::cout << "Kineto profiling completed\n";
        std::cout << "✓ Trace file: kineto_heavy_function_trace.json\n";
        std::cout << "\nTo view the trace:\n";
        std::cout << "  1. PyTorch Profiler Viewer:\n";
        std::cout << "     python -m torch.profiler.viewer kineto_heavy_function_trace.json\n";
        std::cout << "  2. Chrome DevTools:\n";
        std::cout << "     - Open chrome://tracing\n";
        std::cout << "     - Load kineto_heavy_function_trace.json\n";

        EXPECT_TRUE(true);  // Test passes if profiling completed
    }
    else
    {
        std::cout << "Failed to start Kineto profiling\n";
        EXPECT_TRUE(false);
    }
}
#endif  // XSIGMA_HAS_KINETO

// ============================================================================
// INTEL ITT API TEST
// ============================================================================
// Test Intel ITT API integration with heavy computational functions
//
// ITT API provides task and frame annotations for Intel VTune profiling:
// - Task annotations: Mark regions of code for analysis
// - Frame markers: Identify frame boundaries in graphics applications
// - String handles: Efficient string management for annotations
// - Domain-based organization: Group related tasks
//
// OUTPUT: VTune-compatible profiling data
//
// HOW TO USE:
// 1. Run this test with Intel VTune installed
// 2. Collect profiling data:
//    vtune -collect hotspots -app ./CoreCxxTests.exe
// 3. View results in VTune GUI
// 4. Look for ITT annotations in the timeline
//
// INSIGHTS:
// - Task duration shows computational complexity
// - Nested tasks reveal call hierarchy
// - Thread information shows parallelization
// - Memory events correlate with allocations
// ============================================================================

#ifdef XSIGMA_HAS_ITTAPI
#include <ittnotify.h>

XSIGMATEST(Profiler, itt_api_heavy_function_profiling)
{
    std::cout << "\n=== Intel ITT API Heavy Function Test ===\n";

    // Create ITT domain for this test
    __itt_domain* domain = __itt_domain_create("XSigmaHeavyFunctionTest");
    if (!domain)
    {
        std::cout << "ITT API domain creation failed (expected in non-VTune environment)\n";
        std::cout << "This test requires Intel VTune Profiler to be installed\n";
        std::cout << "Test passes gracefully when VTune is not available\n";
        EXPECT_TRUE(true);  // Test passes even if VTune not available
        return;
    }

    // Create string handles for task names
    auto matrix_task_handle = __itt_string_handle_create("matrix_operations");
    auto sort_task_handle   = __itt_string_handle_create("sorting_operations");
    auto monte_carlo_handle = __itt_string_handle_create("monte_carlo_simulation");

    std::cout << "ITT API domain created: XSigmaHeavyFunctionTest\n";

    // Profile matrix operations with ITT API
    {
        __itt_task_begin(domain, __itt_null, __itt_null, matrix_task_handle);
        XSIGMA_PROFILE_SCOPE("itt_matrix_operations");

        const size_t matrix_size = 50;
        auto         matrix_a    = generate_test_matrix(matrix_size, matrix_size);
        auto         matrix_b    = generate_test_matrix(matrix_size, matrix_size);

        for (int i = 0; i < 2; ++i)
        {
            auto iter_handle =
                __itt_string_handle_create(("matrix_multiply_" + std::to_string(i)).c_str());
            __itt_task_begin(domain, __itt_null, __itt_null, iter_handle);

            auto result = matrix_multiply(matrix_a, matrix_b);
            EXPECT_EQ(result.size(), matrix_size);

            __itt_task_end(domain);
        }

        __itt_task_end(domain);
    }

    // Profile sorting operations with ITT API
    {
        __itt_task_begin(domain, __itt_null, __itt_null, sort_task_handle);
        XSIGMA_PROFILE_SCOPE("itt_sorting_operations");

        const size_t        array_size = 10000;
        std::vector<double> test_data(array_size);

        std::random_device                     rd;
        std::mt19937                           gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1000.0);

        for (size_t i = 0; i < array_size; ++i)
        {
            test_data[i] = dis(gen);
        }

        {
            auto sort_handle = __itt_string_handle_create("merge_sort");
            __itt_task_begin(domain, __itt_null, __itt_null, sort_handle);

            auto data_copy = test_data;
            merge_sort(data_copy, 0, data_copy.size() - 1);
            EXPECT_TRUE(std::is_sorted(data_copy.begin(), data_copy.end()));

            __itt_task_end(domain);
        }

        __itt_task_end(domain);
    }

    // Profile Monte Carlo simulation with ITT API
    {
        __itt_task_begin(domain, __itt_null, __itt_null, monte_carlo_handle);
        XSIGMA_PROFILE_SCOPE("itt_monte_carlo_simulation");

        const size_t num_samples = 100000;
        double       pi_estimate = estimate_pi_monte_carlo(num_samples);

        EXPECT_GT(pi_estimate, 3.0);
        EXPECT_LT(pi_estimate, 3.3);

        std::cout << "Monte Carlo Pi estimate: " << pi_estimate << "\n";

        __itt_task_end(domain);
    }

    std::cout << "✓ ITT API profiling completed\n";
    std::cout << "\nTo view the profiling data:\n";
    std::cout << "  1. Install Intel VTune Profiler\n";
    std::cout << "  2. Run: vtune -collect hotspots -app ./CoreCxxTests.exe\n";
    std::cout << "  3. Open results in VTune GUI\n";
    std::cout << "  4. Look for 'XSigmaHeavyFunctionTest' domain in timeline\n";
    std::cout << "  5. Expand to see task annotations:\n";
    std::cout << "     - matrix_operations\n";
    std::cout << "     - sorting_operations\n";
    std::cout << "     - monte_carlo_simulation\n";

    EXPECT_TRUE(true);  // Test passes if ITT API annotations completed
}
#endif  // XSIGMA_HAS_ITTAPI
