# SMP_NEW Backend Examples

## Example 1: Auto Backend Selection

```cpp
#include "smp_new/parallel/parallel_api.h"
#include "smp_new/native/parallel_native.h"

int main() {
    // Use AUTO backend - automatically selects best available
    xsigma::smp_new::parallel::set_backend(2);  // AUTO
    
    // Print backend info
    std::cout << xsigma::smp_new::native::GetBackendInfo() << std::endl;
    
    // Use parallel_for
    std::vector<int> data(1000);
    xsigma::smp_new::parallel::parallel_for(
        0, 1000, 100,
        [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                data[i] = i * 2;
            }
        }
    );
    
    return 0;
}
```

## Example 2: Native Backend (std::thread)

```cpp
#include "smp_new/parallel/parallel_api.h"

int main() {
    // Explicitly use native backend
    xsigma::smp_new::parallel::set_backend(0);  // NATIVE
    
    // Configure thread count
    xsigma::smp_new::parallel::set_num_intraop_threads(8);
    xsigma::smp_new::parallel::set_num_interop_threads(4);
    
    // Parallel reduction
    int sum = xsigma::smp_new::parallel::parallel_reduce(
        0, 1000, 100,
        0,
        [](int64_t begin, int64_t end, int ident) {
            int local_sum = ident;
            for (int64_t i = begin; i < end; ++i) {
                local_sum += i;
            }
            return local_sum;
        },
        [](int a, int b) { return a + b; }
    );
    
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
```

## Example 3: OpenMP Backend

```cpp
#include "smp_new/parallel/parallel_api.h"
#include "smp_new/native/parallel_native.h"

int main() {
    // Check if OpenMP is available
    if (!xsigma::smp_new::parallel::is_openmp_available()) {
        std::cerr << "OpenMP not available!" << std::endl;
        return 1;
    }
    
    // Use OpenMP backend
    xsigma::smp_new::parallel::set_backend(1);  // OPENMP
    
    // Print OpenMP info
    std::cout << xsigma::smp_new::native::GetOpenMPBackendInfo() << std::endl;
    
    // Parallel for with OpenMP
    std::vector<double> matrix(1000 * 1000);
    xsigma::smp_new::parallel::parallel_for(
        0, 1000, 100,
        [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    matrix[i * 1000 + j] = i * j;
                }
            }
        }
    );
    
    return 0;
}
```

## Example 4: Conditional Backend Selection

```cpp
#include "smp_new/parallel/parallel_api.h"
#include "smp_new/native/parallel_native.h"

int main() {
    // Select backend based on availability and performance
    int backend = 0;  // Default to NATIVE
    
    if (xsigma::smp_new::parallel::is_openmp_available()) {
        // Check if we want MKL integration
        #ifdef AT_MKL_ENABLED
        backend = 1;  // Use OpenMP for MKL
        #endif
    }
    
    xsigma::smp_new::parallel::set_backend(backend);
    
    std::cout << "Using backend: " 
              << xsigma::smp_new::parallel::get_backend() << std::endl;
    
    return 0;
}
```

## Example 5: Matrix Multiplication with Backend Selection

```cpp
#include "smp_new/parallel/parallel_api.h"

void matrix_multiply(
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C,
    int N,
    int backend = 2)  // AUTO by default
{
    // Set backend
    xsigma::smp_new::parallel::set_backend(backend);
    
    // Parallel matrix multiplication
    xsigma::smp_new::parallel::parallel_for(
        0, N, 64,
        [&](int64_t i_begin, int64_t i_end) {
            for (int64_t i = i_begin; i < i_end; ++i) {
                for (int j = 0; j < N; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < N; ++k) {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    );
}

int main() {
    const int N = 1024;
    std::vector<double> A(N * N, 1.0);
    std::vector<double> B(N * N, 2.0);
    std::vector<double> C(N * N, 0.0);
    
    // Try with AUTO backend
    matrix_multiply(A, B, C, N, 2);  // AUTO
    
    return 0;
}
```

## Example 6: Nested Parallelism

```cpp
#include "smp_new/parallel/parallel_api.h"

int main() {
    // Use OpenMP for better nested parallelism support
    if (xsigma::smp_new::parallel::is_openmp_available()) {
        xsigma::smp_new::parallel::set_backend(1);  // OPENMP
    } else {
        xsigma::smp_new::parallel::set_backend(0);  // NATIVE
    }
    
    // Outer parallel loop
    xsigma::smp_new::parallel::parallel_for(
        0, 100, 10,
        [&](int64_t i_begin, int64_t i_end) {
            // Inner parallel loop (nested)
            xsigma::smp_new::parallel::parallel_for(
                0, 100, 10,
                [&](int64_t j_begin, int64_t j_end) {
                    for (int64_t i = i_begin; i < i_end; ++i) {
                        for (int64_t j = j_begin; j < j_end; ++j) {
                            // Do work
                        }
                    }
                }
            );
        }
    );
    
    return 0;
}
```

## Example 7: Backend Switching

```cpp
#include "smp_new/parallel/parallel_api.h"
#include "smp_new/native/parallel_native.h"

int main() {
    std::vector<int> data(10000);
    
    // First, use native backend
    xsigma::smp_new::parallel::set_backend(0);  // NATIVE
    xsigma::smp_new::parallel::parallel_for(
        0, 10000, 1000,
        [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                data[i] = i;
            }
        }
    );
    
    std::cout << "Native backend info:\n" 
              << xsigma::smp_new::native::GetNativeBackendInfo() << std::endl;
    
    // Switch to OpenMP if available
    if (xsigma::smp_new::parallel::is_openmp_available()) {
        xsigma::smp_new::parallel::set_backend(1);  // OPENMP
        
        std::cout << "OpenMP backend info:\n" 
                  << xsigma::smp_new::native::GetOpenMPBackendInfo() << std::endl;
    }
    
    return 0;
}
```

## Example 8: Performance Comparison

```cpp
#include "smp_new/parallel/parallel_api.h"
#include <chrono>

void benchmark_backend(int backend, const std::string& name) {
    xsigma::smp_new::parallel::set_backend(backend);
    
    std::vector<double> data(1000000);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    xsigma::smp_new::parallel::parallel_for(
        0, 1000000, 10000,
        [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                data[i] = std::sqrt(i);
            }
        }
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << name << ": " << duration.count() << " ms" << std::endl;
}

int main() {
    benchmark_backend(0, "Native");
    
    if (xsigma::smp_new::parallel::is_openmp_available()) {
        benchmark_backend(1, "OpenMP");
    }
    
    benchmark_backend(2, "Auto");
    
    return 0;
}
```

## Compilation

```bash
# Compile with OpenMP support
g++ -std=c++17 -fopenmp -O3 example.cpp -o example

# Compile with MKL support
g++ -std=c++17 -fopenmp -O3 example.cpp -o example \
    -I/opt/intel/mkl/include \
    -L/opt/intel/mkl/lib/intel64 \
    -lmkl_core -lmkl_sequential -lmkl_intel_lp64
```

## Environment Variables

```bash
# Set number of threads
export OMP_NUM_THREADS=8

# Set thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Disable dynamic thread adjustment
export OMP_DYNAMIC=false

# Run example
./example
```

