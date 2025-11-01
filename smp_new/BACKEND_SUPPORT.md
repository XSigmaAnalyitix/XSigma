# SMP_NEW Backend Support

## Overview

XSigma's `smp_new` module now supports multiple threading backends, matching PyTorch's CPU threading architecture. This provides flexibility in choosing the best parallelism strategy for your use case.

## Supported Backends

### 1. Native Backend (std::thread)
- **Type**: `BackendType::NATIVE` (value: 0)
- **Implementation**: Uses `std::thread` for worker threads
- **Features**:
  - Master-worker pattern for reduced overhead
  - Lazy thread pool initialization
  - Separate intra-op and inter-op thread pools
  - Work-stealing for load balancing
  - Robust exception handling
  - NUMA support
- **Best For**: 
  - Portable code that doesn't require external dependencies
  - Fine-grained control over thread management
  - Systems without OpenMP support
- **Performance**: Excellent, minimal overhead

### 2. OpenMP Backend
- **Type**: `BackendType::OPENMP` (value: 1)
- **Implementation**: Uses `#pragma omp` directives
- **Features**:
  - Directive-based parallelism
  - Integration with MKL and Intel OpenMP
  - Automatic thread pool management
  - Environment variable configuration (OMP_NUM_THREADS, MKL_NUM_THREADS)
  - Nested parallelism support
  - Exception handling
- **Best For**:
  - Code that benefits from MKL integration
  - Systems with Intel OpenMP
  - Applications requiring nested parallelism
  - Performance-critical code
- **Performance**: Excellent, especially with MKL
- **Requirements**: OpenMP support at compile time

### 3. Auto Backend
- **Type**: `BackendType::AUTO` (value: 2)
- **Behavior**: Automatically selects the best available backend
  - Prefers OpenMP if available
  - Falls back to Native if OpenMP is not available
- **Best For**: 
  - Portable code that adapts to available resources
  - Default choice for most applications

## API Usage

### Setting the Backend

```cpp
#include "smp_new/parallel/parallel_api.h"

// Set backend before any parallel work
xsigma::smp_new::parallel::set_backend(0);  // NATIVE
xsigma::smp_new::parallel::set_backend(1);  // OPENMP
xsigma::smp_new::parallel::set_backend(2);  // AUTO (recommended)
```

### Checking Backend Availability

```cpp
// Check if OpenMP is available
if (xsigma::smp_new::parallel::is_openmp_available()) {
    xsigma::smp_new::parallel::set_backend(1);  // Use OpenMP
} else {
    xsigma::smp_new::parallel::set_backend(0);  // Use Native
}
```

### Getting Current Backend

```cpp
int backend = xsigma::smp_new::parallel::get_backend();
// 0 = NATIVE, 1 = OPENMP, 2 = AUTO
```

### Backend Information

```cpp
#include "smp_new/native/parallel_native.h"

// Get detailed backend information
std::string info = xsigma::smp_new::native::GetBackendInfo();
std::cout << info << std::endl;

// Get specific backend info
std::string native_info = xsigma::smp_new::native::GetNativeBackendInfo();
std::string openmp_info = xsigma::smp_new::native::GetOpenMPBackendInfo();
```

## Compilation

### CMake Configuration

The `smp_new` module automatically detects and enables available backends:

```bash
# Build with OpenMP support (if available)
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Build with explicit OpenMP
cmake -DCMAKE_BUILD_TYPE=Release -DOpenMP_CXX_FOUND=ON ..
make

# Build with MKL support
cmake -DCMAKE_BUILD_TYPE=Release -DMKL_FOUND=ON ..
make
```

### Compiler Flags

**For OpenMP support:**
```bash
# GCC/Clang
-fopenmp

# Intel Compiler
-qopenmp

# MSVC
/openmp
```

**For MKL support:**
```bash
# Link with MKL
-lmkl_core -lmkl_sequential -lmkl_intel_lp64
```

## Environment Variables

### OpenMP Configuration

```bash
# Set number of OpenMP threads
export OMP_NUM_THREADS=8

# Set MKL threads (if using MKL)
export MKL_NUM_THREADS=8

# Disable dynamic thread adjustment
export OMP_DYNAMIC=false

# Set thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

### Native Backend Configuration

The native backend respects the same environment variables for consistency:
- `OMP_NUM_THREADS` - Number of threads
- `MKL_NUM_THREADS` - MKL thread count

## Performance Considerations

### Native Backend
- **Pros**: 
  - Minimal overhead
  - Portable
  - Fine-grained control
- **Cons**: 
  - No MKL integration
  - Manual thread management

### OpenMP Backend
- **Pros**:
  - MKL integration
  - Automatic thread management
  - Nested parallelism
- **Cons**:
  - Requires OpenMP support
  - Less fine-grained control

## Migration from SMP to SMP_NEW

### Backend Selection

```cpp
// Old SMP code (functor-based)
struct MyFunctor {
    void operator()(int first, int last) { }
};
MyFunctor f;
xsigma::tools::For(0, N, grain_size, f);

// New SMP_NEW code (lambda-based, with backend selection)
xsigma::smp_new::parallel::set_backend(2);  // AUTO
xsigma::smp_new::parallel::parallel_for(
    0, N, grain_size,
    [&](int64_t b, int64_t e) { }
);
```

## Troubleshooting

### OpenMP Not Available

If you get an error about OpenMP not being available:

1. Check if OpenMP is installed:
   ```bash
   # Linux
   apt-get install libomp-dev
   
   # macOS
   brew install libomp
   
   # Windows
   # Use MSVC with OpenMP support
   ```

2. Rebuild with OpenMP support:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make clean
   make
   ```

### Performance Issues

1. Check thread count:
   ```cpp
   std::cout << "Intra-op threads: " 
             << xsigma::smp_new::parallel::get_num_intraop_threads() << std::endl;
   std::cout << "Inter-op threads: " 
             << xsigma::smp_new::parallel::get_num_interop_threads() << std::endl;
   ```

2. Set optimal thread count:
   ```cpp
   xsigma::smp_new::parallel::set_num_intraop_threads(8);
   xsigma::smp_new::parallel::set_num_interop_threads(4);
   ```

3. Check backend info:
   ```cpp
   std::cout << xsigma::smp_new::native::GetBackendInfo() << std::endl;
   ```

## See Also

- [README.md](README.md) - Overview and quick start
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage examples
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration instructions

