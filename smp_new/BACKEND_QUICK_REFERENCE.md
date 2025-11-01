# SMP_NEW Backend Quick Reference

## Quick Start

```cpp
#include "smp_new/parallel/parallel_api.h"

// Set backend (do this first!)
xsigma::smp_new::parallel::set_backend(2);  // AUTO

// Use parallel APIs
xsigma::smp_new::parallel::parallel_for(
    0, 1000, 100,
    [&](int64_t begin, int64_t end) {
        // Your code here
    }
);
```

## Backend Types

| Value | Name | Description |
|-------|------|-------------|
| 0 | NATIVE | std::thread-based (default) |
| 1 | OPENMP | OpenMP directives |
| 2 | AUTO | Auto-select best available |

## API Reference

### Backend Selection

```cpp
// Set backend
void set_backend(int backend);

// Get current backend
int get_backend();

// Check OpenMP availability
bool is_openmp_available();
```

### Backend Information

```cpp
#include "smp_new/native/parallel_native.h"

// Get detailed info
std::string GetBackendInfo();
std::string GetNativeBackendInfo();
std::string GetOpenMPBackendInfo();
```

### Thread Configuration

```cpp
// Set thread counts
void set_num_intraop_threads(int nthreads);
void set_num_interop_threads(int nthreads);

// Get thread counts
size_t get_num_intraop_threads();
size_t get_num_interop_threads();
```

## Common Patterns

### Pattern 1: Auto Backend (Recommended)

```cpp
xsigma::smp_new::parallel::set_backend(2);  // AUTO
// Rest of code uses best available backend
```

### Pattern 2: Conditional Backend

```cpp
if (xsigma::smp_new::parallel::is_openmp_available()) {
    xsigma::smp_new::parallel::set_backend(1);  // OPENMP
} else {
    xsigma::smp_new::parallel::set_backend(0);  // NATIVE
}
```

### Pattern 3: Explicit Backend

```cpp
// Always use native
xsigma::smp_new::parallel::set_backend(0);

// Or always use OpenMP (if available)
if (xsigma::smp_new::parallel::is_openmp_available()) {
    xsigma::smp_new::parallel::set_backend(1);
}
```

### Pattern 4: Backend Info

```cpp
std::cout << xsigma::smp_new::native::GetBackendInfo() << std::endl;
```

## Compilation

### With OpenMP

```bash
g++ -std=c++17 -fopenmp -O3 code.cpp -o code
```

### With MKL

```bash
g++ -std=c++17 -fopenmp -O3 code.cpp -o code \
    -I/opt/intel/mkl/include \
    -L/opt/intel/mkl/lib/intel64 \
    -lmkl_core -lmkl_sequential -lmkl_intel_lp64
```

## Environment Variables

```bash
# Number of threads
export OMP_NUM_THREADS=8

# Thread affinity
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Dynamic adjustment
export OMP_DYNAMIC=false
```

## Troubleshooting

### OpenMP Not Available

```cpp
if (!xsigma::smp_new::parallel::is_openmp_available()) {
    std::cerr << "OpenMP not available, using native backend" << std::endl;
    xsigma::smp_new::parallel::set_backend(0);  // NATIVE
}
```

### Check Backend Info

```cpp
std::cout << xsigma::smp_new::native::GetBackendInfo() << std::endl;
```

### Check Thread Count

```cpp
std::cout << "Intra-op: " 
          << xsigma::smp_new::parallel::get_num_intraop_threads() << std::endl;
std::cout << "Inter-op: " 
          << xsigma::smp_new::parallel::get_num_interop_threads() << std::endl;
```

## Performance Tips

1. **Use AUTO backend** for best portability
2. **Set thread count early** before parallel work
3. **Use OpenMP** if MKL integration is needed
4. **Use NATIVE** for fine-grained control
5. **Benchmark** to find best backend for your workload

## Examples

See `BACKEND_EXAMPLES.md` for:
- Auto backend selection
- Native backend usage
- OpenMP backend usage
- Conditional backend selection
- Matrix multiplication
- Nested parallelism
- Backend switching
- Performance comparison

## Full Documentation

See `BACKEND_SUPPORT.md` for:
- Detailed backend descriptions
- API documentation
- Compilation instructions
- Environment variables
- Troubleshooting guide
- Migration guide

## Key Points

✅ Set backend before parallel work  
✅ Use AUTO for best portability  
✅ Check OpenMP availability  
✅ Configure thread count early  
✅ Use environment variables for tuning  
✅ Benchmark for your workload  

## Status

✅ Production Ready  
✅ Fully Tested  
✅ Backward Compatible  
✅ PyTorch Compatible  

