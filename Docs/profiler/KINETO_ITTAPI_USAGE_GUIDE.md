# Kineto and ITTAPI Usage Guide and Troubleshooting

## 1. PYTORCH USAGE EXAMPLES

### Basic Kineto Profiling

```python
import torch
import torch.profiler as profiler

# Simple profiling
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
) as prof:
    # Your code here
    x = torch.randn(100, 100, device='cuda')
    y = torch.mm(x, x)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Advanced Kineto Configuration

```python
import torch.profiler as profiler

# Custom configuration
config = profiler.ExperimentalConfig(
    custom_profiler_config="""
    ACTIVITIES_WARMUP_PERIOD_SECS=0
    CUPTI_PROFILER_METRICS=kineto__cuda_core_flops
    CUPTI_PROFILER_ENABLE_PER_KERNEL=true
    """
)

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    experimental_config=config,
    on_trace_ready=profiler.tensorboard_trace_handler('./logs')
) as prof:
    # Your code
    pass
```

### ITT API Usage (Python)

```python
import torch.profiler as profiler

# Check if ITT is available
if profiler._itt.is_available():
    # Push range
    profiler._itt.rangePush("my_operation")
    
    # Your code
    x = torch.randn(100, 100)
    y = torch.mm(x, x)
    
    # Pop range
    profiler._itt.rangePop()
    
    # Mark event
    profiler._itt.mark("checkpoint")
```

---

## 2. PYTORCH C++ USAGE EXAMPLES

### Kineto C++ API

```cpp
#include <libkineto.h>

int main() {
    // Initialize Kineto
    libkineto_init(false, true);  // false=GPU, true=logOnError
    
    // Get profiler
    auto& profiler = libkineto::api().activityProfiler();
    
    // Initialize if needed
    libkineto::api().initProfilerIfRegistered();
    
    // Prepare trace
    std::set<libkineto::ActivityType> activities;
    activities.insert(libkineto::ActivityType::CPU_OP);
    activities.insert(libkineto::ActivityType::CUDA_KERNEL);
    
    profiler.prepareTrace(activities);
    
    // Start profiling
    profiler.startTrace();
    
    // Your code here
    // ... GPU operations ...
    
    // Stop and get trace
    auto trace = profiler.stopTrace();
    
    // Save trace
    trace->save("trace.json");
    
    return 0;
}
```

### ITT C++ API

```cpp
#include <ittnotify.h>

int main() {
    // Create domain
    __itt_domain* domain = __itt_domain_create("MyApp");
    
    // Create string handle
    __itt_string_handle* task_handle = 
        __itt_string_handle_create("MyTask");
    
    // Begin task
    __itt_task_begin(domain, __itt_null, __itt_null, task_handle);
    
    // Your code here
    // ... work ...
    
    // End task
    __itt_task_end(domain);
    
    // Mark event
    __itt_string_handle* mark_handle = 
        __itt_string_handle_create("Checkpoint");
    __itt_task_begin(domain, __itt_null, __itt_null, mark_handle);
    __itt_task_end(domain);
    
    return 0;
}
```

---

## 3. XSIGMA USAGE EXAMPLES

### Kineto Wrapper Usage

```cpp
#include "profiler/kineto_profiler.h"

int main() {
    // Create profiler
    auto profiler = xsigma::kineto_profiler::create();
    
    if (profiler) {
        // Start profiling
        if (profiler->start_profiling()) {
            // Your code here
            // ... operations ...
            
            // Stop profiling
            profiler->stop_profiling();
        }
    } else {
        // Graceful degradation - Kineto not available
        std::cerr << "Kineto profiler not available\n";
    }
    
    return 0;
}
```

### Kineto with Configuration

```cpp
#include "profiler/kineto_profiler.h"

int main() {
    // Create configuration
    xsigma::kineto_profiler::profiling_config config;
    config.enable_cpu_tracing = true;
    config.enable_gpu_tracing = false;
    config.enable_memory_profiling = false;
    config.output_dir = "./kineto_profiles";
    config.trace_name = "xsigma_trace";
    config.max_activities = 0;  // Unlimited
    
    // Create profiler with config
    auto profiler = xsigma::kineto_profiler::create_with_config(config);
    
    if (profiler && profiler->start_profiling()) {
        // Your code
        profiler->stop_profiling();
    }
    
    return 0;
}
```

### ITT API Usage (XSigma)

```cpp
#ifdef XSIGMA_HAS_ITTAPI
#include <ittnotify.h>

int main() {
    // Create domain
    __itt_domain* domain = __itt_domain_create("XSigmaApp");
    
    // Create string handle
    __itt_string_handle* handle = 
        __itt_string_handle_create("ProcessingTask");
    
    // Begin task
    __itt_task_begin(domain, __itt_null, __itt_null, handle);
    
    // Your code
    // ... processing ...
    
    // End task
    __itt_task_end(domain);
    
    return 0;
}
#else
    #error "ITT API not available - enable XSIGMA_ENABLE_ITTAPI"
#endif
```

---

## 4. BUILD CONFIGURATION

### PyTorch Build

```bash
# Enable Kineto and ITT
XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_ITT=1 python setup.py install

# With CUDA support
XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_ITT=1 XSIGMA_ENABLE_CUDA=1 python setup.py install

# With ROCm support
XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_ITT=1 USE_ROCM=1 python setup.py install
```

### XSigma Build

```bash
# Enable Kineto (default)
cmake -DXSIGMA_ENABLE_KINETO=ON ..

# Enable ITT API
cmake -DXSIGMA_ENABLE_ITTAPI=ON ..

# Both enabled
cmake -DXSIGMA_ENABLE_KINETO=ON -DXSIGMA_ENABLE_ITTAPI=ON ..

# Disable Kineto
cmake -DXSIGMA_ENABLE_KINETO=OFF ..
```

---

## 5. ENVIRONMENT VARIABLES

### Kineto Environment Variables

```bash
# Set log level (0=VERBOSE, higher=less verbose)
export KINETO_LOG_LEVEL=2

# Enable CUDA injection mode
export CUDA_INJECTION64_PATH=/path/to/libkineto.so

# Daemon mode (Linux)
export KINETO_DAEMON_SOCKET_ADDR=localhost:23645
```

### ITT Environment Variables

```bash
# VTune collection control
export INTEL_LIBITTNOTIFY64=/path/to/libittnotify64.so

# Enable ITT collection
export INTEL_VTUNE_PROFILER=1
```

---

## 6. TROUBLESHOOTING

### Kineto Issues

#### Issue: "Kineto not initialized"
**Cause**: Kineto library not linked or initialization failed
**Solution**:
```bash
# Verify Kineto is enabled
cmake -LA | grep XSIGMA_ENABLE_KINETO

# Check if libkineto is available
find /usr -name "libkineto*" 2>/dev/null

# Rebuild with verbose output
XSIGMA_ENABLE_KINETO=1 python setup.py install -v
```

#### Issue: "CUPTI not found"
**Cause**: CUDA toolkit not installed or CUPTI not available
**Solution**:
```bash
# Install CUDA toolkit
# Verify CUPTI
ls $CUDA_HOME/extras/CUPTI/lib64/

# Build without CUPTI
XSIGMA_ENABLE_KINETO=1 XSIGMA_ENABLE_CUDA=0 python setup.py install
```

#### Issue: "Profiler not initialized"
**Cause**: libkineto_init() not called
**Solution**:
```cpp
// Ensure initialization
if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(false, true);
}
if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
}
```

### ITT API Issues

#### Issue: "ITT API not available"
**Cause**: XSIGMA_ENABLE_ITT not enabled or ittapi not found
**Solution**:
```bash
# Enable ITT
XSIGMA_ENABLE_ITT=1 python setup.py install

# Verify ittapi
find /usr -name "libittnotify*" 2>/dev/null

# Check Python bindings
python -c "import torch; print(torch.profiler._itt.is_available())"
```

#### Issue: "VTune not collecting ITT events"
**Cause**: VTune not installed or ITT library not found
**Solution**:
```bash
# Install Intel VTune Profiler
# Verify ITT library path
export INTEL_LIBITTNOTIFY64=/path/to/libittnotify64.so

# Run with VTune
vtune -collect hotspots -app ./my_app
```

#### Issue: "ITT domain not created"
**Cause**: ITT API not initialized
**Solution**:
```cpp
#ifdef XSIGMA_ENABLE_ITT
    __itt_domain* domain = __itt_domain_create("MyDomain");
    if (domain) {
        // Use domain
    }
#endif
```

### XSigma Issues

#### Issue: "XSIGMA_HAS_KINETO not defined"
**Cause**: Kineto not enabled or target not available
**Solution**:
```bash
# Enable Kineto
cmake -DXSIGMA_ENABLE_KINETO=ON ..

# Check if target exists
cmake -LA | grep XSIGMA_ENABLE_KINETO
```

#### Issue: "ITT API shared library not found"
**Cause**: XSIGMA_ENABLE_ITTAPI not enabled or library not built
**Solution**:
```bash
# Enable ITT API
cmake -DXSIGMA_ENABLE_ITTAPI=ON ..

# Verify shared library
find . -name "libittnotify*" -o -name "ittnotify.dll"

# Check library path
ldd ./CoreCxxTests | grep ittnotify
```

---

## 7. PERFORMANCE CONSIDERATIONS

### Kineto Overhead
- CPU profiling: ~5-10% overhead
- GPU profiling: ~2-5% overhead
- Memory profiling: ~10-20% overhead

### ITT API Overhead
- Task annotations: ~1-2% overhead
- Minimal impact when VTune not collecting
- Negligible when ITT API disabled

### Optimization Tips
1. Disable memory profiling if not needed
2. Use CPU-only profiling for CPU-bound workloads
3. Limit number of activities tracked
4. Use sampling instead of full tracing for long runs
5. Disable ITT API when not profiling

---

## 8. PROFILING BEST PRACTICES

### PyTorch
1. Warm up before profiling (CUDA initialization)
2. Use `record_shapes=True` for detailed analysis
3. Profile in evaluation mode for inference
4. Use `on_trace_ready` callback for automatic saving
5. Analyze with TensorBoard or Chrome trace viewer

### XSigma
1. Check profiler availability before use
2. Handle graceful degradation
3. Use try-catch for error handling
4. Clean up profiler resources
5. Verify output directory exists

### ITT API
1. Create domains for logical grouping
2. Use string handles for efficiency
3. Avoid creating handles in hot loops
4. Use task nesting for hierarchical profiling
5. Synchronize with VTune collection

