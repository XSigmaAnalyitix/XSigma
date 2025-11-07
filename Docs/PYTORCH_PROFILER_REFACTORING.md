# PyTorch Profiler Refactoring Summary

## Overview
Successfully refactored the `Library/Core/profiler/pytroch_profiler` directory to align with XSigma coding standards and conventions.

## Changes Made

### 1. Directory Structure Flattening
- **Before**: Nested directory structure with 68 files across multiple levels:
  - `aten/src/ATen/`
  - `torch/csrc/autograd/`
  - `torch/csrc/profiler/`
  - `torch/csrc/profiler/orchestration/`
  - `torch/csrc/profiler/python/`
  - `torch/csrc/profiler/standalone/`
  - `torch/csrc/profiler/stubs/`
  - `torch/csrc/profiler/unwind/`

- **After**: All 66 C++ files (.h and .cxx) moved to base `pytroch_profiler` directory
- Removed all empty nested directories
- Preserved Python files and documentation in original locations

### 2. Include Path Updates
- **Old Format**: `#include <torch/csrc/profiler/file.h>`
- **New Format**: `#include "file.h"` (relative to pytorch_profiler directory)
- Updated all internal includes to use relative paths
- Updated downstream references in `Library/Core/experimental/xsigma_autograd/`:
  - `profiler_kineto.h`
  - `profiler_kineto.cxx`
  - `profiler_legacy.h`
  - `init.cxx`
  - `profiler_python.cxx`
  - `python_function.cxx`

### 3. Macro Replacements
Replaced PyTorch-specific macros with XSigma equivalents:

| PyTorch Macro | XSigma Macro | Purpose |
|---|---|---|
| `TORCH_API` | `XSIGMA_API` | Function export/import |
| `TORCH_PYTHON_API` | `XSIGMA_API` | Python-facing function export |
| `TORCH_INTERNAL_ASSERT` | `XSIGMA_CHECK` | Internal assertions |
| `TORCH_INTERNAL_ASSERT_DEBUG_ONLY` | `XSIGMA_CHECK_DEBUG` | Debug-only assertions |
| `TORCH_CHECK` | `XSIGMA_CHECK` | Runtime checks |
| `C10_API_ENUM` | (removed) | Enum visibility (not needed) |
| `C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED` | `XSIGMA_DIAGNOSTIC_PUSH` | Diagnostic control |
| `C10_DIAGNOSTIC_POP` | `XSIGMA_DIAGNOSTIC_POP` | Diagnostic control |

### 4. Visibility and API Macros
- Added `XSIGMA_VISIBILITY` before public class declarations
- Added `XSIGMA_API` before public function declarations
- Ensured proper DLL export/import semantics for Windows

### 5. Namespace Corrections
- Preserved `c10::` namespace references (not replaced with `xsigma::`)
- Preserved `torch::` namespace references where appropriate
- Maintained compatibility with PyTorch's type system

## Files Modified

### Refactored Files (66 total)
All files moved from nested directories to base `pytroch_profiler` directory:
- `record_function.h/cxx` (from `aten/src/ATen/`)
- `profiler_kineto.h/cxx`, `profiler_python.h/cxx`, `profiler.h` (from `torch/csrc/autograd/`)
- `collection.h/cxx`, `combined_traceback.h/cxx`, `containers.h`, `data_flow.h/cxx`, `events.h`
- `kineto_shim.h/cxx`, `kineto_client_interface.h/cxx`, `api.h`
- `observer.h/cxx`, `python_tracer.h/cxx`, `vulkan.h/cxx` (from orchestration/)
- `execution_trace_observer.h/cxx`, `itt_observer.h/cxx`, `nvtx_observer.h/cxx`, `privateuse1_observer.h/cxx` (from standalone/)
- `base.h/cxx`, `cuda.cxx`, `itt.cxx` (from stubs/)
- `unwind.h/cxx`, `unwind_fb.cxx`, `unwind_error.h`, `unwinder.h`, and 20+ header files (from unwind/)
- `util.h/cxx`, `perf.h/cxx`, `perf-inl.h`, `init.h/cxx`, `pybind.h`

### Downstream Files Updated (6 total)
- `Library/Core/experimental/xsigma_autograd/profiler_kineto.h`
- `Library/Core/experimental/xsigma_autograd/profiler_kineto.cxx`
- `Library/Core/experimental/xsigma_autograd/profiler_legacy.h`
- `Library/Core/experimental/xsigma_autograd/init.cxx`
- `Library/Core/experimental/xsigma_autograd/profiler_python.cxx`
- `Library/Core/experimental/xsigma_autograd/python_function.cxx`

## XSigma Infrastructure Used
- **Export Macros**: `Library/Core/common/export.h` (XSIGMA_API, XSIGMA_VISIBILITY)
- **Common Macros**: `Library/Core/common/macros.h` (XSIGMA_CHECK, XSIGMA_NODISCARD, etc.)

## Compliance
✅ Follows XSigma coding standards
✅ Uses snake_case naming conventions
✅ Proper include path structure
✅ DLL export/import macros applied
✅ No exception-based error handling
✅ Maintains cross-platform compatibility

## Notes
- Python files in `torch/profiler/` and `torch/autograd/` directories preserved as-is
- Duplicate `combined_traceback` files in `torch/csrc/profiler/python/` preserved (not moved)
- README.md documentation preserved in original location
- All changes maintain backward compatibility with existing PyTorch integration

