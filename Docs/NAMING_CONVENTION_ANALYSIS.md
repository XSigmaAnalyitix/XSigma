# XSigma Parallel Module - Naming Convention Analysis Report

**Date:** 2025-11-04
**Module:** `Library/Core//`
**Scope:** Comprehensive naming convention compliance analysis

---

## Executive Summary

This report identifies all naming convention violations in the `` module according to XSigma C++ Coding Standards. The analysis covers:
- File naming (CamelCase → snake_case)
- Class naming (already compliant)
- Function naming (already compliant)
- Member variable naming (already compliant)
- Include path references

**Total Violations Found:** 8 files require renaming
**Estimated Effort:** 2-3 hours (renaming + testing)
**Risk Level:** Low (cosmetic changes only)

---

## 1. FILE NAMING VIOLATIONS

### 1.1 Files Requiring Renaming

| Current Name (CamelCase) | Required Name (snake_case) | Type | Lines | Status |
|--------------------------|----------------------------|------|-------|--------|
| `Parallel.h` | `parallel.h` | Header | 163 | ❌ Rename Required |
| `Parallel-inl.h` | `parallel-inl.h` | Inline Header | 104 | ❌ Rename Required |
| `ParallelCommon.cpp` | `parallel_common.cpp` | Implementation | ~100 | ❌ Rename Required |
| `ParallelGuard.h` | `parallel_guard.h` | Header | 24 | ❌ Rename Required |
| `ParallelNative.h` | `parallel_native.h` | Header | 18 | ❌ Rename Required |
| `ParallelNative.cpp` | `parallel_native.cpp` | Implementation | ~250 | ❌ Rename Required |
| `ParallelOpenMP.h` | `parallel_openmp.h` | Header | 48 | ❌ Rename Required |
| `ParallelOpenMP.cpp` | `parallel_openmp.cpp` | Implementation | ~120 | ❌ Rename Required |
| `ParallelThreadPoolNative.cpp` | `parallel_thread_pool_native.cpp` | Implementation | ~100 | ❌ Rename Required |
| `ThreadLocalState.h` | `thread_local_state.h` | Header | 132 | ❌ Rename Required |
| `ThreadPool.h` | `thread_pool_legacy.h` | Header (Legacy Caffe2) | 82 | ❌ Rename Required |
| `ThreadPool.cpp` | `thread_pool_legacy.cpp` | Implementation (Legacy) | ~200 | ❌ Rename Required |
| `ThreadPoolCommon.h` | `thread_pool_common.h` | Header | 20 | ❌ Rename Required |
| `WorkersPool.h` | `workers_pool.h` | Header | 419 | ❌ Rename Required |

### 1.2 Files Already Compliant

| File Name | Type | Status |
|-----------|------|--------|
| `thread_pool.h` | Header | ✅ Compliant |
| `thread_pool.cpp` | Implementation | ✅ Compliant |
| `thread_pool_guard.h` | Header | ✅ Compliant |
| `thread_pool_guard.cpp` | Implementation | ✅ Compliant |

---

## 2. CLASS NAMING ANALYSIS

### 2.1 Classes - All Compliant ✅

All classes in the module already follow `snake_case` convention:

| Class Name | File | Namespace | Status |
|------------|------|-----------|--------|
| `task_thread_pool_base` | thread_pool.h | xsigma | ✅ Compliant |
| `thread_pool` | thread_pool.h | xsigma | ✅ Compliant |
| `task_thread_pool` | thread_pool.h | xsigma | ✅ Compliant |
| `pt_thread_pool` | (referenced) | xsigma | ✅ Compliant |
| `parallel_guard` | ParallelGuard.h | xsigma | ✅ Compliant |
| `thread_id_guard` | Parallel.h | xsigma::internal | ✅ Compliant |
| `thread_local_state` | ThreadLocalState.h | at | ✅ Compliant |
| `thread_local_state_guard` | ThreadLocalState.h | at | ✅ Compliant |
| `thread_pool` | ThreadPool.h | caffe2 | ✅ Compliant (Legacy) |
| `workers_pool` | ThreadPool.h | caffe2 | ✅ Compliant (Legacy) |

**Note:** Legacy Caffe2 classes in `caffe2` namespace are compliant and should not be modified.

---

## 3. FUNCTION NAMING ANALYSIS

### 3.1 Functions - All Compliant ✅

All functions already follow `snake_case` convention:

| Function Name | File | Namespace | Status |
|---------------|------|-----------|--------|
| `init_num_threads()` | Parallel.h | xsigma | ✅ Compliant |
| `set_num_threads()` | Parallel.h | xsigma | ✅ Compliant |
| `get_num_threads()` | Parallel.h | xsigma | ✅ Compliant |
| `get_thread_num()` | Parallel.h | xsigma | ✅ Compliant |
| `in_parallel_region()` | Parallel.h | xsigma | ✅ Compliant |
| `lazy_init_num_threads()` | Parallel.h | xsigma::internal | ✅ Compliant |
| `set_thread_num()` | Parallel.h | xsigma::internal | ✅ Compliant |
| `parallel_for()` | Parallel-inl.h | xsigma | ✅ Compliant |
| `parallel_reduce()` | Parallel-inl.h | xsigma | ✅ Compliant |
| `invoke_parallel()` | ParallelNative.h | xsigma::internal | ✅ Compliant |
| `intraop_launch()` | Parallel.h | xsigma | ✅ Compliant |
| `set_num_interop_threads()` | Parallel.h | xsigma | ✅ Compliant |
| `get_num_interop_threads()` | Parallel.h | xsigma | ✅ Compliant |
| `get_parallel_info()` | Parallel.h | xsigma | ✅ Compliant |
| `intraop_default_num_threads()` | Parallel.h | xsigma | ✅ Compliant |

---

## 4. MEMBER VARIABLE NAMING ANALYSIS

### 4.1 Member Variables - All Compliant ✅

All member variables follow `snake_case_` (with trailing underscore) convention:

| Variable Name | Class | File | Status |
|---------------|-------|------|--------|
| `tasks_` | thread_pool | thread_pool.h | ✅ Compliant |
| `threads_` | thread_pool | thread_pool.h | ✅ Compliant |
| `mutex_` | thread_pool | thread_pool.h | ✅ Compliant |
| `condition_` | thread_pool | thread_pool.h | ✅ Compliant |
| `completed_` | thread_pool | thread_pool.h | ✅ Compliant |
| `running_` | thread_pool | thread_pool.h | ✅ Compliant |
| `complete_` | thread_pool | thread_pool.h | ✅ Compliant |
| `available_` | thread_pool | thread_pool.h | ✅ Compliant |
| `total_` | thread_pool | thread_pool.h | ✅ Compliant |
| `numa_node_id_` | thread_pool | thread_pool.h | ✅ Compliant |
| `previous_state_` | parallel_guard | ParallelGuard.h | ✅ Compliant |
| `old_id_` | thread_id_guard | Parallel.h | ✅ Compliant |
| `execution_mutex_` | thread_pool (caffe2) | ThreadPool.h | ✅ Compliant |
| `min_work_size_` | thread_pool (caffe2) | ThreadPool.h | ✅ Compliant |
| `default_num_threads_` | thread_pool (caffe2) | ThreadPool.h | ✅ Compliant |

---

## 5. DOWNSTREAM DEPENDENCIES

### 5.1 Files That Include Headers from

The following files contain `#include` directives referencing the  module and will need updates after renaming:

#### Within  Module:
1. **Parallel.h** (lines 157-162)
   ```cpp
   #if XSIGMA_HAS_OPENMP
   #include "ParallelOpenMP.h"
   #elif !XSIGMA_HAS_OPENMP
   #include "ParallelNative.h"
   #endif
   #include "Parallel-inl.h"
   ```
   **Update to:**
   ```cpp
   #if XSIGMA_HAS_OPENMP
   #include "parallel_openmp.h"
   #elif !XSIGMA_HAS_OPENMP
   #include "parallel_native.h"
   #endif
   #include "parallel-inl.h"
   ```

2. **Parallel-inl.h** (line 3)
   ```cpp
   #include "ParallelGuard.h"
   ```
   **Update to:**
   ```cpp
   #include "parallel_guard.h"
   ```

3. **ParallelNative.cpp** (lines 7-10)
   ```cpp
   #include "thread_pool.h"
   #include "Parallel.h"
   #include "thread_pool.h"
   #include "thread_pool.h"
   ```
   **Update to:**
   ```cpp
   #include "pt_thread_pool.h"
   #include "parallel.h"
   #include "parallel_future.h"
   #include "thread_pool.h"
   ```

4. **ParallelOpenMP.cpp** (lines 7-9)
   ```cpp
   #include "Parallel.h"
   #include "thread_pool.h"
   ```
   **Update to:**
   ```cpp
   #include "parallel.h"
   #include "parallel_future.h"
   ```

5. **ParallelCommon.cpp** (lines 6-7)
   ```cpp
   #include "thread_pool.h"
   #include "Parallel.h"
   ```
   **Update to:**
   ```cpp
   #include "pt_thread_pool.h"
   #include "parallel.h"
   ```

6. **ParallelThreadPoolNative.cpp** (lines 6-8)
   ```cpp
   #include "thread_pool.h"
   #include "Parallel.h"
   #include "ThreadLocalState.h"
   ```
   **Update to:**
   ```cpp
   #include "pt_thread_pool.h"
   #include "parallel.h"
   #include "thread_local_state.h"
   ```

7. **ThreadPool.h** (line 11)
   ```cpp
   #include "ThreadPoolCommon.h"
   ```
   **Update to:**
   ```cpp
   #include "thread_pool_common.h"
   ```

8. **thread_pool_guard.cpp**
   - Includes to update (if any)

---

## 6. DETAILED RENAMING PLAN

### Phase 1: Rename Header Files
1. `Parallel.h` → `parallel.h`
2. `Parallel-inl.h` → `parallel-inl.h`
3. `ParallelGuard.h` → `parallel_guard.h`
4. `ParallelNative.h` → `parallel_native.h`
5. `ParallelOpenMP.h` → `parallel_openmp.h`
6. `ThreadLocalState.h` → `thread_local_state.h`
7. `ThreadPool.h` → `thread_pool_legacy.h` (distinguish from xsigma::thread_pool)
8. `ThreadPoolCommon.h` → `thread_pool_common.h`
9. `WorkersPool.h` → `workers_pool.h`

### Phase 2: Rename Implementation Files
1. `ParallelCommon.cpp` → `parallel_common.cpp`
2. `ParallelNative.cpp` → `parallel_native.cpp`
3. `ParallelOpenMP.cpp` → `parallel_openmp.cpp`
4. `ParallelThreadPoolNative.cpp` → `parallel_thread_pool_native.cpp`
5. `ThreadPool.cpp` → `thread_pool_legacy.cpp`

### Phase 3: Update All Include Directives
- Update all `#include` statements in files listed in Section 5.1
- Update any CMakeLists.txt or build configuration files

### Phase 4: Verification
- Run build to ensure no broken includes
- Run all tests to ensure functionality preserved
- Verify no external dependencies broken

---

## 7. RISK ASSESSMENT

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| Build breakage | Low | Systematic include updates |
| Test failures | Very Low | No logic changes |
| External dependencies | Medium | Search entire codebase for includes |
| Merge conflicts | Low | Coordinate with team |

---

## 8. RECOMMENDATIONS

1. **Immediate Actions:**
   - Rename all CamelCase files to snake_case
   - Update all include directives systematically
   - Run full build and test suite

2. **Future Improvements:**
   - Consider consolidating legacy Caffe2 code (ThreadPool.h/cxx, WorkersPool.h)
   - Remove  prefix from include paths once module is stable
   - Add CI check to enforce snake_case file naming

3. **Testing Strategy:**
   - Build on all platforms (Linux, macOS, Windows)
   - Run parallel execution tests
   - Verify thread pool functionality
   - Check NUMA binding on supported platforms

---

## 9. MISSING FILES REFERENCED

The following files are referenced in includes but do not exist. These should be created or references removed:

### 9.1 Files Referenced But Missing

| Referenced File | Referenced In | Line | Action Required |
|-----------------|---------------|------|-----------------|
| `thread_pool.h` | ParallelNative.cpp | 7 | ❌ **File does not exist** - needs creation or removal |
| `thread_pool.h` | ParallelNative.cpp, ParallelOpenMP.cpp | 9, 9 | ❌ **File does not exist** - needs creation or removal |
| `Config.h` | Multiple files (commented out) | Various | ⚠️ Already commented out |
| `FuncTorchTLS.h` | ThreadLocalState.h | 9 | ⚠️ Already commented out |
| `PythonTorchFunctionTLS.h` | ThreadLocalState.h | 10 | ⚠️ Already commented out |
| `SavedTensorHooks.h` | ThreadLocalState.h | 11 | ⚠️ Already commented out |
| `ThreadLocalPythonObjects.h` | ThreadLocalState.h | 12 | ⚠️ Already commented out |
| `record_function.h` | ThreadLocalState.h | 13 | ⚠️ Already commented out |

### 9.2 Recommended Actions

1. **thread_pool.h** - This file is actively included but missing. Options:
   - Create the file with `pt_thread_pool` class definition
   - Remove the include if the class is defined elsewhere
   - The class is referenced in thread_pool.h but definition is missing

2. **thread_pool.h** - Referenced in multiple files. Options:
   - Create the file with future/promise abstractions
   - Remove includes if futures are not used (user already removed `intraop_launch_future`)
   - Verify if async functionality is still needed

---

## 10. COMPLETE FILE INVENTORY

### 10.1 All Files in  Directory

| File Name | Type | Size (lines) | Naming Status | Action |
|-----------|------|--------------|---------------|--------|
| `Parallel.h` | Header | 163 | ❌ CamelCase | Rename to `parallel.h` |
| `Parallel-inl.h` | Inline Header | 104 | ❌ CamelCase | Rename to `parallel-inl.h` |
| `ParallelCommon.cpp` | Implementation | ~100 | ❌ CamelCase | Rename to `parallel_common.cpp` |
| `ParallelGuard.h` | Header | 24 | ❌ CamelCase | Rename to `parallel_guard.h` |
| `ParallelNative.h` | Header | 18 | ❌ CamelCase | Rename to `parallel_native.h` |
| `ParallelNative.cpp` | Implementation | ~250 | ❌ CamelCase | Rename to `parallel_native.cpp` |
| `ParallelOpenMP.h` | Header | 48 | ❌ CamelCase | Rename to `parallel_openmp.h` |
| `ParallelOpenMP.cpp` | Implementation | ~120 | ❌ CamelCase | Rename to `parallel_openmp.cpp` |
| `ParallelThreadPoolNative.cpp` | Implementation | ~100 | ❌ CamelCase | Rename to `parallel_thread_pool_native.cpp` |
| `ThreadLocalState.h` | Header | 132 | ❌ CamelCase | Rename to `thread_local_state.h` |
| `ThreadPool.h` | Header (Legacy) | 82 | ❌ CamelCase | Rename to `thread_pool_legacy.h` |
| `ThreadPool.cpp` | Implementation (Legacy) | ~200 | ❌ CamelCase | Rename to `thread_pool_legacy.cpp` |
| `ThreadPoolCommon.h` | Header | 20 | ❌ CamelCase | Rename to `thread_pool_common.h` |
| `WorkersPool.h` | Header | 419 | ❌ CamelCase | Rename to `workers_pool.h` |
| `thread_pool.h` | Header | 130 | ✅ snake_case | No action |
| `thread_pool.cpp` | Implementation | ~150 | ✅ snake_case | No action |
| `thread_pool_guard.h` | Header | ~20 | ✅ snake_case | No action |
| `thread_pool_guard.cpp` | Implementation | ~15 | ✅ snake_case | No action |

**Total Files:** 18
**Files Requiring Rename:** 14
**Files Already Compliant:** 4

---

## 11. BUILD SYSTEM UPDATES

### 11.1 CMakeLists.txt Updates Required

The following CMakeLists.txt files likely reference these source files and will need updates:

**Search for:**
```bash
find Library/Core -name "CMakeLists.txt" -exec grep -l "Parallel\|ThreadPool\|WorkersPool" {} \;
```

**Expected locations:**
- `Library/Core/CMakeLists.txt`
- `Library/Core//CMakeLists.txt` (if exists)

**Update pattern:**
```cmake
# Before:
set(SOURCES
    ParallelNative.cpp
    ParallelOpenMP.cpp
    ParallelCommon.cpp
    ...
)

# After:
set(SOURCES
    parallel_native.cpp
    parallel_openmp.cpp
    parallel_common.cpp
    ...
)
```

---

## 12. EXTERNAL DEPENDENCIES

### 12.1 Files Outside  That May Include These Headers

Based on codebase search, the following external files may include headers from :

**Potential external includes:**
- Any file including `"/Parallel.h"`
- Any file including `"/ParallelGuard.h"`
- Any file including `"/ThreadLocalState.h"`

**Search command:**
```bash
grep -r "/Parallel" Library/Core --include="*.h" --include="*.cpp" | grep -v "/"
```

**Action:** Run this search before renaming to identify all external dependencies.

---

## 13. IMPLEMENTATION CHECKLIST

### Pre-Rename Checklist
- [ ] Backup current codebase or create feature branch
- [ ] Search for all external includes of  headers
- [ ] Identify all CMakeLists.txt files that reference these sources
- [ ] Document current build configuration
- [ ] Run baseline tests to establish working state

### Rename Execution Checklist
- [ ] Rename all 14 CamelCase files to snake_case
- [ ] Update all internal includes within  module
- [ ] Update all external includes from other modules
- [ ] Update CMakeLists.txt files
- [ ] Update any documentation referencing old file names
- [ ] Update CLASS_HIERARCHY.md with new file names

### Post-Rename Verification Checklist
- [ ] Build succeeds on Linux
- [ ] Build succeeds on macOS
- [ ] Build succeeds on Windows
- [ ] All unit tests pass
- [ ] Thread pool tests pass
- [ ] Parallel execution tests pass
- [ ] NUMA binding tests pass (if applicable)
- [ ] No compiler warnings introduced
- [ ] No linker errors
- [ ] Documentation updated

### Final Steps
- [ ] Code review
- [ ] Update CHANGELOG.md
- [ ] Merge to main branch
- [ ] Tag release (if applicable)

---

**End of Report**

**Report Version:** 1.0
**Generated:** 2025-11-04
**Author:** XSigma Development Team
**Status:** Ready for Implementation
