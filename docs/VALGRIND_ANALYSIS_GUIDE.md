# Valgrind Memory Leak Analysis Guide

**Purpose**: Step-by-step guide for analyzing and fixing memory leaks detected by Valgrind in CI run #18260896868

---

## Step 1: Download Valgrind Artifacts

### From GitHub Actions UI
1. Navigate to: https://github.com/XSigmaAnalyitix/XSigma/actions/runs/18260896868
2. Scroll to "Artifacts" section at bottom of page
3. Download: `valgrind-results` (or similar name)
4. Extract the ZIP file to a local directory

### Expected Files
```
valgrind-results/
├── MemoryChecker.CoreCxxTests.log
├── MemoryChecker.*.log
└── valgrind_summary.txt (if available)
```

---

## Step 2: Understand Valgrind Output Format

### Leak Summary Section
```
==12345== LEAK SUMMARY:
==12345==    definitely lost: 1,024 bytes in 8 blocks
==12345==    indirectly lost: 512 bytes in 4 blocks
==12345==      possibly lost: 256 bytes in 2 blocks
==12345==    still reachable: 2,048 bytes in 16 blocks
==12345==         suppressed: 0 bytes in 0 blocks
```

### Leak Categories

| Category | Severity | Description | Action Required |
|----------|----------|-------------|-----------------|
| **definitely lost** | 🔴 CRITICAL | Memory allocated but pointer lost | **MUST FIX** |
| **indirectly lost** | 🟠 HIGH | Memory reachable only through lost pointers | **MUST FIX** |
| **possibly lost** | 🟡 MEDIUM | Pointer exists but may not be valid | **SHOULD FIX** |
| **still reachable** | 🟢 LOW | Memory still pointed to at exit | **REVIEW** |
| **suppressed** | ⚪ INFO | Intentionally ignored via suppressions | **OK** |

---

## Step 3: Locate Leak Details

### Search for Leak Records
Look for blocks like this in the log:

```
==12345== 1,024 bytes in 8 blocks are definitely lost in loss record 42 of 100
==12345==    at 0x4C2E0EF: operator new(unsigned long) (vg_replace_malloc.c:334)
==12345==    by 0x5A3B2C1: xsigma::core::allocate_buffer(unsigned long) (memory_utils.cxx:45)
==12345==    by 0x5A3C4D2: xsigma::core::buffer_manager::create() (buffer_manager.cxx:78)
==12345==    by 0x5A3D5E3: xsigma::core::initialize() (core_init.cxx:123)
==12345==    by 0x4E8F6A4: main (CoreCxxTests.cxx:15)
```

### Key Information
- **Allocation site**: `memory_utils.cxx:45` - Where memory was allocated
- **Call stack**: Shows the path from `main` to allocation
- **Size**: `1,024 bytes in 8 blocks` - Total leaked memory

---

## Step 4: Common Memory Leak Patterns

### Pattern 1: Missing Delete/Free

**Symptom**:
```cpp
// Bad: Memory allocated but never freed
void process_data() {
    int* data = new int[100];
    // ... use data ...
    // Missing: delete[] data;
}
```

**Fix**:
```cpp
// Good: Use RAII with smart pointers
void process_data() {
    auto data = std::make_unique<int[]>(100);
    // ... use data ...
    // Automatically freed when data goes out of scope
}
```

### Pattern 2: Circular Shared Pointer References

**Symptom**:
```cpp
// Bad: Circular reference prevents cleanup
class Node {
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;  // Creates cycle!
};
```

**Fix**:
```cpp
// Good: Use weak_ptr to break cycle
class Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // Breaks cycle
};
```

### Pattern 3: Missing Destructor Cleanup

**Symptom**:
```cpp
// Bad: Resource allocated in constructor but not freed
class ResourceManager {
    void* resource_;
public:
    ResourceManager() : resource_(allocate_resource()) {}
    // Missing destructor!
};
```

**Fix**:
```cpp
// Good: RAII with proper cleanup
class ResourceManager {
    void* resource_;
public:
    ResourceManager() : resource_(allocate_resource()) {}
    ~ResourceManager() { free_resource(resource_); }
    
    // Rule of Five: Also implement copy/move if needed
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;
};
```

### Pattern 4: Static/Global Object Leaks

**Symptom**:
```cpp
// Bad: Singleton with manual memory management
class Logger {
    static Logger* instance_;
public:
    static Logger* get_instance() {
        if (!instance_) {
            instance_ = new Logger();  // Never deleted!
        }
        return instance_;
    }
};
```

**Fix**:
```cpp
// Good: Use static local variable (Meyer's Singleton)
class Logger {
public:
    static Logger& get_instance() {
        static Logger instance;  // Automatically cleaned up
        return instance;
    }
};
```

### Pattern 5: TBB Task Arena Leaks

**Symptom**:
```cpp
// Bad: TBB arena not properly cleaned up
void parallel_work() {
    tbb::task_arena arena(4);
    arena.execute([&] {
        // ... parallel work ...
    });
    // Arena may not be properly finalized
}
```

**Fix**:
```cpp
// Good: Explicit arena cleanup
void parallel_work() {
    {
        tbb::task_arena arena(4);
        arena.execute([&] {
            // ... parallel work ...
        });
    }  // Arena destroyed here
    tbb::task_scheduler_handle::release();  // If needed
}
```

---

## Step 5: Analyze XSigma-Specific Code

### Priority Files to Review

Based on typical leak patterns, check these files first:

1. **Memory Management**
   - `Library/Core/memory/*.cxx`
   - Look for: Manual allocations, buffer management

2. **TBB Integration** (if enabled)
   - `Library/Core/smp/TBB/*.cxx`
   - Look for: Task arenas, parallel algorithms

3. **Logging System**
   - `Library/Core/logging/*.cxx`
   - Look for: Static loggers, file handles

4. **Utility Classes**
   - `Library/Core/util/*.cxx`
   - Look for: Resource wrappers, containers

5. **Test Code**
   - `Library/Core/Testing/Cxx/Test*.cxx`
   - Look for: Test fixtures, setup/teardown

### Search Commands

```bash
# Find all manual memory allocations
grep -r "new \|malloc\|calloc" Library/Core --include="*.cxx" --include="*.cpp"

# Find all deletions (to match with allocations)
grep -r "delete \|free(" Library/Core --include="*.cxx" --include="*.cpp"

# Find all raw pointers in class members
grep -r "^\s*[a-zA-Z_][a-zA-Z0-9_]*\*\s*[a-zA-Z_]" Library/Core --include="*.h" --include="*.hxx"
```

---

## Step 6: Fix Memory Leaks

### General Strategy

1. **Identify the leak** from Valgrind output
2. **Locate the code** using file:line information
3. **Understand the ownership** - Who should free this memory?
4. **Apply RAII pattern** - Use smart pointers or containers
5. **Test the fix** locally with Valgrind
6. **Verify no new leaks** introduced

### Fixing Workflow

```bash
# 1. Make your fix in the source file
vim Library/Core/memory/buffer_manager.cxx

# 2. Rebuild with debug symbols
cd Scripts
python setup.py ninja.clang.debug.config.build

# 3. Run Valgrind on specific test
cd ../build_ninja_python
valgrind --leak-check=full --show-leak-kinds=all ./bin/CoreCxxTests --gtest_filter=BufferManagerTest.*

# 4. Check for leaks in output
# Look for "All heap blocks were freed -- no leaks are possible"
```

---

## Step 7: Verify Fixes

### Local Verification

```bash
cd Scripts

# Run full Valgrind test suite
python setup.py ninja.clang.debug.valgrind.config.build.test

# Check for success message
# Expected: "SUCCESS: No memory leaks or errors detected"
```

### CI Verification

1. Commit and push fixes
2. Wait for CI to run Valgrind job
3. Check "Valgrind Memory Check" job status
4. Download new artifacts if issues remain

---

## Step 8: Common XSigma Patterns

### Smart Pointer Usage

```cpp
// Prefer unique_ptr for exclusive ownership
std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(size);

// Use shared_ptr only when multiple owners needed
std::shared_ptr<Resource> resource = std::make_shared<Resource>();

// Use weak_ptr to break cycles
std::weak_ptr<Node> parent = node->get_parent();
```

### Container Usage

```cpp
// Prefer containers over raw arrays
std::vector<int> data(100);  // Automatically managed

// Use std::array for fixed-size arrays
std::array<int, 100> fixed_data;  // Stack-allocated, no leaks
```

### Resource Management

```cpp
// Use RAII wrappers for C resources
class FileHandle {
    FILE* file_;
public:
    explicit FileHandle(const char* path) : file_(fopen(path, "r")) {}
    ~FileHandle() { if (file_) fclose(file_); }
    
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};
```

---

## Step 9: Suppression File (Last Resort)

If a leak is in third-party code or intentional, add to suppression file:

**File**: `Scripts/sanitizer_ignore.txt`

```
# Intentional leak in test code
src:*/Library/Core/Testing/Cxx/TestIntentionalLeak.cxx

# Known issue in third-party library
fun:third_party_function_with_leak
```

**⚠️ WARNING**: Only use suppressions for:
- Third-party library issues (already suppressed)
- Intentional test leaks
- False positives (rare)

**DO NOT** suppress real leaks in XSigma code!

---

## Step 10: Documentation

After fixing leaks, document:

1. **What was leaking**: Brief description
2. **Root cause**: Why the leak occurred
3. **Fix applied**: What changed
4. **Verification**: How you confirmed the fix

Example commit message:
```
Fix memory leak in buffer_manager

- Issue: Buffer allocated in create() but never freed
- Cause: Missing destructor in BufferManager class
- Fix: Added destructor with proper cleanup
- Verified: Valgrind reports no leaks in BufferManagerTest
```

---

## Quick Reference

### Valgrind Command
```bash
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./bin/CoreCxxTests
```

### Expected Success Output
```
==12345== LEAK SUMMARY:
==12345==    definitely lost: 0 bytes in 0 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
==12345==      possibly lost: 0 bytes in 0 blocks
==12345==    still reachable: 0 bytes in 0 blocks
==12345==         suppressed: 0 bytes in 0 blocks
```

### Key Files
- Valgrind logs: `valgrind-results/*.log`
- Suppression file: `Scripts/sanitizer_ignore.txt`
- Test command: `python setup.py ninja.clang.debug.valgrind.config.build.test`

---

**End of Valgrind Analysis Guide**

