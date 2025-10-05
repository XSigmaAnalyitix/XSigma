# Sanitizer Exclusions

**Last Updated**: 2025-10-05  
**File**: `Scripts/sanitizer_ignore.txt`

---

## Overview

This document explains why specific functions, files, or patterns are excluded from sanitizer checks in the XSigma project. Exclusions should be used sparingly and only when necessary.

---

## Current Exclusions

### 1. Third-Party Libraries

**Pattern**: `src:*/ThirdParty/*`

**Reason**: 
- Third-party code is maintained externally
- We don't control the code quality
- Sanitizer warnings in third-party code are not actionable
- Reduces noise in sanitizer reports

**Sanitizers**: All (ASan, UBSan, TSan, LSan, MSan)

---

### 2. Test Files

**Pattern**: `src:*/test/*`, `src:*/Testing/*`

**Reason**:
- Test files may intentionally trigger edge cases
- Some tests verify error handling that involves undefined behavior
- Tests may use mocking frameworks that trigger sanitizer warnings

**Sanitizers**: All

**Note**: This is a broad exclusion. Consider narrowing to specific test files if possible.

---

### 3. Standard Library Exceptions

**Patterns**:
- `fun:std::__throw_*`
- `fun:std::_Throw_*`
- `type:std::basic_string*`
- `type:std::vector*`
- `type:std::shared_ptr*`
- `type:std::condition_variable*`

**Reason**:
- Standard library implementations may use internal optimizations
- Exception handling mechanisms trigger sanitizer warnings
- These are well-tested by compiler vendors
- False positives from STL internals

**Sanitizers**: All

---

### 4. RenderRegion Function (UBSan)

**Pattern**: `fun:*RenderRegion*`

**File**: `Library/Core/memory/cpu/allocator_bfc.cxx` (lines 1337-1362)

**Reason**:
- Performs pointer arithmetic for memory visualization
- Calculates array indices from pointer differences
- Used in `allocator_bfc::RenderOccupancy()` for ASCII-art memory representation
- The arithmetic is safe but may trigger UBSan warnings for:
  - Unsigned integer overflow in calculations
  - Pointer arithmetic edge cases
  - Array bounds calculations

**Code Context**:
```cpp
void RenderRegion(
    char*        rendered,
    const size_t resolution,
    const size_t total_render_size,
    const size_t offset,
    const void*  base_ptr,
    const void*  ptr,
    const size_t size,
    const char   c)
{
    const char* base_ptr_c = static_cast<const char*>(base_ptr);
    const char* ptr_c      = static_cast<const char*>(ptr);

    // These calculations may trigger UBSan warnings
    size_t start_location = ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
    size_t end_location = ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) / total_render_size;
    
    for (size_t i = start_location; i <= end_location; ++i)
    {
        rendered[i] = c;
    }
}
```

**Sanitizers**: UndefinedBehaviorSanitizer (UBSan) only

**Added**: 2025-10-05

**Alternative Considered**: Adding `__attribute__((no_sanitize("undefined")))` to the function, but using the ignore list is more maintainable.

---

## Guidelines for Adding Exclusions

### When to Add Exclusions

✅ **Valid Reasons**:
1. Third-party code you don't control
2. Intentional undefined behavior for testing
3. Performance-critical code with verified safety
4. False positives from sanitizer limitations
5. Platform-specific code with known quirks

❌ **Invalid Reasons**:
1. "It's too hard to fix" - Fix the code instead
2. "Tests are failing" - Investigate the root cause first
3. "It's legacy code" - Legacy code should be fixed, not ignored
4. "I don't understand the warning" - Ask for help first

### How to Add Exclusions

1. **Identify the pattern**:
   - Function: `fun:function_name` or `fun:*pattern*`
   - Source file: `src:*/path/to/file.cpp`
   - Type: `type:ClassName` or `type:std::*`

2. **Add to `Scripts/sanitizer_ignore.txt`**:
   ```
   # Clear comment explaining why
   fun:*function_name*
   ```

3. **Document in this file**:
   - Add entry with reason, context, and alternatives considered

4. **Review periodically**:
   - Exclusions should be temporary when possible
   - Re-evaluate when upgrading compilers or sanitizers

### Pattern Syntax

- `*` - Wildcard matching any characters
- `fun:name` - Exact function name match
- `fun:*pattern*` - Function name contains pattern
- `src:*/path/*` - Source file path contains pattern
- `type:ClassName` - Type name match

---

## Testing Exclusions

### Verify Exclusion Works

```bash
# Build with sanitizer
cd Scripts
python setup.py ninja.clang.debug.undefined.config.build

# Run tests - should not report warnings for excluded functions
cd ../build_ninja_python
./bin/CoreCxxTests
```

### Check Sanitizer Reports

```bash
# Run with verbose output
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0 ./bin/CoreCxxTests

# Check if excluded function appears in output
# It should NOT appear if exclusion is working
```

---

## Maintenance

### Review Schedule

- **Quarterly**: Review all exclusions for necessity
- **On compiler upgrade**: Re-test exclusions with new compiler
- **On sanitizer upgrade**: Verify exclusions still needed

### Removal Criteria

Remove an exclusion when:
1. ✅ The underlying issue is fixed
2. ✅ The code is refactored to avoid the warning
3. ✅ The sanitizer no longer reports the issue
4. ✅ The excluded code is removed from the codebase

---

## Statistics

### Current Exclusion Count

| Category | Count | Sanitizers |
|----------|-------|------------|
| Third-party libraries | 1 pattern | All |
| Test files | 2 patterns | All |
| Standard library | 6 patterns | All |
| Project-specific | 1 pattern | UBSan |
| **Total** | **10** | - |

### Exclusion by Sanitizer

| Sanitizer | Exclusions |
|-----------|------------|
| All | 9 |
| UBSan only | 1 |
| ASan only | 0 |
| TSan only | 0 |
| LSan only | 0 |
| MSan only | 0 |

---

## Related Documentation

- **Sanitizer Platform Support**: `docs/SANITIZER_PLATFORM_SUPPORT.md`
- **CI Fixes**: `docs/CI_FIXES_IMPLEMENTATION_SUMMARY.md`
- **Sanitizer Ignore File**: `Scripts/sanitizer_ignore.txt`

---

## References

- [Sanitizer Special Case List Format](https://clang.llvm.org/docs/SanitizerSpecialCaseList.html)
- [UBSan Documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [ASan Documentation](https://clang.llvm.org/docs/AddressSanitizer.html)
- [TSan Documentation](https://clang.llvm.org/docs/ThreadSanitizer.html)

---

## Summary

- ✅ **10 total exclusions** in sanitizer ignore list
- ✅ **1 project-specific exclusion** (RenderRegion for UBSan)
- ✅ **9 general exclusions** (third-party, tests, STL)
- 💡 **Use exclusions sparingly** - fix code when possible
- 📅 **Review quarterly** - remove unnecessary exclusions

---

**Last Review**: 2025-10-05  
**Next Review**: 2026-01-05

