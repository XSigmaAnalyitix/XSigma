---
type: "always_apply"
---

# Augmented AI Code Generation Rules

## 1. Error Handling

### No Exception-Based Error Handling
- **Do not use `try/catch` blocks** or exception mechanisms
- Error handling must follow the project's established patterns
- Use return values and conditional checks for error states
- Handle errors gracefully through predictable control flow

❌ **Incorrect**
```cpp
try {
    risky_function();
} catch (...) {
    // handle error
}
```

✅ **Correct**
```cpp
if (!safe_function()) {
    // handle error gracefully without exceptions
}
```

---

## 2. Naming Conventions

### C++ Naming Rules
- **Classes**: `snake_case`
- **Functions**: `snake_case`
- **Class member variables**: `snake_case_` (with trailing underscore)
- **Constants**: `kConstantName` or `CONSTANT_NAME` (follow project standard)

❌ **Incorrect**
```cpp
class MyClass {
    int myVar;
    void DoSomething();
};
```

✅ **Correct**
```cpp
class my_class {
    int value_;
    void do_something();
};
```

---

## 3. Code Formatting and Style

### Formatting Standards
- Follow `.clang-tidy` and `.clang-format` rules strictly
- Adhere to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use consistent spacing, indentation, and brace placement
- Format all code automatically before commit

❌ **Incorrect**
```cpp
if (x==y){foo();}else{bar();}
```

✅ **Correct**
```cpp
if (x == y) {
  foo();
} else {
  bar();
}
```

---

## 4. Include Rules

### Include Path Convention
- Includes must **start from the subfolder** within the project's root
- Use relative paths from the project's root, not absolute paths
- Do not include the full path with the root directory prefix

**Example:**
File path: `Core/xxx/yyy/.../a.h`

❌ **Incorrect**
```cpp
#include "Core/xxx/yyy/a.h"
```

✅ **Correct**
```cpp
#include "xxx/yyy/a.h"
```

---

## 5. Testing Requirements

### Coverage & Reproducibility
- Ensure **minimum 98% test coverage**
- Tests must be reproducible, independent, and follow project testing standards
- Each test should run in isolation without relying on shared state or external test order
- Use fixtures or test factories for consistent test data setup

### Test File Naming & Organization
- Follow naming convention: `Test[ClassName]` in CamelCase
  - Example: `TestMyClass.cpp`, `TestPaymentProcessing.cpp`
- Match test file location to source file structure
- Group related tests using describe blocks or test fixtures by functionality

### Test Performance & Granularity
- **Testing must be fast**: Optimize for quick feedback loops
- **Every function must be called at least once** during testing
- **Use small, focused tests** for specific behaviors rather than monolithic test suites
- Each test should verify one behavior or assertion
- Mock or stub all external dependencies

### Test Scope
- Test happy paths and success cases
- Explicitly test boundary conditions and edge cases
- Test error handling and failure scenarios (return value validation)
- Test null pointers, empty collections, and invalid inputs
- Verify state changes and side effects

✅ **Correct Example**
```cpp
TEST(my_class_test, handles_valid_input) {
  my_class obj;
  EXPECT_TRUE(obj.do_something());
}

TEST(my_class_test, handles_invalid_input) {
  my_class obj;
  EXPECT_FALSE(obj.do_something_with(-1));
}

TEST(my_class_test, handles_null_pointer) {
  my_class obj;
  EXPECT_FALSE(obj.process(nullptr));
}
```

---

## 6. DLL Export and Visibility

### Macro Usage
- **`XSIGMA_API`** → For functions with definitions in `.cxx` files
- **`XSIGMA_VISIBILITY`** → For class declarations
- Functions with implementations in `.cxx` must use `XSIGMA_API`
- Class declarations must use `XSIGMA_VISIBILITY`
- Apply macros consistently across all public APIs

❌ **Incorrect**
```cpp
class my_class {
public:
    void do_something();
};
```

✅ **Correct**
```cpp
class XSIGMA_VISIBILITY my_class {
public:
    XSIGMA_API void do_something();
};
```

---

## 7. Architecture & Design

### Modularity
- Separate concerns: presentation, business logic, data access
- Use dependency injection patterns for testability
- Create clear interfaces/contracts between modules
- Avoid tight coupling; enable easy substitution of components

### Design Principles
- Apply SOLID principles where practical
- Favor immutability and pure functions
- Use C++ RAII (Resource Acquisition Is Initialization) pattern
- Keep state management minimal and predictable

---

## 8. General Principles

### Readability & Maintainability
- **Prioritize readability and maintainability** above all else
- Code should be **clear, consistent, and well-documented**
- Follow established project practices for commits, branching, and reviews
- Shorter, cleaner, and more direct code is easier to maintain

### Documentation
- Explain *why*, not *what* the code does
- Document complex algorithms or non-obvious logic
- Add comments for workarounds or temporary solutions
- Document function behavior, parameters, and return values
- Keep comments up-to-date with code changes

### Code Review Readiness
- Ensure generated code is production-ready
- Follow all project conventions before submission
- Run `clang-format` and `clang-tidy` before commit
- Verify test completeness (98% coverage minimum)

---

## 9. Common Pitfalls to Avoid

- Using exceptions or `try/catch` blocks
- Ignoring include path rules or using absolute paths
- Mixing naming conventions within the same file
- Omitting `XSIGMA_API` or `XSIGMA_VISIBILITY` macros
- Generating untested code or inadequate test coverage
- Creating monolithic functions or classes
- Hardcoding values that should be configurable
- Generating tests that don't verify behavior properly
- Over-engineering simple solutions
- Ignoring `clang-format` and `clang-tidy` rules