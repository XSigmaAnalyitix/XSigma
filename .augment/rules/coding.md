---
## type: "always_apply"

# Augmented AI Code Generation Rules (C++20)

These rules define the standards and constraints for AI-assisted and human-authored C++20 code contributions in the project. They ensure generated code adheres to established engineering, architectural, and stylistic conventions while maintaining maintainability, reliability, and testability.

---

## 1. Error Handling

### 1.1 General Principles

* **No exception-based error handling.** Do not use `try/catch` or exception mechanisms under any circumstances.
* Use **return values** and **conditional control flow** to manage error states.
* Ensure all errors are **predictable, traceable, and testable**.
* Propagate errors through clearly defined return codes or result objects.

### 1.2 Examples

❌ **Incorrect:**

```cpp
try {
  risky_function();
} catch (...) {
  // handle error
}
```

✅ **Correct:**

```cpp
if (!safe_function()) {
  // handle error gracefully without exceptions
}
```

---

## 2. Naming Conventions

### 2.1 C++ Naming Rules

| Element Type    | Convention                          | Example               |
| --------------- | ----------------------------------- | --------------------- |
| Class           | `snake_case`                        | `class my_class`      |
| Function        | `snake_case`                        | `void do_something()` |
| Member Variable | `snake_case_` (trailing underscore) | `int count_`          |
| Constant        | `kConstantName` or `CONSTANT_NAME`  | `const int kMaxCount` |

Maintain uniform naming across the entire file. Mixing conventions is not allowed.

### 2.2 Examples

❌ **Incorrect:**

```cpp
class MyClass {
  int myVar;
  void DoSomething();
};
```

✅ **Correct:**

```cpp
class my_class {
  int value_;
  void do_something();
};
```

---

## 3. Code Formatting and Style

### 3.1 Formatting Standards

* Follow **`.clang-format`** and **`.clang-tidy`** rules strictly.
* Conform to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) unless project-specific overrides exist.
* Maintain consistent indentation, brace placement, and spacing.
* Format code automatically before commit.

✅ **Example:**

```cpp
if (x == y) {
  foo();
} else {
  bar();
}
```

---

## 4. Include Rules

### 4.1 Include Path Convention

* Include paths must **start from the project subfolder**, not the root.
* Do **not** use absolute paths.
* Maintain consistent include structure across all files.

**Example:**

File path: `Core/xxx/yyy/a.h`

❌ **Incorrect:**

```cpp
#include "Core/xxx/yyy/a.h"
```

✅ **Correct:**

```cpp
#include "xxx/yyy/a.h"
```

---

## 5. Testing Requirements

### 5.1 Coverage & Quality

* Minimum **98% code coverage** required for all generated code.
* Tests must be deterministic, reproducible, and isolated.
* Avoid shared state between tests.
* Prefer test fixtures or factory patterns for setup.

### 5.2 File Naming & Structure

* Test files must mirror source file hierarchy.
* Use naming pattern `Test[ClassName].cpp` (CamelCase).

  * Example: `TestMyClass.cpp`, `TestPaymentProcessor.cpp`

### 5.3 Test Practices

* Each test must validate a **single behavior or condition**.
* Mock or stub all external dependencies.
* Verify both **happy path** and **failure** conditions.

✅ **Examples:**

```cpp
TEST(MyClassTest, HandlesValidInput) {
  my_class obj;
  EXPECT_TRUE(obj.do_something());
}

TEST(MyClassTest, HandlesInvalidInput) {
  my_class obj;
  EXPECT_FALSE(obj.do_something_with(-1));
}

TEST(MyClassTest, HandlesNullPointer) {
  my_class obj;
  EXPECT_FALSE(obj.process(nullptr));
}
```

---

## 6. DLL Export and Visibility

### 6.1 Macro Rules

| Context                          | Macro               | Description                                |
| -------------------------------- | ------------------- | ------------------------------------------ |
| Function (implemented in `.cxx`) | `XSIGMA_API`        | Mark all externally visible functions      |
| Class declaration                | `XSIGMA_VISIBILITY` | Required for all public class declarations |

Apply macros consistently across public headers.

✅ **Correct Example:**

```cpp
class XSIGMA_VISIBILITY my_class {
 public:
  XSIGMA_API void do_something();
};
```

---

## 7. Architecture & Design Principles

### 7.1 Modularity

* Separate presentation, business logic, and data access layers.
* Use **dependency injection** for testability.
* Define clear, minimal interfaces between modules.
* Favor loose coupling and composability.

### 7.2 Design Principles

* Follow **SOLID** design principles.
* Favor **immutability** and **pure functions**.
* Use **RAII (Resource Acquisition Is Initialization)** for resource safety.
* Avoid global mutable state.
* When declaring unused parameters, prefix them with `XSIGMA_UNUSED`.

---

## 8. Memory Management

### 8.1 Ownership & Lifetime

* Use **smart pointers** (`std::unique_ptr`, `std::shared_ptr`) instead of raw pointers.
* Avoid `new`/`delete` in code directly; encapsulate allocations.
* Apply RAII for all resource management.

### 8.2 Guidelines

* `std::unique_ptr` → exclusive ownership.
* `std::shared_ptr` → shared ownership (minimize use).
* `std::weak_ptr` → observe shared resources safely.
* Avoid circular references.

---

## 9. Concurrency & Thread Safety

### 9.1 General Rules

* Prefer **`std::thread`, `std::async`, `std::future`, `std::mutex`**, and related utilities.
* Protect shared resources using scoped locks (`std::scoped_lock`).
* Never expose mutable shared state without synchronization.
* Avoid manual thread lifecycle management when possible.

### 9.2 Modern C++20 Practices

* Use **atomic types** for shared counters and flags.
* Prefer **structured concurrency** (RAII-thread ownership).
* Avoid busy-wait loops; use condition variables.

---

## 10. CI/CD and Static Analysis

### 10.1 Automated Validation

* Every code generation pipeline must run:

  * `clang-tidy`
  * `clang-format`
  * Unit and integration test suites
  * Code coverage analysis

### 10.2 Build System Integration

* Use **CMake** targets for all builds and tests.
* Define explicit dependencies to ensure reproducibility.
* CI must fail on any formatting, linting, or coverage violation.

---

## 11. AI Code Review Guidelines

### 11.1 AI-Generated Code Validation

* All AI-generated code must undergo human review before merge.
* Reviewers must check adherence to these standards.
* AI suggestions should be deterministic, documented, and traceable.

### 11.2 Prompt Hygiene

* Avoid ambiguous or overly general prompts.
* Include context: module purpose, dependencies, and naming rules.
* Review generated code for compliance before committing.

---

## 12. Security and Robustness

### 12.1 Secure Coding Practices

* Validate all input; never assume trust.
* Avoid unsafe functions (`strcpy`, `sprintf`, etc.).
* Use safe standard library alternatives.
* Sanitize external data before use.
* Implement boundary checks on all buffers and arrays.

---

## 13. Documentation & Maintainability

### 13.1 Documentation Standards

* Document *intent and rationale*, not obvious implementation.
* Annotate non-trivial algorithms and corner cases.
* Keep all comments synchronized with code changes.

### 13.2 Review Readiness

* Code must be production-ready and fully tested.
* All style, build, and test checks must pass before submission.

---

## 14. Common Pitfalls to Avoid

* Using exceptions or `try/catch`
* Violating include path rules
* Mixing naming conventions
* Omitting required macros (`XSIGMA_API`, `XSIGMA_VISIBILITY`)
* Submitting untested or low-coverage code
* Hardcoding values instead of using configuration
* Creating oversized classes or functions
* Failing to document reasoning behind design choices
* Ignoring `clang-tidy` and `clang-format`

---

**End of Document**
