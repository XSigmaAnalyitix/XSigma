---
type: "always_apply"
---

## 1. Error Handling
- **Do not use `try/catch` blocks**.
- Error handling must follow the project's established patterns without exceptions.

❌ **Incorrect**
```cpp
try {
    risky_function();
} catch (...) {
    // handle error
}
````

✅ **Correct**

```cpp
if (!safe_function()) {
    // handle error gracefully without exceptions
}
```

---

## 2. Naming Conventions

* **Classes**: `snake_case`
* **Functions**: `snake_case`
* **Class member variables**: `snake_case_` (with trailing underscore)

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

* Follow `.clang-tidy` and `.clang-format` rules.
* Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

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

* Includes must **start from the subfolder** within the project’s root.

Example:
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

* Ensure **minimum 98% test coverage**.
* Tests must be reproducible, independent, and follow project testing standards.

✅ **Correct**

```cpp
TEST(my_class_test, handles_valid_input) {
  my_class obj;
  EXPECT_TRUE(obj.do_something());
}

TEST(my_class_test, handles_invalid_input) {
  my_class obj;
  EXPECT_FALSE(obj.do_something_with(-1));
}
```

---

## 6. DLL Export and Visibility

* Use the following macros consistently:

  * `XSIGMA_API` → For functions with definitions in `.cxx` files.
  * `XSIGMA_VISIBILITY` → For class declarations.
  * For functions that have a body in `.cxx`, use **`XSIGMA_API`**.
  * For class declarations, use **`XSIGMA_VISIBILITY`**.

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

## 7. General Principles

* Prioritize **readability and maintainability**.
* Code should be **clear, consistent, and well-documented**.
* Follow established project practices for commits, branching, and reviews.
* Shorter, cleaner, and more direct code is easier to maintain.

