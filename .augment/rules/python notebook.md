---
type: "always_apply"
description: "python"
---

# ✅ Naming Convention: `xsigma_cpp_to_python_mapping`

**Description**  
Defines how `xsigma` C++ objects are mapped to Python objects in Augment.  
Field names are adapted to Python conventions, while function names remain unchanged.

---

## 🔁 Mapping Rules

| Element Type | C++ Convention | Python Convention   | Notes                           |
|--------------|----------------|---------------------|---------------------------------|
| **Field**    | `snake_case`   | `camelCase`         | Automatically transformed       |
| **Function** | `originalName` | `originalName`      | No change in naming             |

---

## 🔧 Example Mappings

### 🔹 Fields

| C++ Field Name    | Python Field Name |
|------------------|-------------------|
| `calendar_id`     | `calendarId`      |
| `notional_amount` | `notionalAmount`  |
| `swap_start_date` | `swapStartDate`   |

### 🔹 Functions

| C++ Function Name  | Python Function Name |
|--------------------|----------------------|
| `build_swap`       | `build_swap`         |
| `with_calendar_id` | `with_calendar_id`   |

---

## 📘 Rule Definition (YAML-style)

```yaml
rule: xsigma_cpp_to_python_mapping
description: >
  Enforces naming convention when mapping xsigma C++ objects to Python objects.
  Field names convert from snake_case to camelCase. Function names remain unchanged.

targets:
  - xsigma C++ classes and methods

checks:
  - field_naming_conversion:
      from: snake_case
      to: camelCase
      comment: "C++ fields must convert to camelCase in Python."

  - function_naming_consistency:
      must_match_cpp: true
      comment: "Function names must remain unchanged between C++ and Python."
```