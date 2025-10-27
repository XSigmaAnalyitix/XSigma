# XSigma Linter Configuration - Quick Reference Guide

**Organization:** XSigmaAnalyitix  
**Project:** XSigma  
**Last Updated:** 2025-10-27

---

## Overview

The XSigma linter system now uses a centralized YAML configuration file to manage paths and settings. This guide helps developers understand and use the new configuration system.

---

## Configuration File Location

```
Tools/linter/config/xsigma_linter_config.yaml
```

---

## Using the Configuration Loader

### Basic Usage

```python
from Tools.linter.config import load_config, get_repo_root

# Load entire configuration
config = load_config()

# Get repository root
repo_root = get_repo_root()
```

### Specific Configuration Values

```python
from Tools.linter.config import (
    get_header_only_apis_file,
    get_header_only_test_globs,
    get_graph_break_registry_path,
    get_ordered_set_import,
    get_import_allowlist,
)

# Get header-only APIs file path
apis_file = get_header_only_apis_file()

# Get test glob patterns
test_globs = get_header_only_test_globs()

# Get graph break registry path
registry_path = get_graph_break_registry_path()

# Get OrderedSet import statement
import_stmt = get_ordered_set_import()

# Get allowed imports list
allowed_imports = get_import_allowlist()
```

### Path Resolution

```python
from Tools.linter.config import resolve_path

# Resolve relative path to absolute
abs_path = resolve_path("Library/Core/header_only_apis.txt")

# Resolve absolute path (returns as-is)
abs_path = resolve_path("/absolute/path/to/file")
```

---

## Configuration Structure

### Header-Only APIs

```yaml
header_only:
  apis_file: "Library/Core/header_only_apis.txt"
  test_globs:
    - "Library/Core/test/cpp/aoti_abi_check/*.cpp"
    - "Tests/**/header_only_test.cpp"
```

### Dynamo Registry

```yaml
dynamo:
  graph_break_registry: "Library/Core/_dynamo/graph_break_registry.json"
  default_dynamo_dir: "Library/Core/_dynamo"
```

### OrderedSet Import

```yaml
ordered_set:
  import_statement: "from xsigma.utils._ordered_set import OrderedSet\n\n"
  error_message: "Builtin `set` is deprecated"
```

### PyBind11 Includes

```yaml
pybind11:
  utils_header: "Library/Core/csrc/utils/pybind.h"
  specialization_header: "Library/Core/csrc/utils/pybind.h"
```

### Meta Registration

```yaml
meta:
  fake_impls: "Library/Core/_subclasses/fake_impls.py"
  meta_registrations: "Library/Core/_meta_registrations.py"
```

### Import Allowlist

```yaml
import_allowlist:
  - "sympy"
  - "einops"
  - "torch"
  - "numpy"
  # ... more modules
```

---

## Adding New Configuration Values

### Step 1: Add to YAML File

Edit `Tools/linter/config/xsigma_linter_config.yaml`:

```yaml
my_feature:
  setting_name: "value"
  another_setting: "another_value"
```

### Step 2: Add Getter Function

Edit `Tools/linter/config/config_loader.py`:

```python
def get_my_feature_config() -> dict[str, Any]:
    """Get my feature configuration."""
    config = load_config()
    return config.get("my_feature", {})

def get_my_setting() -> str:
    """Get my setting value."""
    config = get_my_feature_config()
    return config.get("setting_name", "default_value")
```

### Step 3: Export from Package

Edit `Tools/linter/config/__init__.py`:

```python
from .config_loader import (
    # ... existing imports
    get_my_feature_config,
    get_my_setting,
)

__all__ = [
    # ... existing exports
    "get_my_feature_config",
    "get_my_setting",
]
```

### Step 4: Use in Linter

```python
from Tools.linter.config import get_my_setting

my_value = get_my_setting()
```

---

## Updating Configuration Values

### For Developers

1. Edit `Tools/linter/config/xsigma_linter_config.yaml`
2. Update the relevant section
3. No code changes needed if using existing getter functions
4. Test with: `python3 Tools/linter/config/config_loader.py`

### For Linter Maintainers

If adding a new linter that needs configuration:

1. Add configuration section to YAML file
2. Add getter functions to `config_loader.py`
3. Export from `__init__.py`
4. Use in linter with try/except fallback:

```python
try:
    from config_loader import get_my_config
    my_config = get_my_config()
except Exception:
    my_config = DEFAULT_VALUE  # Fallback
```

---

## Testing Configuration

### Verify Configuration Loads

```bash
python3 Tools/linter/config/config_loader.py
```

Expected output:
```
✓ Configuration loaded successfully
✓ Repo root: /path/to/XSigma
✓ Header-only APIs file: /path/to/XSigma/Library/Core/header_only_apis.txt
✓ Graph break registry: /path/to/XSigma/Library/Core/_dynamo/graph_break_registry.json
✓ OrderedSet import: 'from xsigma.utils._ordered_set import OrderedSet\n\n'
```

### Verify YAML Syntax

```bash
python3 -c "import yaml; yaml.safe_load(open('Tools/linter/config/xsigma_linter_config.yaml')); print('✓ Valid YAML')"
```

### Test in Python

```python
import sys
sys.path.insert(0, 'Tools/linter/config')
from config_loader import load_config
config = load_config()
print(config)
```

---

## Troubleshooting

### Configuration File Not Found

**Error:** `FileNotFoundError: Configuration file not found`

**Solution:** Ensure you're running from the repository root directory.

### PyYAML Not Installed

**Error:** `ImportError: PyYAML is required`

**Solution:** Install PyYAML:
```bash
pip install pyyaml
```

### Import Errors

**Error:** `Cannot find implementation or library stub for module named "config_loader"`

**Solution:** This is a mypy warning and can be ignored. The module loads correctly at runtime.

---

## Cross-Platform Notes

- All paths use `pathlib.Path` for cross-platform compatibility
- Relative paths are resolved from repository root
- Works on Windows, macOS, and Linux
- No OS-specific path separators or assumptions

---

## Related Documentation

- [XSigma Linter Refactoring Complete](XSIGMA_LINTER_REFACTORING_COMPLETE.md)
- [Linter Documentation](linter.md)
- [Linter Configuration Summary](LINTER_CONFIGURATION_SUMMARY.md)

---

## Support

For questions or issues with the linter configuration:

1. Check this quick reference guide
2. Review the YAML configuration file comments
3. Check the config_loader.py docstrings
4. Refer to the complete refactoring report

