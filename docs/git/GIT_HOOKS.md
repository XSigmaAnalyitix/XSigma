# XSigma Git Hooks Documentation

## Overview

The XSigma project uses automated git hooks to maintain code quality and consistency. These hooks run automatically during the git commit process to:

1. **Format C++ code** using clang-format
2. **Validate commit messages** for quality and spelling

All hooks are **cross-platform compatible** and work on Windows, Linux, and macOS.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Hooks Overview](#hooks-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Developer Workflow](#developer-workflow)
- [Bypassing Hooks](#bypassing-hooks)

---

## Quick Start

### Install Hooks

```bash
# From the Scripts directory
cd Scripts
python setup_git_hooks.py --install

# Or from repository root
python Scripts/setup_git_hooks.py --install
```

### Check Status

```bash
python Scripts/setup_git_hooks.py --status
```

### Uninstall Hooks

```bash
python Scripts/setup_git_hooks.py --uninstall
```

---

## Hooks Overview

### 1. Pre-Commit Hook

**Location:** `.git/hooks/pre-commit`

**Purpose:** Automatically format staged C++ files before commit

**What it does:**
- Detects all staged C++ files (`.h`, `.hpp`, `.hxx`, `.cxx`, `.cpp`, `.cc`)
- Runs `clang-format` on each file using the project's `.clang-format` configuration
- Automatically re-stages formatted files
- Skips deleted files
- Provides clear console output showing which files were formatted

**Example output:**
```
ℹ Running XSigma pre-commit hook...

✓ Found clang-format: clang-format
ℹ Found 3 staged file(s)

ℹ Formatting 2 C++ file(s)...
ℹ Formatting: Core/common/types.h
ℹ Formatting: Core/math/vector.cxx

✓ Formatted 2 file(s):
  - Core/common/types.h
  - Core/math/vector.cxx

✓ Pre-commit hook completed successfully!
ℹ Proceeding with commit...
```

### 2. Commit-Msg Hook

**Location:** `.git/hooks/commit-msg`

**Purpose:** Validate commit messages for quality and spelling

**What it validates:**
- **Minimum length:** At least 10 characters (excluding comments)
- **Not empty:** Message must contain actual content
- **Not placeholder:** Rejects messages like "wip", "test", "asdf", etc.
- **Spelling:** Checks for common spelling mistakes
- **Best practices:** Warns about long first lines or lowercase starts

**Example output (validation failed):**
```
✗ ERROR: Commit message validation failed:

  • Commit message is too short (5 characters). Minimum 10 characters required.
  • Spelling mistakes found: 'teh' -> 'the', 'recieve' -> 'receive'

ℹ Please fix the issues above and try again.
ℹ To bypass this check (not recommended), use: git commit --no-verify
```

**Example output (validation passed with warnings):**
```
⚠ WARNING: First line is 85 characters long. Consider keeping it under 72 characters for better readability.
⚠ WARNING: Commit message starts with lowercase letter. Consider starting with an uppercase letter.

ℹ ✓ Commit message validation passed
```

---

## Installation

### Prerequisites

1. **Python 3.6+** (already required by XSigma build system)
2. **Git** (obviously!)
3. **clang-format** (must be in PATH)

### Installing clang-format

#### Windows
```powershell
# Using Chocolatey
choco install llvm

# Or download from https://llvm.org/builds/
```

#### macOS
```bash
brew install clang-format
```

#### Linux
```bash
# Debian/Ubuntu
sudo apt-get install clang-format

# RHEL/CentOS/Fedora
sudo yum install clang-tools-extra

# Or specific version
sudo apt-get install clang-format-18
```

### Installing Hooks

Run the setup script from the `Scripts` directory:

```bash
cd Scripts
python setup_git_hooks.py --install
```

The script will:
1. Check if you're in a git repository
2. Locate the `.git/hooks` directory
3. Install the hooks
4. Make them executable (on Unix-like systems)
5. Backup any existing hooks with the same name

---

## Usage

### Normal Workflow

Once installed, the hooks run automatically:

```bash
# Stage your changes
git add Core/math/vector.cxx

# Commit (hooks run automatically)
git commit -m "Add vector normalization function"
```

**What happens:**
1. **Pre-commit hook** runs:
   - Formats `Core/math/vector.cxx` with clang-format
   - Re-stages the formatted file
   - Proceeds to commit

2. **Commit-msg hook** runs:
   - Validates your commit message
   - Checks for spelling mistakes
   - Ensures message is meaningful

3. If all checks pass, the commit succeeds!

### Checking Hook Status

```bash
python Scripts/setup_git_hooks.py --status
```

Output:
```
ℹ Git hooks directory: /path/to/XSigma/.git/hooks

✓ Installed - pre-commit: Formats C++ files with clang-format
✓ Installed - commit-msg: Validates commit messages

✓ clang-format is available: clang-format version 21.1.2
```

---

## Configuration

### Pre-Commit Hook Configuration

The pre-commit hook uses the project's `.clang-format` file located at the repository root. No additional configuration is needed.

**Supported file extensions:**
- `.h` - C/C++ header files
- `.hpp` - C++ header files
- `.hxx` - C++ header files
- `.cxx` - C++ source files
- `.cpp` - C++ source files
- `.cc` - C/C++ source files

### Commit-Msg Hook Configuration

Edit `.git/hooks/commit-msg` to customize:

**Minimum message length:**
```python
MIN_COMMIT_MESSAGE_LENGTH = 10  # Change to your preference
```

**Add custom spelling mistakes:**
```python
COMMON_SPELLING_MISTAKES = {
    'teh': 'the',
    'recieve': 'receive',
    # Add your own...
}
```

**Ignore technical terms:**
```python
IGNORE_WORDS = {
    'cpp', 'hpp', 'xsigma',
    # Add your own...
}
```

---

## Troubleshooting

### Hook Not Running

**Problem:** Hooks don't run when committing

**Solutions:**
1. Check if hooks are installed:
   ```bash
   python Scripts/setup_git_hooks.py --status
   ```

2. Ensure hooks are executable (Unix/Linux/macOS):
   ```bash
   chmod +x .git/hooks/pre-commit
   chmod +x .git/hooks/commit-msg
   ```

3. Check Python is in PATH:
   ```bash
   python --version
   ```

### clang-format Not Found

**Problem:** `clang-format not found in PATH`

**Solutions:**
1. Install clang-format (see [Installing clang-format](#installing-clang-format))

2. Verify installation:
   ```bash
   clang-format --version
   ```

3. Add clang-format to PATH:
   - **Windows:** Add LLVM bin directory to System PATH
   - **Unix:** Usually installed to `/usr/bin` or `/usr/local/bin`

### Python Not Found (Windows)

**Problem:** `Python was not found`

**Solutions:**
1. Install Python from https://www.python.org/
2. Or use the version already installed for XSigma build system
3. Ensure Python is in PATH

### Hook Fails on Specific File

**Problem:** Hook fails to format a specific file

**Solutions:**
1. Check if file is valid C++ syntax
2. Manually run clang-format to see error:
   ```bash
   clang-format -i path/to/file.cxx
   ```
3. Check `.clang-format` configuration

---

## Developer Workflow

### Typical Commit Process

```bash
# 1. Make changes to code
vim Core/math/vector.cxx

# 2. Stage changes
git add Core/math/vector.cxx

# 3. Commit (hooks run automatically)
git commit -m "Add vector normalization function

This function normalizes a vector to unit length.
Includes edge case handling for zero vectors."

# Output:
# ℹ Running XSigma pre-commit hook...
# ✓ Found clang-format: clang-format
# ℹ Formatting: Core/math/vector.cxx
# ✓ Formatted 1 file(s)
# ✓ Pre-commit hook completed successfully!
# ℹ ✓ Commit message validation passed
# [main abc1234] Add vector normalization function
```

### Amending Commits

```bash
# Amend previous commit (hooks run again)
git commit --amend

# Hooks will re-format any staged files
# and re-validate the commit message
```

### Interactive Staging

```bash
# Stage parts of files
git add -p Core/math/vector.cxx

# Commit (only staged parts are formatted)
git commit -m "Add vector normalization"
```

---

## Bypassing Hooks

### When to Bypass

⚠️ **Use sparingly!** Bypassing hooks should be rare:
- Emergency hotfixes
- Committing intentionally unformatted code (e.g., generated files)
- Working with legacy code that doesn't conform to style

### How to Bypass

```bash
# Bypass all hooks
git commit --no-verify -m "Emergency hotfix"

# Or use short form
git commit -n -m "Emergency hotfix"
```

### Bypassing Specific Hooks

To temporarily disable a specific hook:

```bash
# Rename the hook
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled

# Make your commit
git commit -m "Special commit"

# Re-enable the hook
mv .git/hooks/pre-commit.disabled .git/hooks/pre-commit
```

---

## CI/CD Integration

The hooks are designed to work seamlessly with CI/CD pipelines:

1. **Local enforcement:** Hooks catch issues before push
2. **CI validation:** CI can run the same checks
3. **Consistent formatting:** All developers use the same `.clang-format`

### Running Checks in CI

```yaml
# Example GitHub Actions workflow
- name: Check code formatting
  run: |
    # Find all C++ files and check formatting
    find Core -name "*.cxx" -o -name "*.h" | \
      xargs clang-format --dry-run --Werror
```

---

## Advanced Usage

### Custom Hook Scripts

You can extend the hooks by modifying the Python scripts:

**Pre-commit hook:** `.git/hooks/pre-commit`
**Commit-msg hook:** `.git/hooks/commit-msg`

Both are well-commented Python scripts that can be customized.

### Hook Chaining

If you need multiple pre-commit hooks:

1. Rename the XSigma hook:
   ```bash
   mv .git/hooks/pre-commit .git/hooks/pre-commit-xsigma
   ```

2. Create a new `pre-commit` that calls both:
   ```bash
   #!/bin/bash
   .git/hooks/pre-commit-xsigma || exit 1
   .git/hooks/pre-commit-custom || exit 1
   ```

---

## FAQ

**Q: Do hooks run on `git commit --amend`?**  
A: Yes, hooks run on all commit operations including amend.

**Q: Do hooks run on merge commits?**  
A: Yes, but only the commit-msg hook runs (no files are staged during merge).

**Q: Can I use a different formatter?**  
A: Yes, modify `.git/hooks/pre-commit` to call your preferred formatter.

**Q: Do hooks work with GUI git clients?**  
A: Yes, most GUI clients respect git hooks.

**Q: What if I don't have clang-format installed?**  
A: The pre-commit hook will fail with instructions on how to install it.

**Q: Can I disable hooks for a specific repository?**  
A: Yes, run `python Scripts/setup_git_hooks.py --uninstall`

---

## Support

For issues or questions:
1. Check this documentation
2. Check the [Troubleshooting](#troubleshooting) section
3. Contact the XSigma development team
4. File an issue in the project repository

---

## License

These hooks are part of the XSigma project and follow the same license.

