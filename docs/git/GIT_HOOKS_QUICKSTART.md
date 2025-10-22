# Git Hooks Quick Start Guide

## What Are Git Hooks?

Git hooks are automated scripts that run during git operations. The XSigma project uses hooks to:

1. **Automatically format C++ code** with clang-format before commits
2. **Validate commit messages** for quality and spelling

---

## Quick Commands

### Install Hooks
```bash
python Scripts/setup_git_hooks.py --install
```

### Check Status
```bash
python Scripts/setup_git_hooks.py --status
```

### Test Hooks
```bash
python Scripts/test_git_hooks.py
```

### Uninstall Hooks
```bash
python Scripts/setup_git_hooks.py --uninstall
```

---

## Prerequisites

✅ **Python 3.6+** (already required by XSigma)  
✅ **Git** (obviously!)  
✅ **clang-format** (must be in PATH)

### Installing clang-format

**Windows:**
```powershell
choco install llvm
```

**macOS:**
```bash
brew install clang-format
```

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install clang-format

# RHEL/CentOS/Fedora
sudo yum install clang-tools-extra
```

---

## What Happens When You Commit?

### 1. Pre-Commit Hook Runs

```bash
git add Core/math/vector.cxx
git commit -m "Add vector normalization"
```

**Output:**
```
ℹ Running XSigma pre-commit hook...
✓ Found clang-format: clang-format
ℹ Formatting: Core/math/vector.cxx
✓ Formatted 1 file(s)
✓ Pre-commit hook completed successfully!
```

**What happened:**
- ✅ `vector.cxx` was automatically formatted
- ✅ Formatted file was re-staged
- ✅ Commit proceeds with properly formatted code

### 2. Commit-Msg Hook Runs

**Validates your commit message:**
- ✅ Not empty or whitespace
- ✅ Minimum 10 characters
- ✅ Not a placeholder (e.g., "wip", "test")
- ✅ No spelling mistakes

**If validation fails:**
```
✗ ERROR: Commit message validation failed:

  • Spelling mistakes found: 'teh' -> 'the'

ℹ Please fix the issues above and try again.
```

---

## Bypassing Hooks (Use Sparingly!)

```bash
# Bypass all hooks
git commit --no-verify -m "Emergency hotfix"

# Or short form
git commit -n -m "Emergency hotfix"
```

⚠️ **Only bypass hooks for:**
- Emergency hotfixes
- Intentionally unformatted code
- Working with legacy code

---

## Troubleshooting

### Hook Not Running?

1. **Check installation:**
   ```bash
   python Scripts/setup_git_hooks.py --status
   ```

2. **Make executable (Unix/Linux/macOS):**
   ```bash
   chmod +x .git/hooks/pre-commit
   chmod +x .git/hooks/commit-msg
   ```

### clang-format Not Found?

1. **Install clang-format** (see above)

2. **Verify installation:**
   ```bash
   clang-format --version
   ```

3. **Check PATH:**
   ```bash
   # Windows
   where clang-format
   
   # Unix/Linux/macOS
   which clang-format
   ```

---

## Supported File Extensions

The pre-commit hook formats these C++ file types:
- `.h` - C/C++ headers
- `.hpp` - C++ headers
- `.hxx` - C++ headers
- `.cxx` - C++ source
- `.cpp` - C++ source
- `.cc` - C/C++ source

---

## Example Workflow

```bash
# 1. Make changes
vim Core/math/vector.cxx

# 2. Stage changes
git add Core/math/vector.cxx

# 3. Commit (hooks run automatically)
git commit -m "Add vector normalization function

This function normalizes a vector to unit length.
Includes edge case handling for zero vectors."

# Output:
# ℹ Running XSigma pre-commit hook...
# ✓ Formatted 1 file(s)
# ✓ Pre-commit hook completed successfully!
# ℹ ✓ Commit message validation passed
# [main abc1234] Add vector normalization function
```

---

## Configuration

### Customize Commit Message Validation

Edit `.git/hooks/commit-msg`:

```python
# Change minimum message length
MIN_COMMIT_MESSAGE_LENGTH = 10  # Default: 10

# Add custom spelling mistakes
COMMON_SPELLING_MISTAKES = {
    'teh': 'the',
    'recieve': 'receive',
    # Add your own...
}

# Ignore technical terms
IGNORE_WORDS = {
    'cpp', 'hpp', 'xsigma',
    # Add your own...
}
```

### Customize Code Formatting

The hooks use the project's `.clang-format` file at the repository root. No additional configuration needed!

---

## FAQ

**Q: Do hooks run on `git commit --amend`?**  
A: Yes, hooks run on all commit operations.

**Q: Do hooks work with GUI git clients?**  
A: Yes, most GUI clients respect git hooks.

**Q: Can I disable hooks temporarily?**  
A: Yes, use `git commit --no-verify` or `git commit -n`

**Q: What if I don't have clang-format?**  
A: The hook will fail with installation instructions.

**Q: Do hooks modify my files?**  
A: Yes, the pre-commit hook formats files in-place and re-stages them.

**Q: Can I customize the hooks?**  
A: Yes, edit `.git/hooks/pre-commit` and `.git/hooks/commit-msg`

---

## Full Documentation

For complete documentation, see: [`docs/GIT_HOOKS.md`](docs/GIT_HOOKS.md)

---

## Support

For issues or questions:
1. Check [`docs/GIT_HOOKS.md`](docs/GIT_HOOKS.md)
2. Run `python Scripts/test_git_hooks.py` to diagnose issues
3. Contact the XSigma development team

---

## Files Created

```
.git/hooks/pre-commit          # Pre-commit hook (formats C++ files)
.git/hooks/commit-msg           # Commit message validation hook
Scripts/setup_git_hooks.py      # Installation script
Scripts/test_git_hooks.py       # Test script
docs/GIT_HOOKS.md               # Full documentation
GIT_HOOKS_QUICKSTART.md         # This file
```

---

**Happy coding! 🚀**

