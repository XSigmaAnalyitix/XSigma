# Git Hooks Onboarding Guide for New Developers

Welcome to the XSigma project! This guide will help you set up and understand the automated git hooks that maintain code quality.

---

## 🎯 What You Need to Know

### The Basics

When you commit code to XSigma, two automated checks run:

1. **Code Formatting** - Your C++ files are automatically formatted to match project style
2. **Message Validation** - Your commit message is checked for quality and spelling

**Don't worry!** These hooks are helpful, not restrictive. They:
- ✅ Save you time by auto-formatting code
- ✅ Catch typos before they reach the repository
- ✅ Ensure consistent code style across the team
- ✅ Can be bypassed in emergencies

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install clang-format

**Windows (using Chocolatey):**
```powershell
choco install llvm
```

**macOS:**
```bash
brew install clang-format
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install clang-format
```

**Linux (RHEL/CentOS/Fedora):**
```bash
sudo yum install clang-tools-extra
```

### Step 2: Verify Installation

```bash
clang-format --version
```

You should see something like: `clang-format version 21.1.2`

### Step 3: Install Git Hooks

```bash
cd Scripts
python setup_git_hooks.py --install
```

You should see:
```
✓ Git hooks installed successfully!
```

### Step 4: Test It Works

```bash
python test_git_hooks.py
```

All tests should pass! ✅

---

## 💡 Your First Commit with Hooks

Let's walk through what happens when you make your first commit:

### Example Scenario

```bash
# 1. You edit a file
vim Core/math/vector.cxx

# 2. You stage it
git add Core/math/vector.cxx

# 3. You commit
git commit -m "Add vector normalization function"
```

### What Happens Behind the Scenes

```
ℹ Running XSigma pre-commit hook...
✓ Found clang-format: clang-format
ℹ Found 1 staged file(s)

ℹ Formatting 1 C++ file(s)...
ℹ Formatting: Core/math/vector.cxx

✓ Formatted 1 file(s):
  - Core/math/vector.cxx

✓ Pre-commit hook completed successfully!
ℹ ✓ Commit message validation passed

[main abc1234] Add vector normalization function
 1 file changed, 10 insertions(+)
```

**What just happened:**
1. ✅ Your file was automatically formatted to match project style
2. ✅ The formatted file was re-staged
3. ✅ Your commit message was validated
4. ✅ Your commit succeeded!

---

## ⚠️ Common Scenarios

### Scenario 1: Commit Message Too Short

```bash
git commit -m "fix bug"
```

**Result:**
```
✗ ERROR: Commit message validation failed:

  • Commit message is too short (7 characters). Minimum 10 characters required.

ℹ Please fix the issues above and try again.
```

**Solution:**
```bash
git commit -m "Fix null pointer bug in vector class"
```

### Scenario 2: Spelling Mistake

```bash
git commit -m "Fix teh vector normalization bug"
```

**Result:**
```
✗ ERROR: Commit message validation failed:

  • Spelling mistakes found: 'teh' -> 'the'

ℹ Please fix the issues above and try again.
```

**Solution:**
```bash
git commit -m "Fix the vector normalization bug"
```

### Scenario 3: Placeholder Message

```bash
git commit -m "wip"
```

**Result:**
```
✗ ERROR: Commit message validation failed:

  • Commit message appears to be a placeholder: 'wip'. Please provide a meaningful commit message.

ℹ Please fix the issues above and try again.
```

**Solution:**
```bash
git commit -m "Work in progress: Implementing vector normalization"
```

### Scenario 4: Emergency Hotfix

Sometimes you need to commit quickly without running hooks:

```bash
git commit --no-verify -m "Emergency: Fix production crash"
```

⚠️ **Use sparingly!** Only for true emergencies.

---

## 📝 Writing Good Commit Messages

### ✅ Good Examples

```bash
# Clear and descriptive
git commit -m "Add matrix multiplication function"

# Explains what and why
git commit -m "Fix memory leak in vector destructor

The destructor was not properly releasing allocated memory
when the vector was destroyed. This fix ensures proper cleanup."

# References issue
git commit -m "Fix issue #123: Incorrect vector normalization

The normalization function was dividing by zero for null vectors.
Added check to return zero vector in this case."
```

### ❌ Bad Examples

```bash
# Too short
git commit -m "fix"

# Placeholder
git commit -m "wip"

# Not descriptive
git commit -m "update code"

# Spelling mistakes
git commit -m "Fix teh bug in teh vector class"
```

### Tips for Great Commit Messages

1. **Start with a verb:** "Add", "Fix", "Update", "Remove", "Refactor"
2. **Be specific:** What exactly did you change?
3. **Keep first line under 72 characters**
4. **Use proper spelling and grammar**
5. **Explain why, not just what** (in the body)

---

## 🔧 Customization

### Adjusting Minimum Message Length

If you find the 10-character minimum too restrictive:

1. Open `.git/hooks/commit-msg`
2. Find this line:
   ```python
   MIN_COMMIT_MESSAGE_LENGTH = 10
   ```
3. Change to your preference (e.g., `15` or `20`)

### Adding Technical Terms to Ignore List

If the spell checker flags technical terms:

1. Open `.git/hooks/commit-msg`
2. Find the `IGNORE_WORDS` set
3. Add your terms:
   ```python
   IGNORE_WORDS = {
       'cpp', 'hpp', 'xsigma',
       'myterm',  # Add your term here
   }
   ```

---

## 🐛 Troubleshooting

### Problem: "clang-format not found"

**Solution:**
1. Install clang-format (see Step 1 above)
2. Verify it's in your PATH:
   ```bash
   # Windows
   where clang-format
   
   # Unix/Linux/macOS
   which clang-format
   ```
3. Restart your terminal

### Problem: "Hook not running"

**Solution:**
1. Check hook status:
   ```bash
   python Scripts/setup_git_hooks.py --status
   ```
2. Reinstall hooks:
   ```bash
   python Scripts/setup_git_hooks.py --install
   ```
3. On Unix/Linux/macOS, ensure executable:
   ```bash
   chmod +x .git/hooks/pre-commit
   chmod +x .git/hooks/commit-msg
   ```

### Problem: "Python not found" (Windows)

**Solution:**
1. Ensure Python is installed (required for XSigma anyway)
2. Add Python to PATH
3. Restart terminal

### Problem: "Hook is too slow"

**Solution:**
For large commits, you can bypass hooks:
```bash
git commit --no-verify -m "Large refactoring"
```

Then manually format:
```bash
clang-format -i path/to/file.cxx
```

---

## 🎓 Advanced Usage

### Amending Commits

Hooks run on amended commits too:

```bash
git commit --amend
```

The hooks will re-check everything.

### Interactive Staging

You can stage parts of files:

```bash
git add -p Core/math/vector.cxx
git commit -m "Add normalization function"
```

Only staged parts will be formatted.

### Checking Hook Status

```bash
python Scripts/setup_git_hooks.py --status
```

Shows:
- Which hooks are installed
- If clang-format is available
- Hook versions

---

## 📚 Additional Resources

- **Full Documentation:** `docs/GIT_HOOKS.md`
- **Quick Reference:** `GIT_HOOKS_QUICKSTART.md`
- **Test Suite:** `python Scripts/test_git_hooks.py`
- **Setup Script:** `python Scripts/setup_git_hooks.py --help`

---

## 🤝 Getting Help

If you encounter issues:

1. **Check the documentation** (links above)
2. **Run the test suite** to diagnose:
   ```bash
   python Scripts/test_git_hooks.py
   ```
3. **Ask a team member** - we're here to help!
4. **File an issue** if you find a bug

---

## ✅ Checklist for New Developers

Before your first commit, make sure:

- [ ] clang-format is installed and in PATH
- [ ] Git hooks are installed (`python Scripts/setup_git_hooks.py --install`)
- [ ] Test suite passes (`python Scripts/test_git_hooks.py`)
- [ ] You understand how to write good commit messages
- [ ] You know how to bypass hooks in emergencies (`--no-verify`)

---

## 🎉 You're Ready!

Congratulations! You're all set up with XSigma's git hooks. The hooks will help you:

- ✅ Maintain consistent code style
- ✅ Write better commit messages
- ✅ Catch mistakes early
- ✅ Contribute high-quality code

**Happy coding!** 🚀

---

## 💬 Feedback

We're always improving! If you have suggestions for the hooks or this documentation, please let the team know.

---

**Last Updated:** 2025-10-05  
**Version:** 1.0

