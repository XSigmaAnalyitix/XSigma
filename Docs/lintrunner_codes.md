# Lintrunner Codes Reference

All linters defined in `.lintrunner.toml` can be executed with:

```
lintrunner -a --filter <CODE> [paths...]
```

Formatters (marked below) accept `--fix` to rewrite files. Codes commented out in the config are listed for completeness with notes about their status.

## Python linters

### FLAKE8 *(formatter)*
- Enforces general Python style and the XSigma plug-in set (bugbear, simplify, etc.).
- Run: `lintrunner -a --filter FLAKE8 --fix`

### MYPY
- Runs `mypy` with `mypy.ini` across Scripts/Tools/Tests.
- Run: `lintrunner -a --filter MYPY`

### MYPYSTRICT
- Strict `mypy` profile (`mypy-strict.ini`) for scripts and tooling.
- Run: `lintrunner -a --filter MYPYSTRICT`

### PYREFLY
- Executes PyRefly static analysis for advanced Python correctness checks.
- Run: `lintrunner -a --filter PYREFLY`

### TYPEIGNORE
- Flags bare `# type: ignore` comments that need specific error codes.
- Run: `lintrunner -a --filter TYPEIGNORE --fix`

### TYPENOSKIP
- Ensures `mypy.ini` never uses `follow_imports = skip`.
- Run: `lintrunner -a --filter TYPENOSKIP --fix`

### NOQA
- Requires every `# noqa` to specify a code (e.g., `# noqa: F401`).
- Run: `lintrunner -a --filter NOQA --fix`

### ROOT_LOGGING
- Prevents direct use of the root `logging` logger; enforces module loggers.
- Run: `lintrunner -a --filter ROOT_LOGGING --fix`

### DEPLOY_DETECTION
- Replaces `sys.executable == ".torch_deploy"` checks with `torch._running_with_deploy()`.
- Run: `lintrunner -a --filter DEPLOY_DETECTION --fix`

### SHELLCHECK
- Runs ShellCheck against `.sh` scripts.
- Run: `lintrunner -a --filter SHELLCHECK`

### TESTOWNERS
- Verifies Python tests declare owners per repo policy.
- Run: `lintrunner -a --filter TESTOWNERS --fix`

### TEST_HAS_MAIN
- Ensures Python tests are executed via `test/run_test.py` rather than standalone mains.
- Run: `lintrunner -a --filter TEST_HAS_MAIN`

### CONTEXT_DECORATOR
- Bans using certain context managers as decorators (breaks profiling).
- Run: `lintrunner -a --filter CONTEXT_DECORATOR --fix`

### WORKFLOWSYNC
- Keeps critical GitHub workflow files aligned with canonical templates.
- Run: `lintrunner -a --filter WORKFLOWSYNC`

### NO_WORKFLOWS_ON_FORK
- Blocks workflows that should not run on forks.
- Run: `lintrunner -a --filter NO_WORKFLOWS_ON_FORK`

### CODESPELL *(formatter)*
- Fixes common spelling mistakes via Codespell.
- Run: `lintrunner -a --filter CODESPELL --fix`

### PYFMT *(formatter)*
- Runs the combined usort + ruff formatting stack.
- Run: `lintrunner -a --filter PYFMT --fix`

### PYPROJECT
- Validates `pyproject.toml` metadata and dependency entries.
- Run: `lintrunner -a --filter PYPROJECT`

### RUFF *(formatter)*
- Executes Ruff linting/formatting with repo configuration.
- Run: `lintrunner -a --filter RUFF --fix`

### META_NO_CREATE_UNBACKED
- Ensures `create_unbacked` calls only occur in approved meta registration files.
- Run: `lintrunner -a --filter META_NO_CREATE_UNBACKED --fix`

### DOCSTRING_LINTER
- Checks docstring presence/formatting in Scripts/Tools Python modules.
- Run: `lintrunner -a --filter DOCSTRING_LINTER --fix`

### IMPORT_LINTER
- Enforces repository-specific Python import layering contracts.
- Run: `lintrunner -a --filter IMPORT_LINTER`

### TEST_DEVICE_BIAS
- Detects Python tests that only exercise a single device backend.
- Run: `lintrunner -a --filter TEST_DEVICE_BIAS`

## C/C++ linters

### CLANGFORMAT *(formatter)*
- Applies `.clang-format` style to C/C++ headers and sources.
- Run: `lintrunner -a --filter CLANGFORMAT --fix`

### CLANGTIDY
- Runs clang-tidy with the repo’s `.clang-tidy` ruleset.
- Run: `lintrunner -a --filter CLANGTIDY`

### XSIGMA_UNUSED *(formatter)*
- Rewrites `XSIGMA_UNUSED` to `[[maybe_unused]]`.
- Run: `lintrunner -a --filter XSIGMA_UNUSED --fix`

### XSIGMA_NODISCARD *(formatter)*
- Replaces `XSIGMA_NODISCARD` with `[[nodiscard]]`.
- Run: `lintrunner -a --filter XSIGMA_NODISCARD --fix`

### INCLUDE *(formatter)*
- Converts quoted includes to angle brackets where policy requires.
- Run: `lintrunner -a --filter INCLUDE --fix`

### PYBIND11_INCLUDE *(formatter)*
- Ensures `torch/csrc/utils/pybind.h` is included before raw pybind11 headers.
- Run: `lintrunner -a --filter PYBIND11_INCLUDE --fix`

### ERROR_PRONE_ISINSTANCE
- Prevents `isinstance(..., (int|float))` in symbolic math modules, encouraging type aliases.
- Run: `lintrunner -a --filter ERROR_PRONE_ISINSTANCE --fix`

### PYBIND11_SPECIALIZATION
- Forces pybind11 template specializations into approved headers.
- Run: `lintrunner -a --filter PYBIND11_SPECIALIZATION`

### EXEC *(formatter)*
- Fixes executable bit issues and forbidden binary scripts via `exec_linter`.
- Run: `lintrunner -a --filter EXEC --fix`

### CUBINCLUDE *(formatter)*
- Replaces direct `<cub/...>` includes with `ATen/cuda/cub.cuh`.
- Run: `lintrunner -a --filter CUBINCLUDE --fix`

### RAWCUDA *(formatter)*
- Flags direct calls such as `cudaStreamSynchronize`; use `at::cuda` wrappers instead.
- Run: `lintrunner -a --filter RAWCUDA --fix`

### RAWCUDADEVICE *(formatter)*
- Bans `cudaSetDevice`/`cudaGetDevice` in favor of c10 wrappers.
- Run: `lintrunner -a --filter RAWCUDADEVICE --fix`

### CALL_ONCE *(formatter)*
- Rewrites `std::call_once` to `c10::call_once`.
- Run: `lintrunner -a --filter CALL_ONCE --fix`

### ONCE_FLAG *(formatter)*
- Rewrites `std::once_flag` to `c10::once_flag`.
- Run: `lintrunner -a --filter ONCE_FLAG --fix`

### ATEN_CPU_GPU_AGNOSTIC *(formatter)*
- Blocks compile-time GPU conditionals inside ATen CPU sources.
- Run: `lintrunner -a --filter ATEN_CPU_GPU_AGNOSTIC --fix`

## Build & config linters

### TYPEIGNORE / TYPENOSKIP *(see Python section)*
- Already covered above; they target type configuration.

### GHA
- Validates GitHub Actions YAMLs for structural issues.
- Run: `lintrunner -a --filter GHA`

### ACTIONLINT
- Runs the upstream `actionlint` binary on workflow files.
- Run: `lintrunner -a --filter ACTIONLINT`

### WORKFLOWSYNC / NO_WORKFLOWS_ON_FORK *(see Python section)*
- Already covered; they govern workflow behavior.

### PYPIDEP *(formatter)*
- Ensures any `pip install` statement pins versions.
- Run: `lintrunner -a --filter PYPIDEP --fix`

### CMAKE
- Invokes `cmakelint` with `.cmakelintrc`.
- Run: `lintrunner -a --filter CMAKE`

### CMAKE_MINIMUM_REQUIRED
- Checks CMakeLists/cmake files and requirement manifests for minimum-version statements.
- Run: `lintrunner -a --filter CMAKE_MINIMUM_REQUIRED`

### SHELLCHECK *(see Python section)*
- Shell script linting; listed earlier.

### PYPROJECT *(see Python section)*
- `pyproject.toml` validation; already listed.

### COPYRIGHT *(formatter)*
- Removes “Confidential and proprietary” strings that must not land in the repo.
- Run: `lintrunner -a --filter COPYRIGHT --fix`

### BAZEL_LINTER *(formatter)*
- Formats/validates the Bazel `WORKSPACE` using downloaded tooling.
- Run: `lintrunner -a --filter BAZEL_LINTER --fix`

### LINTRUNNER_VERSION
- Checks contributor lintrunner versions match required minimums.
- Run: `lintrunner -a --filter LINTRUNNER_VERSION`

### NATIVEFUNCTIONS *(commented out)*
- **Status**: Disabled in `.lintrunner.toml`.
- Purpose (when enabled): Formats and validates `aten/src/ATen/native/native_functions.yaml`.

### SET_LINTER *(commented out)*
- **Status**: Disabled in `.lintrunner.toml`.
- Purpose (when enabled): Rewrites raw `set()` usage in codegen scripts to `OrderedSet`.

## Miscellaneous linters

### NEWLINE *(formatter)*
- Normalizes newline endings and ensures files end with `\n`.
- Run: `lintrunner -a --filter NEWLINE --fix`

### SPACES *(formatter)*
- Removes trailing whitespace.
- Run: `lintrunner -a --filter SPACES --fix`

### TABS *(formatter)*
- Converts tabs to spaces where forbidden.
- Run: `lintrunner -a --filter TABS --fix`

### EXEC *(formatter)*
- Already described in the C/C++ section; applies repo-wide.

### TEST DEVICE & OWNERSHIP linters
- `TESTOWNERS`, `TEST_HAS_MAIN`, and `TEST_DEVICE_BIAS` ensure ownership metadata, central entry points, and device parity for tests.

---

Keep this document updated whenever `.lintrunner.toml` changes so contributors know how to run and interpret each linter.
