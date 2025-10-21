# XSigma Code Coverage Tooling – Overview

This document maps out the coverage workflow rooted in `Tools/code_coverage`. It explains how the orchestration script, helper packages, and reporting utilities collaborate to execute C++/Python test binaries and produce human-readable coverage summaries.

## High-Level Goals
- Provide a single entry point (`oss_coverage.py`) that configures output locations, parses flags, and drives coverage collection.
- Support both Clang (llvm-cov) and GCC (gcov) toolchains with the same surface area.
- Capture test runs across C++ and Python binaries, consolidate raw profiles, and emit JSON/HTML reports filtered to relevant source directories.
- Keep OSS usage self-contained while allowing reuse of fbcode-specific utilities via the platform abstraction.

## Execution Flow
1. **Entry Script – `oss_coverage.py`**
   - Derives `XSIGMA_COVERAGE_DIR` from `XSIGMA_BUILD_FOLDER` to keep artifacts beside the build tree.
   - Calls `package.oss.init.initialization()` to construct the `Option` object, resolve the build/test directories, and produce a filtered `TestList`.
   - Invokes `package.oss.cov_json.get_json_report(...)` to run, merge, and export raw coverage artefacts according to the active compiler.
   - Delegates summarisation to `package.tool.summarize_jsons.summarize_jsons(...)` when summaries are requested.

2. **Argument Processing – `package.util.utils_init` & `package.oss.init`**
   - `create_folders()` provisions the profile directory tree (`profile/json`, `profile/merged`, `profile/summary`, etc.).
   - `add_arguments_utils()` and `add_arguments_oss()` register shared flags (`--run`, `--merge`, `--summary`, `--clean`, `--interest-only`, etc.) plus OSS-specific selectors (`--build-folder`, `--test-subfolder`, `--run-only`).
   - `get_options()` instantiates an `Option` value where unspecified stages default to the full pipeline (build → run → merge → export → summary).
   - Test discovery (`get_test_list_by_type`, `get_test_list`) walks the selected build folder and yields `Test` instances for each binary/script.

3. **Running Tests – `package.oss.run`**
   - `clang_run(...)` and `gcc_run(...)` iterate the `TestList`, using tool-specific runners:
     - C++ executables route through `package.tool.utils.run_cpp_test`.
     - Python scripts run via `package.oss.utils.run_oss_python_test`, keeping execution inside the `build/<test_subfolder>` directory.
   - Each clang invocation sets `LLVM_PROFILE_FILE` to capture `.profraw` outputs; GCC relies on `.gcda`.

4. **Exporting Raw Data**
   - **Clang**: `package.tool.clang_coverage` merges `.profraw` into `.merged` (`llvm-profdata merge`) and exports JSON (`llvm-cov export`) while applying filename filters so ThirdParty and test fixtures do not skew results.
   - **GCC**: `package.tool.gcc_coverage` runs `gcov -i` on each `.gcda`, normalises file names, and unpacks JSON.

5. **Summaries & Reports – `package.tool.summarize_jsons`**
   - Detects the toolchain (`detect_compiler_type`) and dispatches to the appropriate parser (`LlvmCoverageParser` or `GcovCoverageParser`).
   - Aggregates per-file `CoverageRecord` values into `covered_lines` / `uncovered_lines`, filtered by user-provided folders.
   - Emits multiple outputs:
     - `line_summary` (per-file line lists).
     - `file_summary` (coverage percentages sorted by health).
     - Optional multi-file HTML dashboards (`HtmlReportGenerator`) with syntax-highlighted source excerpts.
     - For GCC, falls back to traditional `lcov`/`genhtml` reporting.

6. **Utilities & Settings**
   - `package.util.setting`: centralises environment-derived paths (profile tree, log directory), enumerations (`CompilerType`, `TestType`, `TestPlatform`), and data containers (`Option`, `Test`).
   - `package.util.utils`: shared helpers for logging, filesystem hygiene, compiler/platform detection, and error reporting.
   - `package.oss.utils`: OSS-specific path resolution, compiler inference (via `cc -v`), GCDA cleanup, and shared-library discovery for linking coverage reports correctly.

## Directory Layout
```
Tools/code_coverage/
├── oss_coverage.py                     # CLI entry point
├── README.md                           # User-facing instructions
├── package/
│   ├── oss/                            # OSS argument parsing and runners
│   ├── tool/                           # Compiler-agnostic coverage tooling
│   │   ├── parser/                     # JSON → CoverageRecord translators
│   │   └── html_report_generator.py
│   └── util/                           # Shared settings, logging, helpers
└── profile/ (generated)                # Runtime output (json, merged, summary, log)
```

## Key Environment Variables
- `XSIGMA_BUILD_FOLDER`: build tree containing compiled tests; required by the CLI.
- `XSIGMA_TEST_SUBFOLDER`: subdirectory inside the build folder where executables reside (`bin` by default).
- `XSIGMA_COVERAGE_DIR`: overrides the default profile output root (auto-derived when running via `oss_coverage.py`).
- `LLVM_TOOL_PATH`: optional override used by clang tooling.
- `RAW_PROFILES_FOLDER`: alternative location for `.profraw` when set.

## Typical CLI Invocations
- `python oss_coverage.py --build-folder build --run-only atest`: run a specific binary and collect coverage with default summaries.
- `python oss_coverage.py --build-folder build --run --summary --interest-only Library/Core`: run without merging/exporting, then summarise limited to `Library/Core`.
- `python oss_coverage.py --build-folder build --clean`: purge historical profiles and logs before exiting.

Refer to `docs/code_coverage/module_reference.md` for a module-by-module breakdown of classes and functions.
