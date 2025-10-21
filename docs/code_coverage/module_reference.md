# Code Coverage Module Reference

This reference enumerates the modules under `Tools/code_coverage`, capturing the key classes, functions, and their responsibilities. Use it when extending functionality or wiring the coverage tooling into new build/test workflows.

## Entry Point

### `oss_coverage.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `report_coverage()` | function | Main driver: initialises options/test lists, runs coverage collection, triggers summaries, and records total duration. |

## OSS Platform Layer (`package/oss`)

### `init.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `BLOCKED_PYTHON_TESTS` | set[str] | Known Python tests to skip under clang to avoid oversized profiles. |
| `initialization()` | function | Creates output directories, parses CLI args, cleans artefacts, returns `(Option, TestList, interested_folders)`. |
| `add_arguments_oss(parser)` | function | Adds `--build-folder`, `--test-subfolder`, `--run-only` flags to the shared parser. |
| `parse_arguments(parser)` | function | Parses arguments into option flags plus build/test folder names. |
| `get_test_list_by_type(...)` | function | Walks build output and returns discovered `Test` objects for C++ or Python binaries/scripts. |
| `get_test_list(...)` | function | Combines C++ and Python test lists, enforcing `--run-only` constraints. |
| `empty_list_if_none(arg)` | function | Normalises optional lists to `[]`. |
| `gcc_export_init()` | function | Clears the JSON folder and recreates it before a GCC export. |
| `get_python_run_only(...)` | function | Selects Python tests to execute based on compiler and blocked list. |
| `print_init_info(...)` | function | Logs resolved paths and compiler metadata to the coverage log. |

### `cov_json.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `get_json_report(test_list, options, build_folder, test_subfolder)` | function | Detects compiler, runs tests, merges profiles, exports JSON, and triggers HTML report generation (clang) according to `Option` flags. |

### `run.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `clang_run(tests, build_folder, test_subfolder)` | function | Executes each test with `LLVM_PROFILE_FILE` configured so llvm-cov can collect `.profraw` data. |
| `gcc_run(tests, build_folder, test_subfolder)` | function | Executes each test under GCC/gcov, producing `.gcda` files. |

### `utils.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `get_oss_binary_folder(test_type, build_folder, test_subfolder)` | function | Resolves the directory that contains test executables/scripts. |
| `get_oss_shared_library(build_folder, test_subfolder)` | function | Lists shared libraries to pass to llvm-cov for symbol resolution. |
| `get_oss_binary_file(test_name, test_type, build_folder, test_subfolder)` | function | Builds the full command for invoking a test binary or Python script. |
| `get_llvm_tool_path()` | function | Determines the llvm toolchain path, honouring `LLVM_TOOL_PATH`. |
| `get_pytorch_folder()` | function | Resolves the XSigma source tree root. |
| `detect_compiler_type()` | function | Identifies active compiler (clang/gcc) via `$CXX` or `cc -v`. |
| `clean_up_gcda()` / `get_gcda_files()` | function | Finds and removes residual `.gcda` files before a GCC pass. |
| `run_oss_python_test(binary_file, build_folder, test_subfolder)` | function | Runs Python tests in the correct working directory. |

## Shared Utilities (`package/util`)

### `setting.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `HOME_DIR`, `TOOLS_FOLDER`, `PROFILE_DIR`, `JSON_FOLDER_BASE_DIR`, `MERGED_FOLDER_BASE_DIR`, `SUMMARY_FOLDER_DIR`, `LOG_DIR` | constants | Canonical locations for coverage artefacts and logs. |
| `TestType`, `TestPlatform`, `CompilerType` | enums | Normalise compiler/platform/test distinctions throughout the tooling. |
| `Test` | class | Data holder for individual test descriptors (`name`, `target_pattern`, `test_type`, etc.). |
| `Option` | class | Flag container toggling the `run`, `merge`, `export`, `summary`, and `pytest` stages. |

### `utils.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `convert_time(seconds)` | function | Formats durations as `H:MM:SS`. |
| `print_time(message, start_time, summary_time=False)` | function | Logs elapsed time to `log.txt`. |
| `print_log(*args)` / `print_error(*args)` | functions | Append log entries with `[LOG]` or `[ERROR]` prefixes. |
| `remove_file(path)` / `remove_folder(path)` / `create_folder(*paths)` | functions | File-system helpers for cleaning or provisioning directories. |
| `clean_up()` | function | Removes the profile directory and exits—used by `--clean`. |
| `convert_to_relative_path(whole_path, base_path)` | function | Strips a base directory from a path when mirroring folder structures. |
| `replace_extension(filename, ext)` | function | Rewrites file extensions (e.g., `.merged` → `.json`). |
| `related_to_test_list(file_name, test_list)` | function | Determines whether an artefact belongs to a requested test. |
| `get_raw_profiles_folder()` | function | Returns the profiling folder (`RAW_PROFILES_FOLDER` override aware). |
| `detect_compiler_type(platform)` | function | Platform-aware compiler detection wrapper. |
| `get_test_name_from_whole_path(path)` | function | Extracts a test name from merged profile filenames. |
| `check_compiler_type`, `check_platform_type`, `check_test_type` | functions | Validation helpers that raise when encountering unsupported values. |
| `raise_no_test_found_exception(cpp_folder, py_folder)` | function | Raises a descriptive exception when no tests are discovered. |

### `utils_init.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `remove_files()` | function | Clears the log file before a new run. |
| `create_folders()` | function | Creates all profile subdirectories and log folder. |
| `add_arguments_utils(parser)` | function | Adds shared CLI flags controlling coverage stages and filters. |
| `have_option(have_stage, option)` | function | Helper used during option aggregation. |
| `get_options(args)` | function | Builds the `Option` structure from parsed CLI arguments (defaults to running the full pipeline). |

## Coverage Tooling (`package/tool`)

### `utils.py`
| Symbol | Type | Purpose |
| --- | --- | --- |
| `run_cpp_test(binary_file)` | function | Runs a C++ binary via `subprocess.check_call`, logging failures. |
| `get_tool_path_by_platform(platform)` | function | Chooses the llvm tool directory for OSS vs fbcode environments. |

### `clang_coverage.py`
Key functions:
| Symbol | Purpose |
| --- | --- |
| `get_coverage_filters()` | Returns regex filters to exclude ThirdParty/testing files from coverage. |
| `build_llvm_cov_filter_args()` | Converts filters into `llvm-cov` argument pairs. |
| `create_corresponding_folder(cur_path, prefix_cur_path, dir_list, new_base_folder)` | Mirrors directory hierarchy when producing merged profiles. |
| `run_target(...)` | Runs a single test under clang coverage, handling Python vs C++ dispatch. |
| `merge_target(raw_file, merged_file, platform_type)` | Invokes `llvm-profdata merge` for a `.profraw`. |
| `export_target(merged_file, json_file, binary_file, shared_library_list, platform_type)` | Runs `llvm-cov export` to convert merged profiles into JSON. |
| `export(test_list, platform_type, build_folder, test_subfolder)` | Orchestrates export across all tests. |
| `merge(test_list, platform_type)` | Performs merges for each raw profile. |
| `show_html(test_list, platform_type, build_folder, test_subfolder)` | Generates per-test HTML reports via `llvm-cov show`. |
| `show_multifile_html(covered_lines, uncovered_lines, source_root)` | Builds aggregated HTML dashboards using `HtmlReportGenerator`. |

### `gcc_coverage.py`
| Symbol | Purpose |
| --- | --- |
| `update_gzip_dict(gzip_dict, file_name)` | Disambiguates gcov output filenames when duplicates arise. |
| `run_target(binary_file, test_type, build_folder, test_subfolder)` | Executes GCC coverage targets for C++ or Python tests. |
| `export()` | Processes `.gcda` files into JSON using `gcov -i`, storing them under `profile/json`. |

### `summarize_jsons.py`
Key symbols:
| Symbol | Purpose |
| --- | --- |
| `covered_lines`, `uncovered_lines`, `tests_type` | Module-level stores accumulating coverage data. |
| `transform_file_name(file_path, interested_folders, platform)` | Normalises file paths and applies interested-folder slicing. |
| `is_intrested_file(file_path, interested_folders, platform)` | Filters out unwanted files (third-party, generated, build artefacts). |
| `get_json_obj(json_file)` | Reads exported JSON while skipping preliminary warning text. |
| `parse_json(json_file, platform)` | Builds `CoverageRecord` lists by invoking the appropriate parser. |
| `parse_jsons(test_list, interested_folders, platform)` | Walks the JSON directory and collects coverage data. |
| `update_coverage(coverage_records, interested_folders, platform)` | Populates `covered_lines` / `uncovered_lines` dictionaries. |
| `update_set()` | Removes overlapping lines from the uncovered set. |
| `summarize_jsons(test_list, interested_folders, coverage_only, platform)` | Top-level summary routine: generates textual and HTML reports. |

### `print_report.py`
| Symbol | Purpose |
| --- | --- |
| `CoverageItem` | Tuple alias capturing `(path, percentage, covered, total)`. |
| `key_by_percentage`, `key_by_name` | Sort keys for coverage summaries. |
| `is_intrested_file(file_path, interested_folders)` | Helper used by HTML filtering. |
| `is_this_type_of_tests(target_name, test_set_by_type)` | Determines which tests succeeded/failed. |
| `print_test_by_type(...)`, `print_test_condition(...)` | Emit textual summaries of test execution outcomes. |
| `line_oriented_report(...)` | Writes `line_summary` files listing covered/uncovered line numbers. |
| `print_file_summary(...)`, `print_file_oriented_report(...)` | Build and emit aggregate file coverage stats. |
| `file_oriented_report(...)` | Creates `file_summary` sorted by weakest coverage. |
| `get_html_ignored_pattern()` | Supplies patterns for pruning `lcov` reports. |
| `html_oriented_report()` | Runs `lcov` + `genhtml` to produce traditional HTML coverage (GCC). |
| `generate_multifile_html_report(...)` | Convenience wrapper around `HtmlReportGenerator`. |

### `html_report_generator.py`
| Symbol | Purpose |
| --- | --- |
| `HtmlReportGenerator` | Class that produces index and per-file HTML coverage pages with basic navigation and statistics. |
| `generate_report(...)` | Entry point combining stats, index page, and per-file views. |
| `_calculate_statistics(...)`, `_generate_index_page(...)`, `_generate_file_reports(...)` | Internal helpers to aggregate coverage data and write files. |
| `_get_index_html(stats)` | Renders the summary HTML template. |
| `_get_file_html(file_path, covered, uncovered, source_root)` | Renders individual file coverage pages. |
| `_load_source_lines(file_path, source_root)` | Reads source file contents when available. |
| `_get_coverage_class(percentage)` | Maps coverage percentages to CSS classes. |

### Parser Subpackage (`package/tool/parser`)
| Module | Key Symbols | Purpose |
| --- | --- | --- |
| `coverage_record.py` | `CoverageRecord` | Immutable container for per-file coverage data. |
| `llvm_coverage_parser.py` | `LlvmCoverageParser`, `_collect_coverage`, `_skip_coverage` | Converts `llvm-cov export` JSON into `CoverageRecord` instances. |
| `llvm_coverage_segment.py` | `LlvmCoverageSegment`, `parse_segments` | Represents and interprets llvm coverage segments to produce line ranges. |
| `gcov_coverage_parser.py` | `GcovCoverageParser` | Converts gcov JSON output into `CoverageRecord` lists. |

## Reporting Artefacts
- Text files (`profile/summary/line_summary`, `file_summary`) summarise coverage numerically.
- HTML reports (per-test and aggregated) live under `profile/html` or `profile/summary/html_details`.
- Raw data persist under `profile/raw` (`.profraw`), `profile/merged` (`.merged`), and `profile/json` (`.json`).

Use this map as the authoritative guide when adding new coverage stages, introducing additional report formats, or integrating with CI jobs.
