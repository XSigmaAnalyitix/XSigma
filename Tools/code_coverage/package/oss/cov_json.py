from ..tool import clang_coverage, gcc_coverage
from ..util.setting import CompilerType, Option, TestList, TestPlatform
from ..util.utils import check_compiler_type
from .init import detect_compiler_type  # type: ignore[attr-defined]
from .run import clang_run, gcc_run


def get_json_report(
    test_list: TestList,
    options: Option,
    build_folder: str,
    test_subfolder: str = "bin",
) -> None:
    cov_type = detect_compiler_type()
    check_compiler_type(cov_type)
    if cov_type == CompilerType.CLANG:
        # run
        if options.need_run:
            clang_run(test_list, build_folder, test_subfolder)
        # merge && export
        if options.need_merge:
            clang_coverage.merge(test_list, TestPlatform.OSS)
        if options.need_export:
            clang_coverage.export(
                test_list, TestPlatform.OSS, build_folder, test_subfolder
            )
            # Generate HTML coverage report
            clang_coverage.show_html(
                test_list, TestPlatform.OSS, build_folder, test_subfolder
            )
    elif cov_type == CompilerType.GCC:
        # run
        if options.need_run:
            gcc_run(test_list, build_folder, test_subfolder)
        # export
        if options.need_export:
            gcc_coverage.export()
