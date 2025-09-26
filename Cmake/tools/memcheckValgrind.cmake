# XSigma Valgrind Configuration
find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
message("--Executing test suite with Valgrind (${CMAKE_MEMORYCHECK_COMMAND})")
set(CMAKE_MEMORYCHECK_COMMAND_OPTIONS
    " --gen-suppressions=all -v --trace-children=yes --leak-check=yes --show-reachable=yes --num callers=50"
)
set(memcheck_command
    "${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS}")
separate_arguments(memcheck_command)
set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE
    "${PROJECT_SOURCE_DIR}/CMake/xsigmaValgrindSuppression.txt")

