find_program(iwyu_path NAMES include-what-you-use iwyu)
if(iwyu_path)
  message("IWYU found: ${iwyu_path}")
  set(INCLUDE_WHAT_YOU_USE
    "${iwyu_path}"
    "-Xiwyu" "--cxx17ns"
    "-Xiwyu" "--no_stdinc"
    "-Xiwyu" "--mapping_file=${XSIGMA_SOURCE_DIR}/Scripts/iwyu_exclusion.imp"
    "-Xiwyu" "--max_line_length=100"
    "-Xiwyu" "--pch_in_code"
    "-Xiwyu" "--verbose=3")
  mark_as_advanced(INCLUDE_WHAT_YOU_USE)
endif()
