find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)

if(NOT CMAKE_CXX_CPPCHECK)
   message(FATAL_ERROR "Could not find the program cppcheck.")
else()
   list(
         APPEND CMAKE_CXX_CPPCHECK 
             "--suppressions-list=${CMAKE_CURRENT_SOURCE_DIR}/Scripts/cppcheck_suppressions.txt"
             "--platform=unspecified" 
             "--enable=all"
             "--force"
             "-q"
             "--inline-suppr" 
             "--library=qt"
             "--library=posix"
             "--library=gnu"
             "--library=bsd"
             "--library=windows"
             "--relative-paths=${CMAKE_CURRENT_SOURCE_DIR}"
             "--template='{id}:{file} :{line},{severity},{message}'"
     )
endif()