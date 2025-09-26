if(UNIX)
  find_package(Numa)
  if(NUMA_FOUND)
    include_directories(SYSTEM ${Numa_INCLUDE_DIR})
    list(APPEND XSIGMA_DEPENDENCY_LIBS "${Numa_LIBRARIES}")
  else()
    message(
      WARNING
        "Not compiling with NUMA. Suppress this warning with -DXSIGMA_ENABLE_NUMA=OFF"
    )
  endif()
endif()
