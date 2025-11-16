# =============================================================================
# Intel TBB (Threading Building Blocks) BUILD Configuration
# =============================================================================
# High-performance task scheduling and memory allocation library
# Equivalent to ThirdParty/oneTBB
# =============================================================================

package(default_visibility = ["//visibility:public"])

# =============================================================================
# TBB Core Library
# =============================================================================
# Note: This uses the system-installed TBB library (e.g., from Homebrew).
# The headers are from the downloaded archive, but the libraries are linked
# from the system installation.

cc_library(
    name = "tbb",
    hdrs = glob([
        "include/tbb/**/*.h",
        "include/tbb/**/*.hpp",
        "include/oneapi/tbb/**/*.h",
        "include/oneapi/tbb/**/*.hpp",
    ]),
    includes = [
        "include",
        "include/oneapi",
    ],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [
            "-L/opt/homebrew/lib",
            "-ltbb",
        ],
        "//conditions:default": ["-ltbb"],
    }),
)

# =============================================================================
# TBB Malloc Library
# =============================================================================

cc_library(
    name = "tbbmalloc",
    hdrs = glob([
        "include/tbb/tbbmalloc.h",
        "include/oneapi/tbb/tbbmalloc.h",
    ]),
    includes = [
        "include",
        "include/oneapi",
    ],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [
            "-L/opt/homebrew/lib",
            "-ltbbmalloc",
        ],
        "//conditions:default": ["-ltbbmalloc"],
    }),
)

# =============================================================================
# TBB Malloc Proxy Library
# =============================================================================

cc_library(
    name = "tbbmalloc_proxy",
    hdrs = glob([
        "include/tbb/tbbmalloc_proxy.h",
        "include/oneapi/tbb/tbbmalloc_proxy.h",
    ]),
    includes = [
        "include",
        "include/oneapi",
    ],
    deps = [":tbbmalloc"],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [
            "-L/opt/homebrew/lib",
            "-ltbbmalloc_proxy",
        ],
        "//conditions:default": ["-ltbbmalloc_proxy"],
    }),
)

