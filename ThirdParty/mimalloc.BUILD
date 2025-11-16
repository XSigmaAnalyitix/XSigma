# =============================================================================
# mimalloc Library BUILD Configuration
# =============================================================================
# Microsoft's high-performance memory allocator
# Equivalent to ThirdParty/mimalloc
# =============================================================================

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mimalloc",
    srcs = [
        "src/static.c",
    ],
    hdrs = glob([
        "include/**/*.h",
    ]),
    textual_hdrs = glob([
        "src/**/*.c",
        "src/**/*.h",
    ]),
    copts = select({
        "@platforms//os:windows": [
            "/W0",  # Disable warnings
        ],
        "//conditions:default": [
            "-w",  # Disable warnings
            "-DMI_MALLOC_OVERRIDE=0",
        ],
    }),
    defines = [
        "MI_MALLOC_OVERRIDE=0",
    ],
    includes = [
        "include",
        "src",  # Include src directory for internal includes
    ],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["-lpthread"],
    }),
    linkstatic = True,
)
