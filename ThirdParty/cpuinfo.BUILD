# =============================================================================
# cpuinfo Library BUILD Configuration
# =============================================================================
# CPU feature detection library
# Equivalent to ThirdParty/cpuinfo
# =============================================================================

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cpuinfo",
    srcs = glob(
        [
            "src/**/*.c",
            "deps/clog/src/*.c",
        ],
        exclude = [
            # Exclude platform-specific files
            "src/arm/linux/*.c",
            "src/arm/mach/*.c",
            "src/x86/windows/*.c",
            "src/x86/linux/*.c",
            "src/x86/mach/*.c",
        ],
    ) + select({
        "@platforms//os:linux": glob(["src/arm/linux/*.c", "src/x86/linux/*.c"]),
        "@platforms//os:macos": glob(["src/arm/mach/*.c", "src/x86/mach/*.c"]),
        "@platforms//os:windows": glob(["src/x86/windows/*.c"]),
        "//conditions:default": [],
    }),
    hdrs = glob([
        "include/*.h",
        "include/cpuinfo/*.h",
        "src/**/*.h",
        "deps/clog/include/*.h",
    ]),
    copts = select({
        "@platforms//os:windows": [],
        "//conditions:default": ["-w"],  # Suppress warnings for third-party code
    }),
    defines = [
        "CPUINFO_LOG_LEVEL=2",  # Set log level (0=none, 1=error, 2=warning, 3=info, 4=debug)
    ],
    includes = [
        "include",
        "src",
        "deps/clog/include",
    ],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["-lpthread"],
    }),
    linkstatic = True,
)
