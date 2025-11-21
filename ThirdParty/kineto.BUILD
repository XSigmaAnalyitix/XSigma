# =============================================================================
# Kineto Library BUILD Configuration
# =============================================================================
# Kineto profiling library (CPU-only build)
# Equivalent to ThirdParty/kineto/libkineto
# =============================================================================

package(default_visibility = ["//visibility:public"])

# Kineto compiler flags
KINETO_COPTS = [
    "-fexceptions",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-private-field",
    "-Wno-unused-variable",
    "-Wno-unused-parameter",
    "-DKINETO_NAMESPACE=libkineto",
    "-DLIBKINETO_NOCUPTI",
    "-DLIBKINETO_NOROCTRACER",
] + select({
    "@platforms//os:windows": [],
    "//conditions:default": ["-w"],  # Suppress warnings for third-party code
})

cc_library(
    name = "kineto",
    srcs = glob(
        [
            "libkineto/src/*.cpp",
        ],
        exclude = [
            # Exclude CUDA/ROCm specific files
            "libkineto/src/CuptiActivity.cpp",
            "libkineto/src/CuptiActivityApi.cpp",
            "libkineto/src/CuptiCallbackApi.cpp",
            "libkineto/src/CuptiEventApi.cpp",
            "libkineto/src/CuptiMetricApi.cpp",
            "libkineto/src/CuptiRangeProfiler.cpp",
            "libkineto/src/CuptiRangeProfilerApi.cpp",
            "libkineto/src/CuptiRangeProfilerConfig.cpp",
            "libkineto/src/CuptiNvPerfMetric.cpp",
            "libkineto/src/EventProfiler.cpp",
            "libkineto/src/EventProfilerController.cpp",
            "libkineto/src/KernelRegistry.cpp",
            "libkineto/src/WeakSymbols.cpp",
            "libkineto/src/cupti_strings.cpp",
            "libkineto/src/RocprofActivityApi.cpp",
            "libkineto/src/RocprofLogger.cpp",
            "libkineto/src/RoctracerActivityApi.cpp",
            "libkineto/src/RoctracerLogger.cpp",
            "libkineto/src/RocLogger.cpp",
            # Exclude plugin files
            "libkineto/src/plugin/**/*.cpp",
        ],
    ),
    hdrs = glob([
        "libkineto/include/*.h",
        "libkineto/include/**/*.h",
        "libkineto/src/*.h",
    ]),
    copts = KINETO_COPTS,
    includes = [
        "libkineto",
        "libkineto/include",
        "libkineto/src",
    ],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": ["-lpthread"],
        "//conditions:default": ["-lpthread", "-ldl"],
    }),
    linkstatic = True,
    deps = [
        "@fmt//:fmt",
    ],
)

