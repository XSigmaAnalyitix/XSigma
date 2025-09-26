#pragma once

  // For export macro
#include "common/macros.h"
#include "xsigma_version_macros.h"  // For version macros

#define __XSIGMA_SOURCE_VERSION__ "xsigma version " __XSIGMA_VERSION__

namespace xsigma
{
class XSIGMA_API version
{
    XSIGMA_DELETE_CLASS(version);

public:
    static const char* GetXSIGMAVersion() { return __XSIGMA_VERSION__; }
    // static int         GetXSIGMAMajorVersion() { return __XSIGMA_MAJOR_VERSION__; }
    // static int         GetXSIGMAMinorVersion() { return __XSIGMA_MINOR_VERSION__; }
    // static int         GetXSIGMABuildVersion() { return __XSIGMA_BUILD_VERSION__; }
    // static const char* GetXSIGMASourceVersion() { return __XSIGMA_SOURCE_VERSION__; }
};
}  // namespace xsigma
extern "C"
{
    XSIGMA_API const char* GetXSIGMAVersion();
}
