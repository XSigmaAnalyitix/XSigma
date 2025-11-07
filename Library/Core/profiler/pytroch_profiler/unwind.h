#pragma once

#include "common/export.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xsigma::unwind
{
// gather current stack, relatively fast.
// gets faster once the cache of program counter locations is warm.
XSIGMA_API std::vector<void*> unwind();

class XSIGMA_VISIBILITY Frame {
    std::string filename;
    std::string funcname;
    uint64_t    lineno;
};

enum class Mode
{
    addr2line,
    fast,
    dladdr
};

// note: symbolize is really slow
// it will launch an addr2line process that has to parse dwarf
// information from the libraries that frames point into.
// Callers should first batch up all the unique void* pointers
// across a number of unwind states and make a single call to
// symbolize.
XSIGMA_API std::vector<Frame> symbolize(const std::vector<void*>& frames, Mode mode);

// returns path to the library, and the offset of the addr inside the library
XSIGMA_API std::optional<std::pair<std::string, uint64_t>> libraryFor(void* addr);

class XSIGMA_VISIBILITY Stats {
    size_t hits        = 0;
    size_t misses      = 0;
    size_t unsupported = 0;
    size_t resets      = 0;
};
Stats stats();

}  // namespace xsigma::unwind
