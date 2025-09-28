/*
 * XSigma: High-Performance Quantitative Library
 *
 * Original work Copyright 2015 The TensorFlow Authors
 * Modified work Copyright 2025 XSigma Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file contains code modified from TensorFlow (Apache 2.0 licensed)
 * and is part of XSigma, licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * MODIFICATIONS FROM ORIGINAL:
 * - Adapted for XSigma quantitative computing requirements
 * - Added high-performance memory allocation optimizations
 * - Integrated NUMA-aware allocation strategies
 * - Unified platform-specific implementations using conditional compilation
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include "memory/cpu/helper/mem.h"

#include <cpuinfo.h>

#include <cstdint>
#include <limits>
#include <string>

#include "common/configure.h"
#include "common/macros.h"
#include "memory/cpu/allocator.h"

#ifdef _WIN32
// Windows-specific includes
#include <Windows.h>
#include <processthreadsapi.h>
#include <shlwapi.h>

#include <cstdio>
#include <cstdlib>
#else
// Unix/Linux-specific includes
#if defined(__linux__)
#include <sched.h>
#include <sys/sysinfo.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__) || defined(__HAIKU__)
#include <thread>
#endif

#if XSIGMA_ENABLE_NUMA
#include "hwloc.h"  // from @hwloc
#endif

#if XSIGMA_HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif
#endif  // _WIN32

#ifdef XSIGMA_ENABLE_SNAPPY
#include "snappy.h"
#endif

namespace xsigma
{
namespace port
{
// ========== Snappy Compression Functions ==========
bool snappy_compress(const char* input, size_t length, std::string* output)  //NOLINT
{
#ifdef XSIGMA_ENABLE_SNAPPY
    output->resize(snappy::MaxCompressedLength(length));
    size_t outlen;
    snappy::RawCompress(input, length, &(*output)[0], &outlen);
    output->resize(outlen);
    return true;
#else
    // Suppress unused parameter warnings when Snappy is disabled
    (void)input;
    (void)length;
    (void)output;
    return false;
#endif
}

bool snappy_compress_from_io_vec(
    const struct iovec* iov, size_t uncompressed_length, std::string* output)
{
#ifdef XSIGMA_ENABLE_SNAPPY
    output->resize(snappy::MaxCompressedLength(uncompressed_length));
    size_t outlen;

// Platform-specific handling of iovec structure
#ifdef _WIN32
    // Windows requires casting to snappy::iovec
    const snappy::iovec* snappy_iov = reinterpret_cast<const snappy::iovec*>(iov);
    snappy::RawCompressFromIOVec(snappy_iov, uncompressed_length, &(*output)[0], &outlen);
#else
    // Unix/Linux can use iovec directly
    snappy::RawCompressFromIOVec(iov, uncompressed_length, &(*output)[0], &outlen);
#endif

    output->resize(outlen);
    return true;
#else
    // Suppress unused parameter warnings when Snappy is disabled
    (void)iov;
    (void)uncompressed_length;
    (void)output;
    return false;
#endif
}

bool snappy_get_uncompressed_length(const char* input, size_t length, size_t* result)  //NOLINT
{
#ifdef XSIGMA_ENABLE_SNAPPY
    return snappy::GetUncompressedLength(input, length, result);
#else
    // Suppress unused parameter warnings when Snappy is disabled
    (void)input;
    (void)length;
    (void)result;
    return false;
#endif
}

bool snappy_uncompress(const char* input, size_t length, char* output)  //NOLINT
{
#ifdef XSIGMA_ENABLE_SNAPPY
    return snappy::RawUncompress(input, length, output);
#else
    // Suppress unused parameter warnings when Snappy is disabled
    (void)input;
    (void)length;
    (void)output;
    return false;
#endif
}

bool snappy_uncompress_to_io_vec(
    const char* compressed, size_t compressed_length, const struct iovec* iov, size_t iov_cnt)
{
#ifdef XSIGMA_ENABLE_SNAPPY
// Platform-specific handling of iovec structure
#ifdef _WIN32
    // Windows requires casting to snappy::iovec
    const snappy::iovec* snappy_iov = reinterpret_cast<const snappy::iovec*>(iov);
    return snappy::RawUncompressToIOVec(compressed, compressed_length, snappy_iov, iov_cnt);
#else
    // Unix/Linux can use iovec directly
    return snappy::RawUncompressToIOVec(compressed, compressed_length, iov, iov_cnt);
#endif
#else
    // Suppress unused parameter warnings when Snappy is disabled
    (void)compressed;
    (void)compressed_length;
    (void)iov;
    (void)iov_cnt;
    return false;
#endif
}

memory_info GetMemoryInfo()
{
    memory_info mem_info = {INT64_MAX, INT64_MAX};
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex))
    {
        mem_info.free  = statex.ullAvailPhys;
        mem_info.total = statex.ullTotalPhys;
    }
#else
#if defined(__linux__)
    struct sysinfo info;
    int            err = sysinfo(&info);
    if (err == 0)
    {
        mem_info.free  = info.freeram;
        mem_info.total = info.totalram;
    }
#endif
#endif
    return mem_info;
}

memory_bandwidth_info GetMemoryBandwidthInfo()
{
    memory_bandwidth_info membw_info = {INT64_MAX};
    return membw_info;
}
}  // namespace port
}  // namespace xsigma