/*
 * Some parts of this implementation were inspired by code from VTK
 * (The Visualization Toolkit), distributed under a BSD-style license.
 * See LICENSE for details.
 */
#ifndef __xsigma_configure_h__
#define __xsigma_configure_h__

#include "xsigma_features.h"
#include "xsigma_threads.h"
#include "xsigma_version_macros.h"

#if defined(__linux__) && defined(XSIGMA_ENABLE_NUMA)
#define XSIGMA_NUMA_ENABLED
#endif

#if defined(XSIGMA_AVX512) || defined(XSIGMA_AVX2) || defined(XSIGMA_AVX) || defined(XSIGMA_SSE)
#define XSIGMA_VECTORIZED
#endif

// Compression configuration
#ifdef XSIGMA_ENABLE_COMPRESSION
#ifdef XSIGMA_COMPRESSION_TYPE_SNAPPY
#define XSIGMA_COMPRESSION_TYPE_SNAPPY
#endif
#endif

#endif  // __xsigma_configure_h__
