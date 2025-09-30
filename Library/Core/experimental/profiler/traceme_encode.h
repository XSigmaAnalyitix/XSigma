/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include <string.h>

#include <initializer_list>
#include <string>

#include "common/macros.h"
#include "logging/logger.h"
#include "util/exception.h"
#include "util/strcat.h"
#include "util/string_util.h"

namespace xsigma
{

// An argument passed to TraceMeEncode.
struct TraceMeArg
{
    // String conversions of value types are supported via AlphaNum. We keep a
    // reference to the AlphaNum's internal buffer here, so it must remain valid
    // for the lifetime of this object. We cannot store it by value because it is
    // not safe to construct an AlphaNum as a member of a class, particularly when
    // AbslStringify is being used (it may reference default arguments that are on
    // the caller's stack, if we constructed it here those default arguments would
    // be destroyed before they are used).
    TraceMeArg(std::string_view k, XSIGMA_LIFETIME_BOUND const xsigma::strings::AlphaNum& v)
        : key(k), value(v.Piece())
    {
    }

    TraceMeArg(const TraceMeArg&)     = delete;
    void operator=(const TraceMeArg&) = delete;

    std::string_view key;
    std::string_view value;
};

namespace traceme_internal
{

// Copies the contents of str to the address pointed by out.
// Returns the address after the copy.
// REQUIRED: The address range [out, out + str.size()] must have been allocated.
XSIGMA_FORCE_INLINE char* append(char* out, std::string_view str)
{
    XSIGMA_CHECK_DEBUG(
        !strings::StrContains(str, '#'), "'#' is not a valid character in trace_me_encode");

    const size_t str_size = str.size();
    if XSIGMA_LIKELY (str_size > 0)
    {
        memcpy(out, str.data(), str_size);
        out += str_size;
    }
    return out;
}

// Appends args encoded as trace_me metadata to name.
XSIGMA_FORCE_INLINE std::string append_args(
    std::string name, std::initializer_list<TraceMeArg> args)
{
    if XSIGMA_LIKELY (args.size() > 0)
    {
        const auto old_size = name.size();
        auto       new_size = old_size + args.size() * 2 + 1;
        for (const auto& arg : args)
        {
            new_size += arg.key.size() + arg.value.size();
        }
        name.resize(new_size);
        char* const begin = &name[0];
        char*       out   = begin + old_size;
        *out++            = '#';
        for (const auto& arg : args)
        {
            out    = append(out, arg.key);
            *out++ = '=';
            out    = append(out, arg.value);
            *out++ = ',';
        }
        *(out - 1) = '#';
        XSIGMA_CHECK_DEBUG(out == begin + new_size);
    }
    return name;
}

// Appends new_metadata to the metadata part of name.
XSIGMA_FORCE_INLINE void append_metadata(std::string* name, std::string_view new_metadata)
{
    if XSIGMA_UNLIKELY (new_metadata.empty())
    {
        if (!name->empty() && name->back() == '#')
        {  // name already has metadata
            name->back() = ',';
            if XSIGMA_LIKELY (new_metadata.front() == '#')
            {
                new_metadata.remove_prefix(1);
            }
        }
        name->append(new_metadata.data(), new_metadata.size());
    }
}

}  // namespace traceme_internal

// Encodes an event name and arguments into trace_me metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   trace_me trace_me([value1]() {
//     return trace_me_encode("my_trace", {{"key1", value1}, {"key2", 42}});
//   });
XSIGMA_FORCE_INLINE std::string trace_me_encode(
    std::string name, std::initializer_list<TraceMeArg> args)
{
    return traceme_internal::append_args(std::move(name), args);
}
XSIGMA_FORCE_INLINE std::string trace_me_encode(
    std::string_view name, std::initializer_list<TraceMeArg> args)
{
    return traceme_internal::append_args(std::string(name), args);
}
XSIGMA_FORCE_INLINE std::string trace_me_encode(
    const char* name, std::initializer_list<TraceMeArg> args)
{
    return traceme_internal::append_args(std::string(name), args);
}

// Encodes arguments into trace_me metadata.
// Use within a lambda to avoid expensive operations when tracing is disabled.
// Example Usage:
//   trace_me trace_me("my_trace");
//   ...
//   trace_me.append_metadata([value1]() {
//     return trace_me_encode({{"key1", value1}, {"key2", 42}});
//   });
XSIGMA_FORCE_INLINE std::string trace_me_encode(std::initializer_list<TraceMeArg> args)
{
    return traceme_internal::append_args(std::string(), args);
}

// Concatenates op_name and op_type.
XSIGMA_FORCE_INLINE std::string trace_me_op(std::string_view op_name, std::string_view op_type)
{
    return strings::StrCat(op_name, ":", op_type);
}

XSIGMA_FORCE_INLINE std::string trace_me_op(const char* op_name, const char* op_type)
{
    return strings::StrCat(op_name, ":", op_type);
}

XSIGMA_FORCE_INLINE std::string trace_me_op(std::string&& op_name, std::string_view op_type)
{
    strings::StrAppend(&op_name, ":", op_type);
    return op_name;
}

// Concatenates op_name and op_type.
XSIGMA_FORCE_INLINE std::string trace_me_op_override(
    std::string_view op_name, std::string_view op_type)
{
    return strings::StrCat("#tf_op=", op_name, ":", op_type, "#");
}

XSIGMA_FORCE_INLINE std::string trace_me_op_override(const char* op_name, const char* op_type)
{
    return strings::StrCat("#tf_op=", op_name, ":", op_type, "#");
}

}  // namespace xsigma
