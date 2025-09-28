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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "experimental/profiler/env_var.h"

#include <algorithm>
#include <algorithm>  // for transform
#include <cctype>
#include <cctype>  // for tolower
#include <cstdlib>
#include <cstdlib>  // for getenv
#include <memory>   // for _Simple_types
#include <ostream>  // for operator<<
#include <string>   // for char_traits, allocator, operator==, string, basic_string

#include "util/exception.h"    // for check_msg_impl, XSIGMA_CHECK
#include "util/logging.h"      // for LogMessageFatal, LOG, _XSIGMA_LOG_FATAL
#include "util/strcat.h"       // for StrCat
#include "util/string_util.h"  // for safe_strto64, safe_strtof, split_string

namespace xsigma
{
std::string ascii_str_to_lower(std::string& result)
{
    std::transform(
        result.begin(),
        result.end(),
        result.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return result;
}

bool read_bool_from_env_var(std::string_view env_var_name, bool default_val, bool* value)
{
    *value              = default_val;
    const char* env_val = std::getenv(env_var_name.data());
    if (env_val == nullptr)
    {
        return false;
    }
    std::string str_value = env_val;

    str_value = ascii_str_to_lower(str_value);

    if (str_value == "0" || str_value == "false")
    {
        *value = false;
        return true;
    }
    else if (str_value == "1" || str_value == "true")
    {
        *value = true;
        return true;
    }

    LOG(FATAL) << strings::StrCat(
        "Failed to parse the env-var ${",
        env_var_name,
        "} into bool: ",
        str_value,
        ". Use the default value: ",
        default_val);
    return false;
}

bool read_int64_from_env_var(std::string_view env_var_name, int64_t default_val, int64_t* value)
{
    *value                     = default_val;
    const char* tf_env_var_val = std::getenv(env_var_name.data());
    if (tf_env_var_val == nullptr)
    {
        return true;
    }
    if (numbers::safe_strto64(tf_env_var_val, value))
    {
        return true;
    }
    LOG(FATAL) << "InvalidArgument "
               << strings::StrCat(
                      "Failed to parse the env-var ${",
                      env_var_name,
                      "} into int64: ",
                      tf_env_var_val,
                      ". Use the default value: ",
                      default_val);

    return false;
}

bool read_float_from_env_var(std::string_view env_var_name, float default_val, float* value)
{
    *value                     = default_val;
    const char* tf_env_var_val = std::getenv(env_var_name.data());
    if (tf_env_var_val == nullptr)
    {
        return true;
    }
    if (numbers::safe_strtof(tf_env_var_val, value))
    {
        return true;
    }
    LOG(FATAL) << "InvalidArgument "
               << strings::StrCat(
                      "Failed to parse the env-var ${",
                      env_var_name,
                      "} into float: ",
                      tf_env_var_val,
                      ". Use the default value: ",
                      default_val);
    return false;
}

bool read_string_from_env_var(
    std::string_view env_var_name, std::string_view default_val, std::string& value)
{
    const char* tf_env_var_val = std::getenv(env_var_name.data());
    if (tf_env_var_val != nullptr)
    {
        value = tf_env_var_val;
    }
    else
    {
        value = std::string(default_val);
    }
    return true;
}

bool read_strings_from_env_var(
    std::string_view env_var_name, std::string_view default_val, std::vector<std::string>& value)
{
    std::string str_val;
    XSIGMA_CHECK(read_string_from_env_var(env_var_name, default_val, str_val));

    value = split_string(str_val, {','});

    return true;
}

}  // namespace xsigma
