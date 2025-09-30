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

#include "profiler_report.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "logging/logger.h"

namespace xsigma
{

//=============================================================================
// profiler_report Implementation
//=============================================================================

profiler_report::profiler_report(const xsigma::profiler_session& session) : session_(session) {}

std::string profiler_report::generate_console_report() const
{
    std::stringstream ss;

    ss << generate_header_section();
    ss << generate_summary_section();

    if (include_hierarchical_data_)
    {
        ss << generate_hierarchical_section();
    }

    ss << generate_timing_section();
    ss << generate_memory_section();
    ss << generate_statistical_section();

    if (include_thread_info_)
    {
        ss << generate_thread_section();
    }

    return ss.str();
}

std::string profiler_report::generate_json_report() const
{
    std::stringstream ss;

    ss << "{\n";
    ss << "  \"profiler_report\": {\n";
    ss << "    \"header\": " << generate_json_object(generate_header_section()) << ",\n";
    ss << "    \"summary\": " << generate_json_object(generate_summary_section()) << ",\n";

    if (include_hierarchical_data_)
    {
        ss << "    \"hierarchical_data\": " << generate_json_object(generate_hierarchical_section())
           << ",\n";
    }

    ss << "    \"timing\": " << generate_json_object(generate_timing_section()) << ",\n";
    ss << "    \"memory\": " << generate_json_object(generate_memory_section()) << ",\n";
    ss << "    \"statistics\": " << generate_json_object(generate_statistical_section());

    if (include_thread_info_)
    {
        ss << ",\n    \"threads\": " << generate_json_object(generate_thread_section());
    }

    ss << "\n  }\n";
    ss << "}\n";

    return ss.str();
}

std::string profiler_report::generate_csv_report() const
{
    std::stringstream ss;

    // CSV Header
    ss << generate_csv_header() << "\n";

    // Process hierarchical data into CSV rows
    std::vector<std::string> rows;
    if (include_hierarchical_data_ && session_.get_root_scope())
    {
        process_scope_data_csv_recursive(*session_.get_root_scope(), rows);
    }

    for (const auto& row : rows)
    {
        ss << row << "\n";
    }

    return ss.str();
}

std::string profiler_report::generate_xml_report() const
{
    std::stringstream ss;

    ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    ss << "<profiler_report>\n";

    ss << "  <header>\n" << generate_header_section() << "  </header>\n";
    ss << "  <summary>\n" << generate_summary_section() << "  </summary>\n";

    if (include_hierarchical_data_)
    {
        ss << "  <hierarchical_data>\n"
           << generate_hierarchical_section() << "  </hierarchical_data>\n";
    }

    ss << "  <timing>\n" << generate_timing_section() << "  </timing>\n";
    ss << "  <memory>\n" << generate_memory_section() << "  </memory>\n";
    ss << "  <statistics>\n" << generate_statistical_section() << "  </statistics>\n";

    if (include_thread_info_)
    {
        ss << "  <threads>\n" << generate_thread_section() << "  </threads>\n";
    }

    ss << "</profiler_report>\n";

    return ss.str();
}

bool profiler_report::export_to_file(
    const std::string& filename, xsigma::profiler_options::output_format_enum format) const
{
    std::string content;

    switch (format)
    {
    case xsigma::profiler_options::output_format_enum::CONSOLE:
    case xsigma::profiler_options::output_format_enum::FILE:
        content = generate_console_report();
        break;
    case xsigma::profiler_options::output_format_enum::JSON:
        content = generate_json_report();
        break;
    case xsigma::profiler_options::output_format_enum::CSV:
        content = generate_csv_report();
        break;
    case xsigma::profiler_options::output_format_enum::STRUCTURED:
        content = generate_xml_report();
        break;
    default:
        content = generate_console_report();
        break;
    }

    std::ofstream file(filename);
    if (!file.is_open())
    {
        return false;
    }

    file << content;
    file.close();
    return true;
}

bool profiler_report::export_console_report(const std::string& filename) const
{
    return export_to_file(filename, xsigma::profiler_options::output_format_enum::CONSOLE);
}

bool profiler_report::export_json_report(const std::string& filename) const
{
    return export_to_file(filename, xsigma::profiler_options::output_format_enum::JSON);
}

bool profiler_report::export_csv_report(const std::string& filename) const
{
    return export_to_file(filename, xsigma::profiler_options::output_format_enum::CSV);
}

bool profiler_report::export_xml_report(const std::string& filename) const
{
    return export_to_file(filename, xsigma::profiler_options::output_format_enum::STRUCTURED);
}

void profiler_report::print_summary() const
{
    XSIGMA_LOG_INFO(generate_header_section());
    XSIGMA_LOG_INFO(generate_summary_section());
}

void profiler_report::print_detailed_report() const
{
    XSIGMA_LOG_INFO(generate_console_report());
}

void profiler_report::print_memory_report() const
{
    XSIGMA_LOG_INFO(generate_header_section());
    XSIGMA_LOG_INFO(generate_memory_section());
}

void profiler_report::print_timing_report() const
{
    XSIGMA_LOG_INFO(generate_header_section());
    XSIGMA_LOG_INFO(generate_timing_section());
}

void profiler_report::print_statistical_report() const
{
    XSIGMA_LOG_INFO(generate_header_section());
    XSIGMA_LOG_INFO(generate_statistical_section());
}

// Helper method implementations
std::string profiler_report::format_duration(double duration_ns) const
{
    if (time_unit_ == "ms")
    {
        return std::to_string(duration_ns / 1000000.0) + " ms";
    }
    else if (time_unit_ == "us")
    {
        return std::to_string(duration_ns / 1000.0) + " us";
    }
    else
    {
        return std::to_string(duration_ns) + " ns";
    }
}

std::string profiler_report::format_memory_size(size_t bytes) const
{
    if (memory_unit_ == "MB")
    {
        return std::to_string(bytes / (1024.0 * 1024.0)) + " MB";
    }
    else if (memory_unit_ == "KB")
    {
        return std::to_string(bytes / 1024.0) + " KB";
    }
    else
    {
        return std::to_string(bytes) + " bytes";
    }
}

std::string profiler_report::format_percentage(double value) const
{
    return std::to_string(value * 100.0) + "%";
}

std::string profiler_report::format_thread_id(const std::thread::id& thread_id) const
{
    std::stringstream ss;
    ss << thread_id;
    return ss.str();
}

// Section generators - simplified implementations
std::string profiler_report::generate_header_section() const
{
    return "=== Enhanced Profiler Report ===\n";
}

std::string profiler_report::generate_summary_section() const
{
    return "=== Summary ===\n";
}

std::string profiler_report::generate_timing_section() const
{
    return "=== Timing Analysis ===\n";
}

std::string profiler_report::generate_memory_section() const
{
    return "=== Memory Analysis ===\n";
}

std::string profiler_report::generate_hierarchical_section() const
{
    return "=== Hierarchical Analysis ===\n";
}

std::string profiler_report::generate_statistical_section() const
{
    return "=== Statistical Analysis ===\n";
}

std::string profiler_report::generate_thread_section() const
{
    return "=== Thread Analysis ===\n";
}

// JSON helpers
std::string profiler_report::escape_json_string(const std::string& str) const
{
    return "\"" + str + "\"";
}

std::string profiler_report::generate_json_object(const std::string& content) const
{
    return R"({"content": ")" + content + R"("})";
}

std::string profiler_report::generate_json_array(const std::vector<std::string>& /*items*/) const
{
    return "[]";
}

// CSV helpers
std::string profiler_report::escape_csv_field(const std::string& field) const
{
    return "\"" + field + "\"";
}

std::string profiler_report::generate_csv_header() const
{
    return "Name,Duration,Memory,Thread";
}

std::string profiler_report::generate_csv_row(const std::vector<std::string>& fields) const
{
    std::string result;
    for (size_t i = 0; i < fields.size(); ++i)
    {
        if (i > 0)
            result += ",";
        result += escape_csv_field(fields[i]);
    }
    return result;
}

// XML helpers
std::string profiler_report::escape_xml_string(const std::string& str) const
{
    return str;  // Simplified
}

std::string profiler_report::generate_xml_element(
    const std::string& tag, const std::string& content) const
{
    return "<" + tag + ">" + content + "</" + tag + ">";
}

std::string profiler_report::generate_xml_attribute(
    const std::string& name, const std::string& value) const
{
    return name + "=\"" + value + "\"";
}

// Hierarchical data processing - simplified implementations
void profiler_report::process_scope_data_recursive(
    const xsigma::profiler_scope_data& scope, std::stringstream& ss, int indent) const
{
    // Simplified implementation
    for (int i = 0; i < indent; ++i)
        ss << "  ";
    ss << scope.name_ << "\n";
}

void profiler_report::process_scope_data_json_recursive(
    const xsigma::profiler_scope_data& scope, std::stringstream& ss, int /*indent*/) const
{
    // Simplified implementation
    ss << R"({"name": ")" << scope.name_ << R"("})";
}

void profiler_report::process_scope_data_csv_recursive(
    const xsigma::profiler_scope_data& scope, std::vector<std::string>& rows) const
{
    // Simplified implementation
    rows.push_back(generate_csv_row({scope.name_, "0", "0", "main"}));
}

//=============================================================================
// profiler_report_builder Implementation
//=============================================================================

profiler_report_builder::profiler_report_builder(const xsigma::profiler_session& session)
    : session_(session)
{
}

std::unique_ptr<xsigma::profiler_report> profiler_report_builder::build() const
{
    auto report = std::make_unique<xsigma::profiler_report>(session_);
    report->set_precision(precision_);
    report->set_time_unit(time_unit_);
    report->set_memory_unit(memory_unit_);
    report->set_include_thread_info(include_thread_info_);
    report->set_include_hierarchical_data(include_hierarchical_data_);
    return report;
}

}  // namespace xsigma
