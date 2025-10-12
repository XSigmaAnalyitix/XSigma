#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "logging/back_trace.h"

#include <fmt/format.h>

#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "common/pointer.h"
#include "util/string_util.h"

// Platform-specific includes
#ifdef _WIN32
#include <dbghelp.h>
#include <windows.h>
#pragma comment(lib, "Dbghelp.lib")
#define SUPPORTS_BACKTRACE 1
#elif defined(__unix__) || defined(__APPLE__)
#include <execinfo.h>
#define SUPPORTS_BACKTRACE 1
#else
#define SUPPORTS_BACKTRACE 0
#endif

#if !defined(_WIN32)
// Unix/Linux/macOS-specific helper functions
namespace xsigma
{
namespace
{
struct FrameInformation
{
    /// If available, the demangled name of the function at this frame, else
    /// whatever (possibly mangled) name we got from `backtrace()`.
    std::string function_name;
    /// This is a number in hexadecimal form (e.g. "0xdead") representing the
    /// offset into the function's machine code at which the function's body
    /// starts, i.e. skipping the "prologue" that handles stack manipulation and
    /// other calling convention things.
    std::string offset_into_function;
    /// NOTE: In debugger parlance, the "object file" refers to the ELF file that
    /// the symbol originates from, i.e. either an executable or a library.
    std::string object_file;
};

bool is_python_frame(const FrameInformation& frame)
{
    return frame.object_file == "python" || frame.object_file == "python3" ||
           (frame.object_file.find("libpython") != std::string::npos);
}

std::optional<FrameInformation> parse_frame_information(const std::string& frame_string)
{
    FrameInformation frame;

    // This is the function name in the CXX ABI mangled format, e.g. something
    // like _Z1gv. Reference:
    // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling
    std::string mangled_function_name;

#if defined(__GLIBCXX__)
    // In GLIBCXX, `frame_string` follows the pattern
    // `<object-file>(<mangled-function-name>+<offset-into-function>)
    // [<return-address>]`

    auto function_name_start = frame_string.find("(");
    if (function_name_start == std::string::npos)
    {
        return std::nullopt;
    }
    function_name_start += 1;

    auto offset_start = frame_string.find('+', function_name_start);
    if (offset_start == std::string::npos)
    {
        return std::nullopt;
    }
    offset_start += 1;

    const auto offset_end = frame_string.find(')', offset_start);
    if (offset_end == std::string::npos)
    {
        return std::nullopt;
    }

    frame.object_file          = frame_string.substr(0, function_name_start - 1);
    frame.offset_into_function = frame_string.substr(offset_start, offset_end - offset_start);

    // NOTE: We don't need to parse the return address because
    // we already have it from the call to `backtrace()`.

    mangled_function_name =
        frame_string.substr(function_name_start, (offset_start - 1) - function_name_start);
#elif defined(_LIBCPP_VERSION)
    // In LIBCXX, The pattern is
    // `<frame number> <object-file> <return-address> <mangled-function-name> +
    // <offset-into-function>`
    std::string        skip;
    std::istringstream input_stream(frame_string);
    // operator>>() does not fail -- if the input stream is corrupted, the
    // strings will simply be empty.
    input_stream >> skip >> frame.object_file >> skip >> mangled_function_name >> skip >>
        frame.offset_into_function;
#else
    // Unknown standard library, backtraces may have incomplete verbose information
    return std::nullopt;
#endif  // defined(__GLIBCXX__)

    // Some system-level functions don't have sufficient verbose information, so
    // we'll display them as "<unknown function>". They'll still have a return
    // address and other pieces of information.
    if (mangled_function_name.empty())
    {
        frame.function_name = "<unknown function>";
        return frame;
    }

    frame.function_name = demangle(mangled_function_name.c_str());
    return frame;
}
}  // anonymous namespace
}  // namespace xsigma
#endif  // !defined(_WIN32)

namespace xsigma
{
// ============================================================================
// Public API Implementation
// ============================================================================

std::string back_trace::print(
    XSIGMA_UNUSED size_t frames_to_skip,
    XSIGMA_UNUSED size_t maximum_number_of_frames,
    XSIGMA_UNUSED bool   skip_python_frames)
{
    backtrace_options options;
    options.frames_to_skip           = frames_to_skip;
    options.maximum_number_of_frames = maximum_number_of_frames;
    options.skip_python_frames       = skip_python_frames;
    return print(options);
}

std::string back_trace::print(const backtrace_options& options)
{
    auto frames = capture(options);
    return format(frames, options);
}

std::vector<stack_frame> back_trace::capture(const backtrace_options& options)
{
    std::vector<stack_frame> result;

#if SUPPORTS_BACKTRACE
    // Skip this frame (capture) plus user-requested frames
    size_t  frames_to_skip = options.frames_to_skip + 1ULL;//NOLINT

#ifdef _WIN32
    // Windows implementation using CaptureStackBackTrace
    const size_t       max_frames = frames_to_skip + options.maximum_number_of_frames;
    std::vector<void*> callstack(max_frames, nullptr);

    // Capture stack frames
    USHORT const captured = CaptureStackBackTrace(
        static_cast<DWORD>(frames_to_skip),
        static_cast<DWORD>(options.maximum_number_of_frames),
        callstack.data(),
        nullptr);

    callstack.resize(captured);

    // Initialize symbol handler (thread-safe)
    static bool symbol_handler_initialized = false;
    if (!symbol_handler_initialized)
    {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitialize(GetCurrentProcess(), nullptr, TRUE);
        symbol_handler_initialized = true;
    }

    // Resolve symbols
    HANDLE            process     = GetCurrentProcess();
    const size_t      buffer_size = sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR);
    std::vector<char> buffer(buffer_size);
    auto*             symbol = reinterpret_cast<SYMBOL_INFO*>(buffer.data());

    for (size_t i = 0; i < callstack.size(); ++i)
    {
        stack_frame frame;
        frame.frame_number   = i;
        frame.return_address = callstack[i];

        // Get symbol information
        symbol->MaxNameLen   = MAX_SYM_NAME;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

        DWORD64 displacement = 0;
        if (SymFromAddr(process, reinterpret_cast<DWORD64>(callstack[i]), &displacement, symbol) !=
            0)
        {
            frame.function_name        = symbol->Name;
            frame.offset_into_function = fmt::format("0x{:x}", displacement);
        }
        else
        {
            frame.function_name =
                fmt::format("<unknown> [0x{:x}]", reinterpret_cast<uintptr_t>(callstack[i]));
        }

        // Get module information
        IMAGEHLP_MODULE64 module_info;
        module_info.SizeOfStruct = sizeof(IMAGEHLP_MODULE64);
        if (SymGetModuleInfo64(process, reinterpret_cast<DWORD64>(callstack[i]), &module_info) != 0)
        {
            frame.object_file = module_info.ModuleName;
        }

        // Get source line information if requested
        if (options.include_source_info)
        {
            IMAGEHLP_LINE64 line_info;
            line_info.SizeOfStruct  = sizeof(IMAGEHLP_LINE64);
            DWORD line_displacement = 0;
            if (SymGetLineFromAddr64(
                    process,
                    reinterpret_cast<DWORD64>(callstack[i]),
                    &line_displacement,
                    &line_info) != 0)
            {
                frame.source_file = line_info.FileName;
                frame.source_line = static_cast<int>(line_info.LineNumber);
            }
        }

        result.push_back(frame);
    }

#else   // Unix/Linux/macOS implementation
    std::vector<void*> callstack(frames_to_skip + options.maximum_number_of_frames, nullptr);

    // Capture raw stack addresses
    auto number_of_frames = ::backtrace(callstack.data(), static_cast<int>(callstack.size()));

    // Skip requested frames
    for (; frames_to_skip > 0 && number_of_frames > 0; --frames_to_skip, --number_of_frames)
    {
        callstack.erase(callstack.begin());
    }

    callstack.resize(static_cast<size_t>(number_of_frames));

    // Get symbol information
    std::unique_ptr<char*, std::function<void(char**)> > const raw_symbols(
        ::backtrace_symbols(callstack.data(), static_cast<int>(callstack.size())), free);

    if (!raw_symbols)
    {
        return result;  // Failed to get symbols
    }
    // cppcheck-suppress arithOperationsOnVoidPointer
    const std::vector<std::string> symbols(raw_symbols.get(), raw_symbols.get() + callstack.size());

    // Parse each frame
    bool has_skipped_python_frames = false;
    for (size_t frame_number = 0; frame_number < callstack.size(); ++frame_number)
    {
        const auto frame_info = parse_frame_information(symbols[frame_number]);

        // Skip Python frames if requested
        if (options.skip_python_frames && frame_info && is_python_frame(*frame_info))
        {
            if (!has_skipped_python_frames)
            {
                stack_frame python_marker;
                python_marker.function_name = "<omitting python frames>";
                python_marker.frame_number  = frame_number;
                result.push_back(python_marker);
                has_skipped_python_frames = true;
            }
            continue;
        }

        stack_frame frame;
        frame.frame_number   = frame_number;
        frame.return_address = callstack[frame_number];

        if (frame_info)
        {
            frame.function_name        = frame_info->function_name;
            frame.object_file          = frame_info->object_file;
            frame.offset_into_function = frame_info->offset_into_function;
        }
        else
        {
            // Fallback: use raw symbol string
            frame.function_name = symbols[frame_number];
        }

        result.push_back(frame);
    }
#endif  // _WIN32

#endif  // SUPPORTS_BACKTRACE

    return result;
}

std::string back_trace::format(
    const std::vector<stack_frame>& frames, const backtrace_options& options)
{
    if (frames.empty())
    {
        return "No stack trace available\n";
    }

    std::string result;

    if (options.compact_format)
    {
        // Compact format: "func1 -> func2 -> func3"
        for (size_t i = 0; i < frames.size(); ++i)
        {
            if (i > 0)
            {
                result += " -> ";
            }
            result += frames[i].function_name;
        }
        result += "\n";
    }
    else
    {
        // Detailed format
        for (const auto& frame : frames)
        {
            std::string line =
                fmt::format("frame #{}: {}", frame.frame_number, frame.function_name);

            if (options.include_offsets && !frame.offset_into_function.empty())
            {
                line += fmt::format(" + {}", frame.offset_into_function);
            }

            if (options.include_addresses || !frame.object_file.empty())
            {
                line += " (";
                if (options.include_addresses && (frame.return_address != nullptr))
                {
                    line += fmt::format("{}", frame.return_address);
                }
                if (!frame.object_file.empty())
                {
                    if (options.include_addresses && (frame.return_address != nullptr))
                    {
                        line += " in ";
                    }
                    line += frame.object_file;
                }
                line += ")";
            }

            if (options.include_source_info && frame.source_line >= 0)
            {
                line += fmt::format(" at {}:{}", frame.source_file, frame.source_line);
            }

            result += line + "\n";
        }
    }

    return result;
}

std::string back_trace::compact(size_t max_frames)
{
    backtrace_options options;
    options.frames_to_skip           = 1;  // Skip compact() itself
    options.maximum_number_of_frames = max_frames;
    options.compact_format           = true;
    options.include_addresses        = false;
    options.include_offsets          = false;

    return print(options);
}

bool back_trace::is_supported()
{
#if SUPPORTS_BACKTRACE
    return true;
#else
    return false;
#endif
}

void back_trace::set_stack_trace_on_error(int enable)
{
    // Placeholder for future implementation
    // Could integrate with signal handlers or exception hooks
    (void)enable;
}

}  // namespace xsigma
#ifdef __clang__
#pragma clang diagnostic pop
#endif
