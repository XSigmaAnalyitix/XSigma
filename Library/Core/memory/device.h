#pragma once

#include <cstdint>
#include <iosfwd>

#include "common/macros.h"

namespace xsigma
{
enum class device_enum : int16_t
{
    CPU  = 0,
    CUDA = 1,
    HIP  = 2
};

XSIGMA_API std::ostream& operator<<(std::ostream& str, device_enum const& s);

class XSIGMA_VISIBILITY device_option
{
public:
    using int_t = int16_t;

    XSIGMA_API device_option(device_enum type, device_option::int_t index);

    XSIGMA_API device_option(device_enum type, int index);

    XSIGMA_API bool operator==(const xsigma::device_option& rhs) const noexcept;

    XSIGMA_API int_t index() const noexcept;

    XSIGMA_API device_enum type() const noexcept;

private:
    int_t       index_ = -1;
    device_enum type_{};
};

XSIGMA_API std::ostream& operator<<(std::ostream& str, xsigma::device_option const& s);
}  // namespace xsigma
