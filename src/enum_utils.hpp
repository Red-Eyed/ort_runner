#pragma once

#include <magic_enum.hpp>
#include <string>
#include <string_view>

namespace ort_runner {

// Both directions rely on enum members being named to match their CLI string exactly (see
// config.hpp) -- no hand-written string tables needed for any of the CLI-facing enums.

template <typename E>
std::string ToString(E value) {
    return std::string(magic_enum::enum_name(value));
}

// Callers only use this after argparse's .choices() already validated the string, so an
// unmatched value here would indicate a real bug (an enum whose CLI choices() list drifted
// out of sync with its members) rather than bad user input -- letting bad_optional_access
// propagate is appropriate rather than silently guessing a fallback.
template <typename E>
E FromString(std::string_view s) {
    return magic_enum::enum_cast<E>(s).value();
}

}  // namespace ort_runner
