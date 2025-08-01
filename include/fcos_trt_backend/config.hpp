#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>


namespace config
{
// ImageNet normalization constants
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

} // namespace config
