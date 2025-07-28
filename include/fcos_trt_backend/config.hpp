#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace config
{
// ImageNet normalization constants (same as torchvision)
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

// COCO class names (matching your Python script)
const std::vector<std::string> COCO_CLASS_NAMES = {
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// COCO colors for visualization (BGR format for OpenCV)
constexpr std::array<std::array<uint8_t, 3>, 80> COCO_COLORS = {{
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255},
    {0, 255, 255}, {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 0},
    {128, 0, 128}, {0, 128, 128}, {192, 192, 192}, {128, 128, 128}, {255, 165, 0},
    {255, 20, 147}, {0, 191, 255}, {255, 69, 0}, {50, 205, 50}, {138, 43, 226},
    {220, 20, 60}, {255, 215, 0}, {30, 144, 255}, {255, 105, 180}, {34, 139, 34},
    {255, 140, 0}, {75, 0, 130}, {255, 192, 203}, {106, 90, 205}, {0, 250, 154},
    {219, 112, 147}, {255, 20, 147}, {72, 61, 139}, {205, 92, 92}, {240, 230, 140},
    {255, 182, 193}, {144, 238, 144}, {221, 160, 221}, {250, 128, 114}, {255, 228, 181},
    {176, 196, 222}, {255, 160, 122}, {32, 178, 170}, {135, 206, 250}, {119, 136, 153},
    {255, 99, 71}, {64, 224, 208}, {205, 133, 63}, {210, 180, 140}, {102, 205, 170},
    {250, 240, 230}, {255, 218, 185}, {238, 232, 170}, {188, 143, 143}, {255, 239, 213},
    {255, 222, 173}, {245, 245, 220}, {255, 228, 196}, {255, 235, 205}, {255, 240, 245},
    {240, 248, 255}, {230, 230, 250}, {255, 250, 240}, {255, 245, 238}, {245, 255, 250},
    {112, 128, 144}, {47, 79, 79}, {105, 105, 105}, {169, 169, 169}, {128, 128, 128},
    {192, 192, 192}, {211, 211, 211}, {220, 220, 220}, {245, 245, 245}, {255, 255, 255},
    {0, 0, 0}, {25, 25, 112}, {100, 149, 237}, {123, 104, 238}, {106, 90, 205}
}};

} // namespace config
