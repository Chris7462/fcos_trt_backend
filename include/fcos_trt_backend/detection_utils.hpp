#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <string>
#include <unordered_map>

// OpenCV includes
#include <opencv2/core.hpp>

// Local includes
#include "fcos_trt_backend/fcos_types.hpp"


namespace fcos_trt_backend
{

namespace utils
{

// Color structure for RGB values
struct Color {
  int r, g, b;
  Color(int red = 0, int green = 0, int blue = 0) : r(red), g(green), b(blue) {}

  // Convert to OpenCV Scalar (BGR format)
  cv::Scalar toScalar() const {
    return cv::Scalar(b, g, r); // OpenCV uses BGR, not RGB
  }
};

// Get COCO category names mapping - initialize once
const std::unordered_map<int, std::string>& get_coco_names();

// COCO color mapping - initialize once
const std::unordered_map<int, Color>& get_coco_colors();

// Get class name from COCO category ID
std::string get_class_name(int coco_id);

// Get class-specific color
Color get_class_color(int label_id);

// Utility method to print results
void print_detection_results(const Detections& results, size_t max_detections = 20);

// Visualization method to plot detections on image
cv::Mat plot_detections(
  const cv::Mat & image,
  const Detections& detections,
  float confidence_threshold = 0.5f);

std::vector<int> apply_nms(
  const std::vector<cv::Rect2f>& boxes,
  const std::vector<float>& scores,
  const std::vector<int>& labels,
  float nms_threshold);

cv::Rect2f clip_box_to_image(const cv::Rect2f& box, int image_height, int image_width);

std::vector<int> topk_indices(const std::vector<float>& scores, int k);

} // namespace utils

} // namespace fcos_trt_backend
