#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <string>
#include <unordered_map>

// OpenCV includes
#include <opencv2/core.hpp>

// Local includes
#include "fcos_trt_backend/detection_types.hpp"


namespace fcos_trt_backend
{

namespace utils
{

// COCO class names mapping with correct category IDs (with gaps)
const std::unordered_map<int, std::string> COCO_CATEGORY_NAMES = {
  {0, "__background__"},
  {1, "person"}, {2, "bicycle"}, {3, "car"}, {4, "motorcycle"}, {5, "airplane"},
  {6, "bus"}, {7, "train"}, {8, "truck"}, {9, "boat"}, {10, "traffic light"},
  {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
  {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"}, {21, "cow"},
  {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"}, {27, "backpack"},
  {28, "umbrella"}, {31, "handbag"}, {32, "tie"}, {33, "suitcase"}, {34, "frisbee"},
  {35, "skis"}, {36, "snowboard"}, {37, "sports ball"}, {38, "kite"},
  {39, "baseball bat"}, {40, "baseball glove"}, {41, "skateboard"}, {42, "surfboard"},
  {43, "tennis racket"}, {44, "bottle"}, {46, "wine glass"}, {47, "cup"}, {48, "fork"},
  {49, "knife"}, {50, "spoon"}, {51, "bowl"}, {52, "banana"}, {53, "apple"},
  {54, "sandwich"}, {55, "orange"}, {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"},
  {59, "pizza"}, {60, "donut"}, {61, "cake"}, {62, "chair"}, {63, "couch"},
  {64, "potted plant"}, {65, "bed"}, {67, "dining table"}, {70, "toilet"}, {72, "tv"},
  {73, "laptop"}, {74, "mouse"}, {75, "remote"}, {76, "keyboard"}, {77, "cell phone"},
  {78, "microwave"}, {79, "oven"}, {80, "toaster"}, {81, "sink"}, {82, "refrigerator"},
  {84, "book"}, {85, "clock"}, {86, "vase"}, {87, "scissors"}, {88, "teddy bear"},
  {89, "hair drier"}, {90, "toothbrush"}
};

// Get class name from COCO category ID
std::string get_class_name(int coco_id);

// Utility method to print results
void print_detection_results(const Detections& results, size_t max_detections = 20);

// Visualization method to plot detections on image
void plot_detections(
  const std::string& image_path,
  const Detections& detections,
  float confidence_threshold = 0.5f,
  const std::string& output_path = "detection_results.png");

std::vector<int> apply_nms(
  const std::vector<cv::Rect2f>& boxes,
  const std::vector<float>& scores,
  const std::vector<int>& labels,
  float nms_threshold);

std::vector<int> apply_greedy_nms(
  const std::vector<cv::Rect>& boxes,
  const std::vector<float>& scores,
  float nms_threshold);

float compute_iou(const cv::Rect& box1, const cv::Rect& box2);

cv::Rect2f clip_box_to_image(const cv::Rect2f& box, int image_height, int image_width);

std::vector<int> topk_indices(const std::vector<float>& scores, int k);

} // namespace utils

} // namespace fcos_trt_backend
