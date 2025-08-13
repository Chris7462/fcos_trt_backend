#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>

// OpenCV includes
#include <opencv2/core.hpp>

// Local includes
#include "fcos_trt_backend/fcos_trt_backend.hpp"

namespace fcos_trt_backend
{

struct DetectionResult
{
  std::vector<cv::Rect2f> boxes;      // Bounding boxes
  std::vector<float> scores;          // Confidence scores
  std::vector<int> labels;            // Class labels (COCO category IDs)
};

class FCOSPostProcessor
{
public:
  struct Config
  {
    float score_thresh;             // Score threshold for filtering
    float nms_thresh;               // NMS threshold
    int detections_per_img;         // Maximum detections per image
    int topk_candidates;            // Top-k candidates before NMS
    bool normalize_by_size;         // Box coder normalization flag

    Config()
      : score_thresh(0.2f),
      nms_thresh(0.6f),
      detections_per_img(100),
      topk_candidates(1000),
      normalize_by_size(true) {}
  };

  explicit FCOSPostProcessor(const Config& config = Config());

  // Main postprocessing method - now takes original image dimensions
  DetectionResult postprocess_detections(
    const FCOSTrtBackend::DetectionResults& raw_outputs,
    int original_height,
    int original_width);

  // Utility method to print results
  void print_detection_results(const DetectionResult& results, int max_detections = 20);

private:
  Config config_;

  // Helper methods for postprocessing pipeline
  std::vector<std::vector<float>> split_tensor_by_levels(
    const std::vector<float>& tensor,
    const std::vector<int64_t>& num_anchors_per_level,
    int tensor_dim);

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

  // Transform coordinates from processed image space to original image space
  DetectionResult transform_coordinates_to_original(
    const DetectionResult& detections,
    int processed_height,
    int processed_width,
    int original_height,
    int original_width);

  // Get class name from COCO category ID
  std::string get_class_name(int coco_id) const;
};

// COCO class names mapping with correct category IDs (with gaps)
extern const std::unordered_map<int, std::string> COCO_CATEGORY_NAMES;

} // namespace fcos_trt_backend
