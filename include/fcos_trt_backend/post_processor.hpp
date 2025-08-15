#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <vector>
#include <string>

// OpenCV includes
#include <opencv2/core.hpp>

// Local includes
#include "fcos_trt_backend/fcos_trt_backend.hpp"
#include "fcos_trt_backend/detection_types.hpp"


namespace fcos_trt_backend
{

class FCOSPostProcessor
{

public:
  /**
   * @brief Post processor for FCOS
   * @param score_thresh Score threshold for filtering
   * @param nms_thresh NMS threshold
   * @param detections_per_img Maximum detections per image
   * @param topk_candidates Top-k candidates before NMS
   * @details Constructor with default valuse
   */
  explicit FCOSPostProcessor(const float score_thresh = 0.2f,
    const float nms_thresh = 0.6f,
    int detections_per_img = 100,
    int topk_candidates = 1000);

  /**
   * @brief Main postprocessing method
   * @param head_outputs Raw model outputs from FCOSTrtBackend
   * @param original_height Original image height
   * @param original_width Original image width
   * @return Processed detections in original image coordinates
   */
  Detections postprocess_detections(
    const FCOSTrtBackend::HeadOutputs& head_outputs,
    int original_height,
    int original_width);

private:
  // Helper methods for postprocessing pipeline
  std::vector<std::vector<float>> split_tensor_by_levels(
    const std::vector<float>& tensor,
    const std::vector<int64_t>& num_anchors_per_level,
    int tensor_dim);

  // Transform coordinates from processed image space to original image space
  Detections transform_coordinates_to_original(
    const Detections& detections,
    int processed_height,
    int processed_width,
    int original_height,
    int original_width);

private:
  const float score_thresh_;
  const float nms_thresh_;
  const int detections_per_img_;
  const int topk_candidates_;
};

} // namespace fcos_trt_backend
