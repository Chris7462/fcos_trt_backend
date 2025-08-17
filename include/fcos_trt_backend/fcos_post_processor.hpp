#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <vector>
#include <string>

// OpenCV includes
#include <opencv2/core.hpp>

// Local includes
#include "fcos_trt_backend/head_types.hpp"
#include "fcos_trt_backend/detection_types.hpp"


namespace fcos_trt_backend
{

class FCOSPostProcessor
{
public:
  struct Config
  {
    /**
     * @brief Score threshold for filtering
     */
    float score_thresh;

    /**
     * @brief NMS threshold
     */
    float nms_thresh;

    /**
     * @brief Maximum detections per image
     */
    int detections_per_img;

    /**
     * @brief Top-k candidates before NMS
     */
    int topk_candidates;

    /**
     * @brief Default constructor
     * @details Initializes the configuration with default values.
     */
    Config()
    : score_thresh(0.2f), nms_thresh(0.6f), detections_per_img(100),
      topk_candidates(1000) {}
  };

  // Constructor with configuration
  explicit FCOSPostProcessor(const Config & confog = Config());

  /**
   * @brief Main postprocessing method
   * @param head_outputs Raw model outputs from FCOSTrtBackend
   * @param original_height Original image height
   * @param original_width Original image width
   * @return Processed detections in original image coordinates
   */
  Detections postprocess_detections(
    const HeadOutputs& head_outputs,
    int original_height,
    int original_width);

private:
  // Configuration
  Config config_;

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

  static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
  }
};

} // namespace fcos_trt_backend
