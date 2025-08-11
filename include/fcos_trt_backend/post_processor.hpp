#pragma once

// C++ standard library
#include <vector>
#include <memory>
#include <tuple>

// OpenCV includes
#include <opencv2/core.hpp>

// Local includes
#include "fcos_trt_backend.hpp"

namespace fcos_trt_backend
{

struct Detection
{
    cv::Rect2f box;
    float score;
    int64_t label;
};

struct PostProcessResults
{
  std::vector<Detection> detections;
};

class FCOSPostProcessor
{
public:
  struct Config
  {
      float score_thresh;
      float nms_thresh;
      int detections_per_img;
      int topk_candidates;

      Config()
      : score_thresh(0.2f), nms_thresh(0.6f),
        detections_per_img(100), topk_candidates(1000) {}
  };

  explicit FCOSPostProcessor(const Config& config = Config());
  ~FCOSPostProcessor() = default;

  // Disable copy and move semantics
  FCOSPostProcessor(const FCOSPostProcessor&) = delete;
  FCOSPostProcessor& operator=(const FCOSPostProcessor&) = delete;
  FCOSPostProcessor(FCOSPostProcessor&&) = delete;
  FCOSPostProcessor& operator=(FCOSPostProcessor&&) = delete;

  /**
   * @brief Post-process detection results from FCOS backbone
   * @param results Raw detection results from TensorRT inference
   * @param original_image_size Original image size before preprocessing (width, height)
   * @param processed_image_size Processed image size used for inference (width, height)
   * @return Post-processed detection results
   */
  PostProcessResults postprocess(
      const FCOSTrtBackend::DetectionResults& results,
      const cv::Size& original_image_size,
      const cv::Size& processed_image_size
  );

private:
  // Helper methods
  std::vector<float> decode_boxes(
      const std::vector<float>& bbox_regression,
      const std::vector<float>& anchors,
      const std::vector<int>& indices
  );

  std::vector<float> clip_boxes_to_image(
      const std::vector<float>& boxes,
      int image_height,
      int image_width
  );

  std::vector<int> batched_nms(
      const std::vector<float>& boxes,
      const std::vector<float>& scores,
      const std::vector<int64_t>& labels,
      float iou_threshold
  );

  float compute_iou(const cv::Rect2f& box1, const cv::Rect2f& box2);

  std::vector<float> transform_boxes_to_original_size(
      const std::vector<float>& boxes,
      const cv::Size& original_size,
      const cv::Size& processed_size
  );

private:
  Config config_;
};

} // namespace fcos_trt_backend
