#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <vector>

// OpenCV includes
#include <opencv2/core.hpp>


namespace fcos_trt_backend
{
/**
 * @brief Head output structure
 * @details Contains cls_logits, bbox_regression, bbox_ctrness, and
 *          anchors. Image sizes is the input tensor size for backbone,
 *          not the original image size. The num_anchers_per_level is
 *          directly outputted from the backbone and will be used for
 *          post-processing.
 */
struct HeadOutputs
{
  std::vector<float> cls_logits;
  std::vector<float> bbox_regression;
  std::vector<float> bbox_ctrness;
  std::vector<float> anchors;
  std::vector<int64_t> image_sizes;
  std::vector<int64_t> num_anchors_per_level;
};

} // namespace fcos_trt_backend
