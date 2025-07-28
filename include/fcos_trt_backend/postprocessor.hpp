#pragma once

#include <vector>
#include <memory>

#include "fcos_trt_backend/types.hpp"

namespace fcos_trt_backend
{

/**
 * @brief FCOS post-processor that mimics the original PyTorch implementation
 *
 * This class implements the same post-processing logic as FCOSPostProcessor in Python:
 * - Decodes bounding boxes from regression outputs and anchors
 * - Computes final scores from classification and centerness
 * - Applies score thresholding and topk filtering
 * - Performs Non-Maximum Suppression (NMS)
 * - Transforms coordinates back to original image space
 */
class FCOSPostProcessor
{
public:
    struct Config
    {
        float score_thresh;     // Score threshold for filtering
        float nms_thresh;       // NMS IoU threshold
        int detections_per_img; // Maximum detections per image
        int topk_candidates;    // Top-k candidates before NMS
        bool normalize_by_size; // Box coder normalization flag

        Config()
        : score_thresh(0.2f), nms_thresh(0.6f), detections_per_img(100),
        topk_candidates(1000), normalize_by_size(true) {}
    };

    explicit FCOSPostProcessor(const Config& config = Config());
    ~FCOSPostProcessor();

    // Disable copy and move semantics
    FCOSPostProcessor(const FCOSPostProcessor&) = delete;
    FCOSPostProcessor& operator=(const FCOSPostProcessor&) = delete;
    FCOSPostProcessor(FCOSPostProcessor&&) = delete;
    FCOSPostProcessor& operator=(FCOSPostProcessor&&) = delete;

    /**
     * @brief Post-process raw FCOS outputs to get final detections
     *
     * @param raw_outputs Raw model outputs from TensorRT inference
     * @param original_width Original image width
     * @param original_height Original image height
     * @return Vector of detections in original image coordinates
     */
    Detections postprocess(
        const RawOutputs& raw_outputs,
        int original_width,
        int original_height) const;

private:
    Config config_;

    // Box decoding functions (mimics torchvision BoxLinearCoder)
    void decode_boxes(
        const float* bbox_regression,
        const float* anchors,
        float* decoded_boxes,
        int num_boxes) const;

    // NMS implementation
    std::vector<int32_t> batched_nms(
        const std::vector<Detection>& detections,
        float nms_thresh) const;

    // Utility functions
    float compute_iou(const Detection& a, const Detection& b) const;
    void clip_boxes_to_image(std::vector<Detection>& detections,
                           int width, int height) const;

    // Transform boxes from processed image space to original image space
    void transform_boxes_to_original(
        std::vector<Detection>& detections,
        int processed_width, int processed_height,
        int original_width, int original_height) const;
};

} // namespace fcos_trt_backend
