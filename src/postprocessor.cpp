#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

#include "fcos_trt_backend/postprocessor.hpp"


namespace fcos_trt_backend
{

FCOSPostProcessor::FCOSPostProcessor(const Config& config)
    : config_(config)
{
}

FCOSPostProcessor::~FCOSPostProcessor() = default;

Detections FCOSPostProcessor::postprocess(
    const RawOutputs& raw_outputs,
    int original_width,
    int original_height) const
{
    if (!raw_outputs.cls_logits || !raw_outputs.bbox_regression ||
        !raw_outputs.bbox_ctrness || !raw_outputs.anchors) {
        return {};
    }

    std::vector<Detection> all_detections;

    // Get processed image dimensions (assuming single image batch)
    int processed_width = raw_outputs.image_sizes[0];
    int processed_height = raw_outputs.image_sizes[1];

    // Process each pyramid level
    int anchor_offset = 0;
    for (int level = 0; level < raw_outputs.num_levels; ++level) {
        int num_anchors_level = raw_outputs.num_anchors_per_level[level];

        // Get pointers to this level's data
        const float* cls_logits_level = raw_outputs.cls_logits +
            anchor_offset * raw_outputs.num_classes;
        const float* bbox_regression_level = raw_outputs.bbox_regression +
            anchor_offset * 4;
        const float* bbox_ctrness_level = raw_outputs.bbox_ctrness + anchor_offset;
        const float* anchors_level = raw_outputs.anchors + anchor_offset * 4;

        // Process each anchor at this level
        for (int anchor_idx = 0; anchor_idx < num_anchors_level; ++anchor_idx) {
            const float* cls_scores = cls_logits_level + anchor_idx * raw_outputs.num_classes;
            const float* bbox_reg = bbox_regression_level + anchor_idx * 4;
            float centerness = bbox_ctrness_level[anchor_idx];
            const float* anchor = anchors_level + anchor_idx * 4;

            // Apply sigmoid to centerness
            float centerness_sigmoid = 1.0f / (1.0f + std::exp(-centerness));

            // Check each class
            for (int class_idx = 0; class_idx < raw_outputs.num_classes; ++class_idx) {
                // Apply sigmoid to classification score
                float cls_sigmoid = 1.0f / (1.0f + std::exp(-cls_scores[class_idx]));

                // Compute final score: sqrt(cls_score * centerness)
                float final_score = std::sqrt(cls_sigmoid * centerness_sigmoid);

                // Apply score threshold
                if (final_score <= config_.score_thresh) {
                    continue;
                }

                // Decode bounding box
                float decoded_box[4];
                decode_boxes(bbox_reg, anchor, decoded_box, 1);

                // Create detection
                Detection detection(
                    decoded_box[0], decoded_box[1], decoded_box[2], decoded_box[3],
                    final_score, class_idx
                );

                all_detections.push_back(detection);
            }
        }

        anchor_offset += num_anchors_level;
    }

    // Apply topk filtering
    if (all_detections.size() > static_cast<size_t>(config_.topk_candidates)) {
        // Sort by score descending
        std::partial_sort(all_detections.begin(),
                         all_detections.begin() + config_.topk_candidates,
                         all_detections.end(),
                         [](const Detection& a, const Detection& b) {
                             return a.score > b.score;
                         });
        all_detections.resize(config_.topk_candidates);
    }

    // Clip boxes to processed image bounds
    clip_boxes_to_image(all_detections, processed_width, processed_height);

    // Apply NMS
    auto keep_indices = batched_nms(all_detections, config_.nms_thresh);

    // Keep only NMS survivors up to detections_per_img limit
    std::vector<Detection> final_detections;
    int max_detections = std::min(config_.detections_per_img,
                                  static_cast<int>(keep_indices.size()));

    for (int i = 0; i < max_detections; ++i) {
        final_detections.push_back(all_detections[keep_indices[i]]);
    }

    // Transform coordinates back to original image space
    transform_boxes_to_original(final_detections,
                               processed_width, processed_height,
                               original_width, original_height);

    return final_detections;
}

void FCOSPostProcessor::decode_boxes(
    const float* bbox_regression,
    const float* anchors,
    float* decoded_boxes,
    int num_boxes) const
{
    // Implementation of torchvision BoxLinearCoder.decode
    // bbox_regression: [dx, dy, dw, dh] (relative to anchor)
    // anchors: [x1, y1, x2, y2] (absolute coordinates)

    for (int i = 0; i < num_boxes; ++i) {
        float anchor_x1 = anchors[i * 4 + 0];
        float anchor_y1 = anchors[i * 4 + 1];
        float anchor_x2 = anchors[i * 4 + 2];
        float anchor_y2 = anchors[i * 4 + 3];

        float anchor_width = anchor_x2 - anchor_x1;
        float anchor_height = anchor_y2 - anchor_y1;
        float anchor_center_x = anchor_x1 + 0.5f * anchor_width;
        float anchor_center_y = anchor_y1 + 0.5f * anchor_height;

        float dx = bbox_regression[i * 4 + 0];
        float dy = bbox_regression[i * 4 + 1];
        float dw = bbox_regression[i * 4 + 2];
        float dh = bbox_regression[i * 4 + 3];

        // Normalize by anchor size if enabled (default behavior)
        if (config_.normalize_by_size) {
            dx *= anchor_width;
            dy *= anchor_height;
            dw *= anchor_width;
            dh *= anchor_height;
        }

        // Compute predicted center
        float pred_center_x = anchor_center_x + dx;
        float pred_center_y = anchor_center_y + dy;

        // Compute predicted size (use exp to ensure positive values)
        float pred_width = anchor_width * std::exp(dw);
        float pred_height = anchor_height * std::exp(dh);

        // Convert back to corners
        decoded_boxes[i * 4 + 0] = pred_center_x - 0.5f * pred_width;  // x1
        decoded_boxes[i * 4 + 1] = pred_center_y - 0.5f * pred_height; // y1
        decoded_boxes[i * 4 + 2] = pred_center_x + 0.5f * pred_width;  // x2
        decoded_boxes[i * 4 + 3] = pred_center_y + 0.5f * pred_height; // y2
    }
}

std::vector<int32_t> FCOSPostProcessor::batched_nms(
    const std::vector<Detection>& detections,
    float nms_thresh) const
{
    if (detections.empty()) {
        return {};
    }

    // Group detections by class
    std::vector<std::vector<int32_t>> class_detections(81); // COCO has 80 classes + background

    for (size_t i = 0; i < detections.size(); ++i) {
        int class_id = detections[i].label;
        if (class_id >= 0 && class_id < 81) {
            class_detections[class_id].push_back(static_cast<int32_t>(i));
        }
    }

    std::vector<int32_t> keep_indices;

    // Apply NMS per class
    for (const auto& class_indices : class_detections) {
        if (class_indices.empty()) {
            continue;
        }

        // Sort indices by score (descending)
        std::vector<int32_t> sorted_indices = class_indices;
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&detections](int32_t a, int32_t b) {
                      return detections[a].score > detections[b].score;
                  });

        // NMS for this class
        std::vector<bool> suppressed(sorted_indices.size(), false);

        for (size_t i = 0; i < sorted_indices.size(); ++i) {
            if (suppressed[i]) {
                continue;
            }

            keep_indices.push_back(sorted_indices[i]);

            // Suppress overlapping boxes
            for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
                if (!suppressed[j]) {
                    float iou = compute_iou(detections[sorted_indices[i]],
                                          detections[sorted_indices[j]]);
                    if (iou > nms_thresh) {
                        suppressed[j] = true;
                    }
                }
            }
        }
    }

    // Sort final results by score
    std::sort(keep_indices.begin(), keep_indices.end(),
              [&detections](int32_t a, int32_t b) {
                  return detections[a].score > detections[b].score;
              });

    return keep_indices;
}

float FCOSPostProcessor::compute_iou(const Detection& a, const Detection& b) const
{
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    return intersection / union_area;
}

void FCOSPostProcessor::clip_boxes_to_image(
    std::vector<Detection>& detections,
    int width, int height) const
{
    for (auto& det : detections) {
        det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(width - 1)));
        det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(height - 1)));
        det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(width - 1)));
        det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(height - 1)));
    }
}

void FCOSPostProcessor::transform_boxes_to_original(
    std::vector<Detection>& detections,
    int processed_width, int processed_height,
    int original_width, int original_height) const
{
    if (processed_width == original_width && processed_height == original_height) {
        return; // No transformation needed
    }

    float scale_x = static_cast<float>(original_width) / processed_width;
    float scale_y = static_cast<float>(original_height) / processed_height;

    for (auto& det : detections) {
        det.x1 *= scale_x;
        det.y1 *= scale_y;
        det.x2 *= scale_x;
        det.y2 *= scale_y;

        // Clip to original image bounds
        det.x1 = std::max(0.0f, std::min(det.x1, static_cast<float>(original_width - 1)));
        det.y1 = std::max(0.0f, std::min(det.y1, static_cast<float>(original_height - 1)));
        det.x2 = std::max(0.0f, std::min(det.x2, static_cast<float>(original_width - 1)));
        det.y2 = std::max(0.0f, std::min(det.y2, static_cast<float>(original_height - 1)));
    }
}

} // namespace fcos_trt_backend
