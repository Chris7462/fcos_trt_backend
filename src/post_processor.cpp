#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>

#include "fcos_trt_backend/post_processor.hpp"
#include "fcos_trt_backend/exception.hpp"


namespace fcos_trt_backend
{

FCOSPostProcessor::FCOSPostProcessor(const Config& config)
: config_(config)
{
}

PostProcessResults FCOSPostProcessor::postprocess(
  const FCOSTrtBackend::DetectionResults& results,
  const cv::Size& original_image_size,
  const cv::Size& processed_image_size)
{
  // Get tensor sizes
  const auto& cls_logits = results.cls_logits;
  const auto& bbox_regression = results.bbox_regression;
  const auto& bbox_ctrness = results.bbox_ctrness;
  const auto& anchors = results.anchors;
  const auto& num_anchors_per_level = results.num_anchors_per_level;

  // Calculate number of classes from cls_logits
  int num_classes = 80;

  // Split outputs per level
  std::vector<std::vector<float>> cls_logits_per_level;
  std::vector<std::vector<float>> bbox_regression_per_level;
  std::vector<std::vector<float>> bbox_ctrness_per_level;
  std::vector<std::vector<float>> anchors_per_level;

  size_t cls_offset = 0, bbox_offset = 0, ctr_offset = 0, anchor_offset = 0;

  for (size_t level = 0; level < num_anchors_per_level.size(); ++level) {
    int64_t num_anchors = num_anchors_per_level[level];

    // Extract cls_logits for this level (num_anchors * num_classes)
    size_t cls_size = num_anchors * num_classes;
    cls_logits_per_level.emplace_back(
      cls_logits.begin() + cls_offset,
      cls_logits.begin() + cls_offset + cls_size
    );
    cls_offset += cls_size;

    // Extract bbox_regression for this level (num_anchors * 4)
    size_t bbox_size = num_anchors * 4;
    bbox_regression_per_level.emplace_back(
      bbox_regression.begin() + bbox_offset,
      bbox_regression.begin() + bbox_offset + bbox_size
    );
    bbox_offset += bbox_size;

    // Extract bbox_ctrness for this level (num_anchors)
    bbox_ctrness_per_level.emplace_back(
      bbox_ctrness.begin() + ctr_offset,
      bbox_ctrness.begin() + ctr_offset + num_anchors
    );
    ctr_offset += num_anchors;

    // Extract anchors for this level (num_anchors * 4)
    size_t anchor_size = num_anchors * 4;
    anchors_per_level.emplace_back(
      anchors.begin() + anchor_offset,
      anchors.begin() + anchor_offset + anchor_size
    );
    anchor_offset += anchor_size;
  }

  // Get image dimensions from results
  int image_height = static_cast<int>(results.image_sizes[0]);
  int image_width = static_cast<int>(results.image_sizes[1]);

  std::vector<float> all_boxes;
  std::vector<float> all_scores;
  std::vector<int64_t> all_labels;

  // Process each pyramid level
  for (size_t level = 0; level < num_anchors_per_level.size(); ++level) {
    const auto& cls_logits_level = cls_logits_per_level[level];
    const auto& bbox_regression_level = bbox_regression_per_level[level];
    const auto& bbox_ctrness_level = bbox_ctrness_per_level[level];
    const auto& anchors_level = anchors_per_level[level];

    int64_t num_anchors = num_anchors_per_level[level];

    // Compute scores: sqrt(sigmoid(cls_logits) * sigmoid(ctrness))
    std::vector<float> scores;
    std::vector<int> valid_indices;
    std::vector<int64_t> labels;

    for (int64_t anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
      float ctrness = 1.0f / (1.0f + std::exp(-bbox_ctrness_level[anchor_idx])); // sigmoid

      for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        float cls_score = 1.0f / (1.0f + std::exp(-cls_logits_level[anchor_idx * num_classes + class_idx])); // sigmoid
        float combined_score = std::sqrt(cls_score * ctrness);

        if (combined_score > config_.score_thresh) {
          scores.push_back(combined_score);
          valid_indices.push_back(static_cast<int>(anchor_idx));
          labels.push_back(class_idx + 1); // COCO classes are 1-indexed
        }
      }
    }

    if (scores.empty()) continue;

    // Keep only top-k candidates per level
    int num_keep = std::min(static_cast<int>(scores.size()), config_.topk_candidates);

    // Create indices for sorting
    std::vector<int> sort_indices(scores.size());
    std::iota(sort_indices.begin(), sort_indices.end(), 0);

      // Sort by score (descending)
    std::partial_sort(
      sort_indices.begin(),
      sort_indices.begin() + num_keep,
      sort_indices.end(),
      [&scores](int a, int b) { return scores[a] > scores[b]; }
    );

    // Extract top-k
    std::vector<float> topk_scores;
    std::vector<int> topk_indices;
    std::vector<int64_t> topk_labels;

    for (int i = 0; i < num_keep; ++i) {
      int idx = sort_indices[i];
      topk_scores.push_back(scores[idx]);
      topk_indices.push_back(valid_indices[idx]);
      topk_labels.push_back(labels[idx]);
    }

    // Decode bounding boxes
    std::vector<float> decoded_boxes = decode_boxes(bbox_regression_level, anchors_level, topk_indices);

    // Clip boxes to image
    decoded_boxes = clip_boxes_to_image(decoded_boxes, image_height, image_width);

    // Add to global lists
    all_boxes.insert(all_boxes.end(), decoded_boxes.begin(), decoded_boxes.end());
    all_scores.insert(all_scores.end(), topk_scores.begin(), topk_scores.end());
    all_labels.insert(all_labels.end(), topk_labels.begin(), topk_labels.end());
  }

  if (all_boxes.empty()) {
    return PostProcessResults{}; // Return empty results
  }

  // Apply NMS
  std::vector<int> keep_indices = batched_nms(all_boxes, all_scores, all_labels, config_.nms_thresh);

  // Limit to max detections
  int max_detections = std::min(static_cast<int>(keep_indices.size()), config_.detections_per_img);
  keep_indices.resize(max_detections);

  // Transform boxes to original image size
  std::vector<float> final_boxes;
  for (int idx : keep_indices) {
    for (int i = 0; i < 4; ++i) {
      final_boxes.push_back(all_boxes[idx * 4 + i]);
    }
  }
  final_boxes = transform_boxes_to_original_size(final_boxes, original_image_size, processed_image_size);

  // Create final results
  PostProcessResults final_results;
  for (size_t i = 0; i < keep_indices.size(); ++i) {
    int idx = keep_indices[i];
    Detection detection;
    detection.box = cv::Rect2f(
      final_boxes[i * 4],     // x
      final_boxes[i * 4 + 1], // y
      final_boxes[i * 4 + 2] - final_boxes[i * 4],     // width
      final_boxes[i * 4 + 3] - final_boxes[i * 4 + 1]  // height
    );
    detection.score = all_scores[idx];
    detection.label = all_labels[idx];
    final_results.detections.push_back(detection);
  }

  return final_results;
}

std::vector<float> FCOSPostProcessor::decode_boxes(
  const std::vector<float>& bbox_regression,
  const std::vector<float>& anchors,
  const std::vector<int>& indices)
{
  std::vector<float> decoded_boxes;
  decoded_boxes.reserve(indices.size() * 4);

  for (int idx : indices) {
    // Get anchor coordinates (x1, y1, x2, y2)
    float anchor_x1 = anchors[idx * 4];
    float anchor_y1 = anchors[idx * 4 + 1];
    float anchor_x2 = anchors[idx * 4 + 2];
    float anchor_y2 = anchors[idx * 4 + 3];

    // Get regression deltas (l, t, r, b) - FCOS format
    float delta_l = bbox_regression[idx * 4];     // left
    float delta_t = bbox_regression[idx * 4 + 1]; // top
    float delta_r = bbox_regression[idx * 4 + 2]; // right
    float delta_b = bbox_regression[idx * 4 + 3]; // bottom

    // Calculate anchor center and size
    float anchor_cx = (anchor_x1 + anchor_x2) / 2.0f;
    float anchor_cy = (anchor_y1 + anchor_y2) / 2.0f;
    float anchor_w = anchor_x2 - anchor_x1;
    float anchor_h = anchor_y2 - anchor_y1;

    // Decode using FCOS format: center point regression
    // The deltas represent distances from center to box edges
    float box_x1 = anchor_cx - delta_l * anchor_w;
    float box_y1 = anchor_cy - delta_t * anchor_h;
    float box_x2 = anchor_cx + delta_r * anchor_w;
    float box_y2 = anchor_cy + delta_b * anchor_h;

    decoded_boxes.push_back(box_x1);
    decoded_boxes.push_back(box_y1);
    decoded_boxes.push_back(box_x2);
    decoded_boxes.push_back(box_y2);
  }

  return decoded_boxes;
}

std::vector<float> FCOSPostProcessor::clip_boxes_to_image(
  const std::vector<float>& boxes,
  int image_height,
  int image_width)
{
  std::vector<float> clipped_boxes = boxes;

  for (size_t i = 0; i < clipped_boxes.size(); i += 4) {
    // Clip x coordinates
    clipped_boxes[i] = std::max(0.0f, std::min(clipped_boxes[i], static_cast<float>(image_width)));
    clipped_boxes[i + 2] = std::max(0.0f, std::min(clipped_boxes[i + 2], static_cast<float>(image_width)));

    // Clip y coordinates
    clipped_boxes[i + 1] = std::max(0.0f, std::min(clipped_boxes[i + 1], static_cast<float>(image_height)));
    clipped_boxes[i + 3] = std::max(0.0f, std::min(clipped_boxes[i + 3], static_cast<float>(image_height)));
  }

  return clipped_boxes;
}

std::vector<int> FCOSPostProcessor::batched_nms(
  const std::vector<float>& boxes,
  const std::vector<float>& scores,
  const std::vector<int64_t>& labels,
  float iou_threshold)
{
  std::vector<int> keep;

  if (boxes.empty() || scores.empty()) {
      return keep;
  }

  // Group by class labels
  std::map<int64_t, std::vector<int>> class_indices;
  for (size_t i = 0; i < labels.size(); ++i) {
    class_indices[labels[i]].push_back(static_cast<int>(i));
  }

  // Apply NMS for each class separately
  for (const auto& [class_id, indices] : class_indices) {
    // Sort indices by score (descending)
    std::vector<int> sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(sorted_indices.size(), false);

    for (size_t i = 0; i < sorted_indices.size(); ++i) {
      if (suppressed[i]) continue;

      int idx_i = sorted_indices[i];
      keep.push_back(idx_i);

      cv::Rect2f box_i(
        boxes[idx_i * 4],
        boxes[idx_i * 4 + 1],
        boxes[idx_i * 4 + 2] - boxes[idx_i * 4],
        boxes[idx_i * 4 + 3] - boxes[idx_i * 4 + 1]
      );

      // Suppress overlapping boxes
      for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
        if (suppressed[j]) continue;

        int idx_j = sorted_indices[j];
        cv::Rect2f box_j(
          boxes[idx_j * 4],
          boxes[idx_j * 4 + 1],
          boxes[idx_j * 4 + 2] - boxes[idx_j * 4],
          boxes[idx_j * 4 + 3] - boxes[idx_j * 4 + 1]
        );

        float iou = compute_iou(box_i, box_j);
        if (iou > iou_threshold) {
          suppressed[j] = true;
        }
      }
    }
  }

  // Sort final results by score (descending)
  std::sort(keep.begin(), keep.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

  return keep;
}

float FCOSPostProcessor::compute_iou(const cv::Rect2f& box1, const cv::Rect2f& box2)
{
  float intersection_area = (box1 & box2).area();
  float union_area = box1.area() + box2.area() - intersection_area;

  if (union_area <= 0.0f) {
    return 0.0f;
  }

  return intersection_area / union_area;
}

std::vector<float> FCOSPostProcessor::transform_boxes_to_original_size(
  const std::vector<float>& boxes,
  const cv::Size& original_size,
  const cv::Size& processed_size)
{
  std::vector<float> transformed_boxes = boxes;

  float scale_x = static_cast<float>(original_size.width) / static_cast<float>(processed_size.width);
  float scale_y = static_cast<float>(original_size.height) / static_cast<float>(processed_size.height);

  for (size_t i = 0; i < transformed_boxes.size(); i += 4) {
    transformed_boxes[i] *= scale_x;         // x1
    transformed_boxes[i + 1] *= scale_y;     // y1
    transformed_boxes[i + 2] *= scale_x;     // x2
    transformed_boxes[i + 3] *= scale_y;     // y2
  }

  return transformed_boxes;
}

} // namespace fcos_trt_backend
