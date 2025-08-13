#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// Local includes
#include "fcos_trt_backend/post_processor.hpp"
#include "fcos_trt_backend/exception.hpp"

namespace fcos_trt_backend
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

FCOSPostProcessor::FCOSPostProcessor(const Config& config)
  : config_(config)
{
  std::cout << "FCOSPostProcessor initialized with:" << std::endl;
  std::cout << "  Score threshold: " << config_.score_thresh << std::endl;
  std::cout << "  NMS threshold: " << config_.nms_thresh << std::endl;
  std::cout << "  Detections per image: " << config_.detections_per_img << std::endl;
  std::cout << "  Top-k candidates: " << config_.topk_candidates << std::endl;
}

std::string FCOSPostProcessor::get_class_name(int coco_id) const
{
  auto it = COCO_CATEGORY_NAMES.find(coco_id);
  if (it != COCO_CATEGORY_NAMES.end()) {
    return it->second;
  }
  return "unknown_class_" + std::to_string(coco_id);
}

DetectionResult FCOSPostProcessor::postprocess_detections(
    const FCOSTrtBackend::DetectionResults& raw_outputs,
    int original_height,
    int original_width)
{
  std::cout << "\n=== Starting FCOS Postprocessing ===" << std::endl;

  // Extract image dimensions from image_sizes (these are the processed dimensions)
  if (raw_outputs.image_sizes.size() != 2) {
    throw std::runtime_error("Expected image_sizes to have 2 elements [height, width]");
  }
  int processed_height = static_cast<int>(raw_outputs.image_sizes[0]);
  int processed_width = static_cast<int>(raw_outputs.image_sizes[1]);

  std::cout << "Processed image dimensions: " << processed_height << "x" << processed_width << std::endl;
  std::cout << "Original image dimensions: " << original_height << "x" << original_width << std::endl;

  // Get number of classes from cls_logits shape
  // Assuming cls_logits is flattened: [total_anchors * num_classes]
  size_t total_anchors = raw_outputs.anchors.size() / 4;  // 4 coords per anchor
  size_t num_classes = raw_outputs.cls_logits.size() / total_anchors;

  std::cout << "Total anchors: " << total_anchors << std::endl;
  std::cout << "Number of classes: " << num_classes << std::endl;

  // Verify that we have the expected 91 classes for COCO
  if (num_classes != 91) {
    std::cout << "Warning: Expected 91 classes for COCO, but got " << num_classes << std::endl;
  }

  // Split tensors by pyramid levels
  auto cls_logits_per_level = split_tensor_by_levels(
      raw_outputs.cls_logits, raw_outputs.num_anchors_per_level, num_classes);
  auto bbox_regression_per_level = split_tensor_by_levels(
      raw_outputs.bbox_regression, raw_outputs.num_anchors_per_level, 4);
  auto bbox_ctrness_per_level = split_tensor_by_levels(
      raw_outputs.bbox_ctrness, raw_outputs.num_anchors_per_level, 1);

  // Split anchors by levels
  std::vector<std::vector<float>> anchors_per_level;
  size_t anchor_offset = 0;
  for (size_t level = 0; level < raw_outputs.num_anchors_per_level.size(); ++level) {
    size_t level_anchors = raw_outputs.num_anchors_per_level[level];
    size_t start_idx = anchor_offset * 4;
    size_t end_idx = (anchor_offset + level_anchors) * 4;

    std::vector<float> level_anchor_data(
        raw_outputs.anchors.begin() + start_idx,
        raw_outputs.anchors.begin() + end_idx
        );
    anchors_per_level.push_back(level_anchor_data);
    anchor_offset += level_anchors;
  }

  std::cout << "Split tensors into " << cls_logits_per_level.size() << " pyramid levels" << std::endl;

  // Collect detections from all levels
  std::vector<cv::Rect2f> all_boxes;
  std::vector<float> all_scores;
  std::vector<int> all_labels; // Will store COCO category IDs

  // Process each pyramid level
  for (size_t level = 0; level < cls_logits_per_level.size(); ++level) {
    const auto& cls_logits = cls_logits_per_level[level];
    const auto& bbox_regression = bbox_regression_per_level[level];
    const auto& bbox_ctrness = bbox_ctrness_per_level[level];
    const auto& anchors = anchors_per_level[level];

    size_t level_anchors = raw_outputs.num_anchors_per_level[level];

    // Compute scores: sqrt(sigmoid(cls_score) * sigmoid(centerness))
    std::vector<float> level_scores;
    std::vector<int> level_labels; // COCO category IDs
    std::vector<size_t> keep_indices;

    for (size_t anchor_idx = 0; anchor_idx < level_anchors; ++anchor_idx) {
      float ctrness_score = 1.0f / (1.0f + std::exp(-bbox_ctrness[anchor_idx])); // sigmoid

      for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        float cls_score = cls_logits[anchor_idx * num_classes + class_idx];
        cls_score = 1.0f / (1.0f + std::exp(-cls_score)); // sigmoid

        float final_score = std::sqrt(cls_score * ctrness_score);

        if (final_score > config_.score_thresh) {
          level_scores.push_back(final_score);
          // Class index directly corresponds to COCO category ID
          level_labels.push_back(static_cast<int>(class_idx));
          keep_indices.push_back(anchor_idx);
        }
      }
    }

    if (level_scores.empty()) {
      continue;
    }
    // Apply top-k filtering
    std::vector<int> topk_idx = topk_indices(level_scores, config_.topk_candidates);

    // Decode boxes for kept detections
    for (int idx : topk_idx) {
      size_t anchor_idx = keep_indices[idx];

      // Extract anchor coordinates
      float x1 = anchors[anchor_idx * 4];
      float y1 = anchors[anchor_idx * 4 + 1];
      float x2 = anchors[anchor_idx * 4 + 2];
      float y2 = anchors[anchor_idx * 4 + 3];

      // Extract regression values (these are l, t, r, b distances, NOT dx, dy, dw, dh!)
      float l = bbox_regression[anchor_idx * 4];      // left distance
      float t = bbox_regression[anchor_idx * 4 + 1];  // top distance
      float r = bbox_regression[anchor_idx * 4 + 2];  // right distance
      float b = bbox_regression[anchor_idx * 4 + 3];  // bottom distance

      // Calculate anchor center
      float anchor_width = x2 - x1;
      float anchor_height = y2 - y1;
      float anchor_cx = x1 + 0.5f * anchor_width;
      float anchor_cy = y1 + 0.5f * anchor_height;

      // Scale regression values by anchor size
      float l_scaled = l * anchor_width;
      float t_scaled = t * anchor_height;
      float r_scaled = r * anchor_width;
      float b_scaled = b * anchor_height;

      // Then decode using scaled values
      float pred_x1 = anchor_cx - l_scaled;
      float pred_y1 = anchor_cy - t_scaled;
      float pred_x2 = anchor_cx + r_scaled;
      float pred_y2 = anchor_cy + b_scaled;

      // Clip to processed image boundaries
      cv::Rect2f box(pred_x1, pred_y1, pred_x2 - pred_x1, pred_y2 - pred_y1);
      box = clip_box_to_image(box, processed_height, processed_width);
      all_boxes.push_back(box);
      all_scores.push_back(level_scores[idx]);
      all_labels.push_back(level_labels[idx]); // COCO category ID
    }
  }

  std::cout << "Collected " << all_boxes.size() << " candidate detections" << std::endl;

  // Apply NMS
  std::vector<int> nms_indices = apply_nms(all_boxes, all_scores, all_labels, config_.nms_thresh);

  // Limit to max detections per image
  int max_detections = std::min(config_.detections_per_img, static_cast<int>(nms_indices.size()));
  nms_indices.resize(max_detections);

  std::cout << "After NMS: " << nms_indices.size() << " final detections" << std::endl;

  // Prepare intermediate results (still in processed image coordinates)
  DetectionResult processed_result;
  processed_result.boxes.reserve(nms_indices.size());
  processed_result.scores.reserve(nms_indices.size());
  processed_result.labels.reserve(nms_indices.size());

  for (int idx : nms_indices) {
    processed_result.boxes.push_back(all_boxes[idx]);
    processed_result.scores.push_back(all_scores[idx]);
    processed_result.labels.push_back(all_labels[idx]); // COCO category ID
  }

  // CRITICAL: Transform coordinates from processed image space to original image space
  DetectionResult final_result = transform_coordinates_to_original(
    processed_result,
    processed_height,
    processed_width,
    original_height,
    original_width
  );

  std::cout << "Postprocessing completed successfully!" << std::endl;
  return final_result;
}

DetectionResult FCOSPostProcessor::transform_coordinates_to_original(
    const DetectionResult& detections,
    int processed_height,
    int processed_width,
    int original_height,
    int original_width)
{
  // Calculate scale factors
  float scale_x = static_cast<float>(original_width) / static_cast<float>(processed_width);
  float scale_y = static_cast<float>(original_height) / static_cast<float>(processed_height);

  DetectionResult transformed_result;
  transformed_result.scores = detections.scores;  // Scores don't change
  transformed_result.labels = detections.labels;  // Labels don't change
  transformed_result.boxes.reserve(detections.boxes.size());

  // Transform each bounding box
  for (const auto& box : detections.boxes) {
    // Scale the coordinates
    float new_x = box.x * scale_x;
    float new_y = box.y * scale_y;
    float new_width = box.width * scale_x;
    float new_height = box.height * scale_y;

    // Clip to original image boundaries
    cv::Rect2f transformed_box(new_x, new_y, new_width, new_height);
    transformed_box = clip_box_to_image(transformed_box, original_height, original_width);

    transformed_result.boxes.push_back(transformed_box);
  }

  return transformed_result;
}

std::vector<std::vector<float>> FCOSPostProcessor::split_tensor_by_levels(
    const std::vector<float>& tensor,
    const std::vector<int64_t>& num_anchors_per_level,
    int tensor_dim)
{
  std::vector<std::vector<float>> split_tensors;
  size_t offset = 0;

  for (int64_t level_anchors : num_anchors_per_level) {
    size_t level_size = level_anchors * tensor_dim;
    std::vector<float> level_tensor(
        tensor.begin() + offset,
        tensor.begin() + offset + level_size
        );
    split_tensors.push_back(level_tensor);
    offset += level_size;
  }

  return split_tensors;
}

std::vector<int> FCOSPostProcessor::apply_nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& labels,
    float nms_threshold)
{
  if (boxes.empty()) {
    return {};
  }

  // Group by class labels for class-wise NMS
  std::unordered_map<int, std::vector<int>> class_indices;
  for (size_t i = 0; i < labels.size(); ++i) {
    class_indices[labels[i]].push_back(i);
  }

  std::vector<int> final_indices;

  // Apply NMS for each class
  for (const auto& class_pair : class_indices) {
    const std::vector<int>& indices = class_pair.second;

    if (indices.size() <= 1) {
      // No need for NMS if only one box
      for (int idx : indices) {
        final_indices.push_back(idx);
      }
      continue;
    }

    // Prepare data for OpenCV NMS - convert to cv::Rect (int) for compatibility
    std::vector<cv::Rect> class_boxes;
    std::vector<float> class_scores;

    for (int idx : indices) {
      // Convert cv::Rect2f to cv::Rect for OpenCV compatibility
      cv::Rect int_box(
          static_cast<int>(boxes[idx].x),
          static_cast<int>(boxes[idx].y),
          static_cast<int>(boxes[idx].width),
          static_cast<int>(boxes[idx].height)
          );
      class_boxes.push_back(int_box);
      class_scores.push_back(scores[idx]);
    }

    // Apply OpenCV NMS with compatible signature
    std::vector<int> nms_indices;
    try {
      cv::dnn::NMSBoxes(class_boxes, class_scores, 0.0f, nms_threshold, nms_indices);
    } catch (const cv::Exception& e) {
      // Fallback: implement simple greedy NMS if OpenCV version issues
      std::cout << "OpenCV NMS failed, using fallback implementation" << std::endl;
      nms_indices = apply_greedy_nms(class_boxes, class_scores, nms_threshold);
    }

    // Convert back to original indices
    for (int nms_idx : nms_indices) {
      final_indices.push_back(indices[nms_idx]);
    }
  }

  // Sort by score (descending)
  std::sort(final_indices.begin(), final_indices.end(),
      [&scores](int a, int b) {
      return scores[a] > scores[b];
      });

  return final_indices;
}

std::vector<int> FCOSPostProcessor::apply_greedy_nms(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float nms_threshold)
{
  std::vector<int> indices(boxes.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Sort by score (descending)
  std::sort(indices.begin(), indices.end(),
      [&scores](int a, int b) {
      return scores[a] > scores[b];
      });

  std::vector<bool> suppressed(boxes.size(), false);
  std::vector<int> keep;

  for (size_t i = 0; i < indices.size(); ++i) {
    int idx = indices[i];
    if (suppressed[idx]) {
      continue;
    }

    keep.push_back(idx);

    // Suppress overlapping boxes
    for (size_t j = i + 1; j < indices.size(); ++j) {
      int next_idx = indices[j];
      if (suppressed[next_idx]) {
        continue;
      }

      float iou = compute_iou(boxes[idx], boxes[next_idx]);
      if (iou > nms_threshold) {
        suppressed[next_idx] = true;
      }
    }
  }

  return keep;
}

float FCOSPostProcessor::compute_iou(const cv::Rect& box1, const cv::Rect& box2)
{
  int x1 = std::max(box1.x, box2.x);
  int y1 = std::max(box1.y, box2.y);
  int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
  int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

  if (x2 <= x1 || y2 <= y1) {
    return 0.0f;
  }

  float intersection = (x2 - x1) * (y2 - y1);
  float area1 = box1.width * box1.height;
  float area2 = box2.width * box2.height;
  float union_area = area1 + area2 - intersection;

  return intersection / union_area;
}

cv::Rect2f FCOSPostProcessor::clip_box_to_image(
    const cv::Rect2f& box, int image_height, int image_width)
{
  float x1 = std::max(0.0f, box.x);
  float y1 = std::max(0.0f, box.y);
  float x2 = std::min(static_cast<float>(image_width), box.x + box.width);
  float y2 = std::min(static_cast<float>(image_height), box.y + box.height);

  return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
}

std::vector<int> FCOSPostProcessor::topk_indices(const std::vector<float>& scores, int k)
{
  std::vector<int> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Partial sort to get top-k
  int actual_k = std::min(k, static_cast<int>(scores.size()));
  std::partial_sort(indices.begin(), indices.begin() + actual_k, indices.end(),
      [&scores](int a, int b) {
      return scores[a] > scores[b];
      });

  indices.resize(actual_k);
  return indices;
}

void FCOSPostProcessor::print_detection_results(
    const DetectionResult& results, int max_detections)
{
  std::cout << "\n=== Detection Results ===" << std::endl;
  std::cout << "Total detections: " << results.boxes.size() << std::endl;

  int print_count = std::min(max_detections, static_cast<int>(results.boxes.size()));

  for (int i = 0; i < print_count; ++i) {
    const auto& box = results.boxes[i];
    float score = results.scores[i];
    int coco_id = results.labels[i]; // This is now a COCO category ID

    std::string class_name = get_class_name(coco_id);

    std::cout << "Detection " << (i + 1) << ": " << class_name
      << " (COCO ID: " << coco_id << ") - Confidence: " << score << std::endl;
    std::cout << "  Box: [" << box.x << ", " << box.y << ", "
      << (box.x + box.width) << ", " << (box.y + box.height) << "]" << std::endl;
  }

  if (results.boxes.size() > max_detections) {
    std::cout << "... and " << (results.boxes.size() - max_detections)
      << " more detections" << std::endl;
  }
}

} // namespace fcos_trt_backend
