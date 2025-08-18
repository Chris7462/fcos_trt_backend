#include <iostream>
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

// Local includes
#include "fcos_trt_backend/detection_utils.hpp"


namespace fcos_trt_backend
{

namespace utils
{

std::string get_class_name(int coco_id)
{
  auto it = COCO_CATEGORY_NAMES.find(coco_id);
  if (it != COCO_CATEGORY_NAMES.end()) {
    return it->second;
  }
  return "unknown_class_" + std::to_string(coco_id);
}

void print_detection_results(
  const Detections& results, size_t max_detections)
{
  std::cout << "\n=== Detection Results ===" << std::endl;
  std::cout << "Total detections: " << results.boxes.size() << std::endl;

  size_t print_count = std::min(max_detections, results.boxes.size());

  for (size_t i = 0; i < print_count; ++i) {
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

cv::Mat plot_detections(
  const cv::Mat & image,
  const Detections& detections,
  float confidence_threshold)
{
  if (image.empty()) {
    std::cerr << "Input image is empty" << std::endl;
    return cv::Mat(); // Return empty Mat on error
  }

  cv::Mat image_for_plot = image.clone();

  // Draw predictions with confidence > threshold
  for (size_t i = 0; i < detections.boxes.size(); ++i) {
    if (detections.scores[i] >= confidence_threshold) {

      const auto& box = detections.boxes[i];
      int x1 = static_cast<int>(box.x);
      int y1 = static_cast<int>(box.y);
      int x2 = static_cast<int>(box.x + box.width);
      int y2 = static_cast<int>(box.y + box.height);

      // Draw bounding box in green
      cv::rectangle(image_for_plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

      // Get class name
      int label_id = detections.labels[i];
      std::string class_name = get_class_name(label_id);

      // Create label text
      std::string label_text = class_name + ": " +
        std::to_string(detections.scores[i]).substr(0, 5);

      // Calculate text size for background rectangle
      int baseline = 0;
      cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
        0.55, 1.8, &baseline);

      // Draw background rectangle for text
      cv::rectangle(image_for_plot,
        cv::Point(x1, y1 - text_size.height - 4),
        cv::Point(x1 + text_size.width, y1),
        cv::Scalar(0, 255, 0), -1);

      // Draw text
      cv::putText(image_for_plot, label_text, cv::Point(x1, y1 - 4),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1.8);
    }
  }

  return image_for_plot;
}

std::vector<int> apply_nms(
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

std::vector<int> apply_greedy_nms(
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

float compute_iou(const cv::Rect& box1, const cv::Rect& box2)
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

cv::Rect2f clip_box_to_image(
  const cv::Rect2f& box, int image_height, int image_width)
{
  float x1 = std::max(0.0f, box.x);
  float y1 = std::max(0.0f, box.y);
  float x2 = std::min(static_cast<float>(image_width), box.x + box.width);
  float y2 = std::min(static_cast<float>(image_height), box.y + box.height);

  return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
}

std::vector<int> topk_indices(const std::vector<float>& scores, int k)
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

} // namespace utils

} // namespace fcos_trt_backend
