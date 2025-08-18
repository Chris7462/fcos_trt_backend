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

// Initialize COCO colors once
const std::unordered_map<int, Color>& get_coco_colors() {
  static std::unordered_map<int, Color> coco_colors = []() {
    std::vector<Color> base_colors = {
      Color(255, 0, 0), Color(0, 255, 0), Color(0, 0, 255), Color(255, 255, 0),
      Color(255, 0, 255), Color(0, 255, 255), Color(255, 128, 0), Color(128, 0, 255),
      Color(255, 0, 128), Color(128, 255, 0), Color(0, 128, 255), Color(255, 128, 128),
      Color(128, 255, 128), Color(128, 128, 255), Color(255, 192, 0), Color(192, 0, 255),
      Color(0, 192, 255), Color(255, 64, 64), Color(64, 255, 64), Color(64, 64, 255),
      Color(255, 255, 128), Color(255, 128, 255), Color(128, 255, 255), Color(192, 96, 0),
      Color(96, 0, 192), Color(0, 96, 192), Color(224, 32, 32), Color(32, 224, 32),
      Color(32, 32, 224), Color(224, 224, 0), Color(224, 0, 224), Color(0, 224, 224),
      Color(160, 82, 45), Color(255, 20, 147), Color(0, 100, 0), Color(139, 69, 19),
      Color(255, 140, 0), Color(218, 112, 214), Color(30, 144, 255), Color(220, 20, 60)
    };

    const std::vector<int> coco_ids = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
      42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
      61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
      84, 85, 86, 87, 88, 89, 90
    };

    std::unordered_map<int, Color> colors;
    for (size_t i = 0; i < coco_ids.size(); ++i) {
      if (coco_ids[i] == 0) {
        // Background should be black
        colors[coco_ids[i]] = Color(0, 0, 0);
      } else {
        colors[coco_ids[i]] = base_colors[(i - 1) % base_colors.size()];
      }
    }
    return colors;
  }();
  return coco_colors;
}

Color get_class_color(int label_id) {
  const auto& color_map = get_coco_colors();
  auto it = color_map.find(label_id);
  if (it != color_map.end()) {
      return it->second;
  }
  // Default color if class not found (shouldn't happen with valid COCO IDs)
  return Color(128, 128, 128); // Gray
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

      // Get class-specific color
      int label_id = detections.labels[i];
      Color class_color = get_class_color(label_id);
      cv::Scalar box_color = class_color.toScalar();

      // Draw bounding box with class-specific color
      cv::rectangle(image_for_plot, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);

      // Get class name
      std::string class_name = get_class_name(label_id);

      // Create label text
      std::string label_text = class_name + ": " +
        std::to_string(detections.scores[i]).substr(0, 5);

      // Calculate text size for background rectangle
      int baseline = 0;
      cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
        0.55, 1.8, &baseline);

      // Draw background rectangle for text (same color as box)
      cv::rectangle(image_for_plot,
        cv::Point(x1, y1 - text_size.height - 4),
        cv::Point(x1 + text_size.width, y1),
        box_color, -1);

      // Choose text color based on background brightness
      // Use white text on dark backgrounds, black text on light backgrounds
      int brightness = (class_color.r + class_color.g + class_color.b) / 3;
      cv::Scalar text_color = brightness < 128 ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);

      // Draw text
      cv::putText(image_for_plot, label_text, cv::Point(x1, y1 - 4),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1.8);
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
      std::cerr << "OpenCV NMS failed with error: " << std::endl;
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
