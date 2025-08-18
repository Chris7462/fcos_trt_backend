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

  // Initialize COCO category names once
const std::unordered_map<int, std::string>& get_coco_names() {
  static const std::unordered_map<int, std::string> coco_category_names = {
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

  return coco_category_names;
}

// Initialize COCO colors once
const std::unordered_map<int, Color>& get_coco_colors() {
  static const std::unordered_map<int, Color> coco_colors = {
    {0, Color(0, 0, 0)},
    {1, Color(255, 0, 0)}, {2, Color(0, 255, 0)}, {3, Color(0, 0, 255)}, {4, Color(255, 255, 0)},
    {5, Color(255, 0, 255)}, {6, Color(0, 255, 255)}, {7, Color(255, 128, 0)}, {8, Color(128, 0, 255)},
    {9, Color(255, 0, 128)}, {10, Color(128, 255, 0)}, {11, Color(0, 128, 255)}, {13, Color(255, 128, 128)},
    {14, Color(128, 255, 128)}, {15, Color(128, 128, 255)}, {16, Color(255, 192, 0)}, {17, Color(192, 0, 255)},
    {18, Color(0, 192, 255)}, {19, Color(255, 64, 64)}, {20, Color(64, 255, 64)}, {21, Color(64, 64, 255)},
    {22, Color(255, 255, 128)}, {23, Color(255, 128, 255)}, {24, Color(128, 255, 255)}, {25, Color(192, 96, 0)},
    {27, Color(96, 0, 192)}, {28, Color(0, 96, 192)}, {31, Color(224, 32, 32)}, {32, Color(32, 224, 32)},
    {33, Color(32, 32, 224)}, {34, Color(224, 224, 0)}, {35, Color(224, 0, 224)}, {36, Color(0, 224, 224)},
    {37, Color(160, 82, 45)}, {38, Color(255, 20, 147)}, {39, Color(0, 100, 0)}, {40, Color(139, 69, 19)},
    {41, Color(255, 140, 0)}, {42, Color(218, 112, 214)}, {43, Color(30, 144, 255)}, {44, Color(220, 20, 60)},
    {46, Color(255, 69, 0)}, {47, Color(75, 0, 130)}, {48, Color(238, 130, 238)}, {49, Color(255, 215, 0)},
    {50, Color(186, 85, 211)}, {51, Color(147, 112, 219)}, {52, Color(255, 182, 193)}, {53, Color(255, 160, 122)},
    {54, Color(32, 178, 170)}, {55, Color(135, 206, 235)}, {56, Color(119, 136, 153)}, {57, Color(255, 127, 80)},
    {58, Color(240, 128, 128)}, {59, Color(144, 238, 144)}, {60, Color(255, 218, 185)}, {61, Color(205, 92, 92)},
    {62, Color(240, 230, 140)}, {63, Color(123, 104, 238)}, {64, Color(152, 251, 152)}, {65, Color(250, 128, 114)},
    {67, Color(216, 191, 216)}, {70, Color(255, 239, 213)}, {72, Color(255, 228, 181)}, {73, Color(255, 222, 173)},
    {74, Color(245, 222, 179)}, {75, Color(255, 228, 196)}, {76, Color(255, 235, 205)}, {77, Color(245, 245, 220)},
    {78, Color(255, 248, 220)}, {79, Color(255, 250, 205)}, {80, Color(250, 250, 210)}, {81, Color(255, 255, 240)},
    {82, Color(240, 255, 240)}, {84, Color(255, 250, 240)}, {85, Color(253, 245, 230)}, {86, Color(245, 255, 250)},
    {87, Color(112, 128, 144)}, {88, Color(119, 136, 153)}, {89, Color(176, 196, 222)}, {90, Color(230, 230, 250)}
  };

  return coco_colors;
}

std::string get_class_name(int coco_id)
{
  const auto& category_names = get_coco_names();
  auto it = category_names.find(coco_id);
  if (it != category_names.end()) {
    return it->second;
  }
  return "unknown_class_" + std::to_string(coco_id);
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
      std::cerr << "OpenCV NMS failed with error: " << e.what() << std::endl;
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
