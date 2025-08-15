#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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
  const Detections& results, int max_detections)
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

void plot_detections(
  const std::string& image_path,
  const Detections& detections,
  const std::string& title,
  float confidence_threshold,
  const std::string& output_path)
{
  try {
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Could not load image: " << image_path << std::endl;
      return;
    }

    cv::Mat image_for_plot = image.clone();

    std::cout << "\n=== " << title << " ===" << std::endl;
    int detection_count = 0;

    // Draw predictions with confidence > threshold
    for (size_t i = 0; i < detections.boxes.size(); ++i) {
      if (detections.scores[i] >= confidence_threshold) {
        detection_count++;

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
          0.6, 2, &baseline);

        // Draw background rectangle for text
        cv::rectangle(image_for_plot,
          cv::Point(x1, y1 - text_size.height - 10),
          cv::Point(x1 + text_size.width, y1),
          cv::Scalar(0, 255, 0), -1);

        // Draw text
        cv::putText(image_for_plot, label_text, cv::Point(x1, y1 - 10),
          cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        std::cout << "Detection " << detection_count << ": " << class_name
                  << " (ID: " << label_id << ") - Confidence: "
                  << detections.scores[i] << std::endl;
        std::cout << "  Box: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]"
                  << std::endl;
      }
    }

    std::cout << "Total detections above " << confidence_threshold
              << " confidence: " << detection_count << std::endl;

    // Add title to the image
    cv::putText(image_for_plot, title, cv::Point(30, 50),
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

    // Save the result
    if (cv::imwrite(output_path, image_for_plot)) {
      std::cout << "✓ Detection plot saved as '" << output_path << "'" << std::endl;
    } else {
      std::cerr << "✗ Could not save plot to '" << output_path << "'" << std::endl;
    }

  } catch (const cv::Exception& e) {
    std::cerr << "Error plotting detections: " << e.what() << std::endl;
  }
}

} // namespace utils

} // namespace fcos_trt_backend
