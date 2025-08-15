#include <iostream>
#include <string>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Local includes
#include "fcos_trt_backend/fcos_backbone.hpp"
#include "fcos_trt_backend/fcos_post_processor.hpp"
#include "fcos_trt_backend/exception.hpp"
#include "fcos_trt_backend/detection_utils.hpp"


int main(int argc, char* argv[])
{
  // Parse command line arguments
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string image_path = argv[2];

  try {
    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Error: Could not load image from " << image_path << std::endl;
      return -1;
    }

    // Store original image dimensions
    int original_height = image.rows;
    int original_width = image.cols;

    // Initialize TensorRT backend
    fcos_trt_backend::FCOSBackbone::Config config;
    // Use default config values (374x1238)

    fcos_trt_backend::FCOSBackbone backbone(engine_path, config);

    // Initialize postprocessor
    const float score_thresh = 0.2f;
    const float nms_thresh = 0.6f;
    const int detections_per_img = 100;
    const int topk_candidates = 1000;

    fcos_trt_backend::FCOSPostProcessor postprocessor(score_thresh, nms_thresh, detections_per_img, topk_candidates);

    // Run inference
    auto head_outputs = backbone.infer(image);

    // Print raw inference results
    //backend.print_results(head_outputs);

    // Run postprocessing with original image dimensions
    auto detection_results = postprocessor.postprocess_detections(
      head_outputs,
      original_height,
      original_width
    );

    // Print postprocessed results
    //fcos_trt_backend::utils::print_detection_results(detection_results, 20);

    // Create visualization of the detection results
    fcos_trt_backend::utils::plot_detections(
      image_path,
      detection_results,
      "FCOS TensorRT Detection Results",
      0.5f,  // confidence threshold
      "fcos_trt_detections.png"
    );

    // Print first 20 values for verification (matching Python script)
    // Print first 20 values for verification (matching Python script)
    std::cout << "\n=== First 20 Values for Verification ===" << std::endl;

    std::cout << "Boxes (first 20):" << std::endl;
    int box_count = std::min(20, static_cast<int>(detection_results.boxes.size()));
    for (int i = 0; i < box_count; ++i) {
      const auto& box = detection_results.boxes[i];
      std::cout << "[" << box.x << ", " << box.y << ", "
        << (box.x + box.width) << ", " << (box.y + box.height) << "] ";
      if ((i + 1) % 4 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Scores (first 20):" << std::endl;
    int score_count = std::min(20, static_cast<int>(detection_results.scores.size()));
    for (int i = 0; i < score_count; ++i) {
      std::cout << detection_results.scores[i] << " ";
      if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Labels (first 20):" << std::endl;
    int label_count = std::min(20, static_cast<int>(detection_results.labels.size()));
    for (int i = 0; i < label_count; ++i) {
      std::cout << detection_results.labels[i] << " ";
      if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\n=== Demo completed successfully! ===" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
