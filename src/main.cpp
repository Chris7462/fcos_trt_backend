#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

// Local includes
#include "fcos_trt_backend/fcos_trt_backend.hpp"
#include "fcos_trt_backend/post_processor.hpp"
#include "fcos_trt_backend/exception.hpp"

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

    // Initialize TensorRT backend
    fcos_trt_backend::FCOSTrtBackend::Config config;
    // Use default config values (374x1238)

    fcos_trt_backend::FCOSTrtBackend backend(engine_path, config);

    // Initialize postprocessor
    fcos_trt_backend::FCOSPostProcessor::Config post_config;
    post_config.score_thresh = 0.2f;
    post_config.nms_thresh = 0.6f;
    post_config.detections_per_img = 100;
    post_config.topk_candidates = 1000;

    fcos_trt_backend::FCOSPostProcessor postprocessor(post_config);

    // Run inference
    auto raw_outputs = backend.infer(image);

    // Print raw inference results
    //backend.print_results(raw_outputs);

    // Run postprocessing
    auto detection_results = postprocessor.postprocess_detections(raw_outputs);

    // Print postprocessed results
    //postprocessor.print_detection_results(detection_results, 20);

  //// Print first 20 values for verification (matching Python script)
  //std::cout << "\n=== First 20 Values for Verification ===" << std::endl;

  //std::cout << "Boxes (first 20):" << std::endl;
  //int box_count = std::min(20, static_cast<int>(detection_results.boxes.size()));
  //for (int i = 0; i < box_count; ++i) {
  //  const auto& box = detection_results.boxes[i];
  //  std::cout << "[" << box.x << ", " << box.y << ", "
  //    << (box.x + box.width) << ", " << (box.y + box.height) << "] ";
  //  if ((i + 1) % 4 == 0) std::cout << std::endl;
  //}
  //std::cout << std::endl;

  //std::cout << "Scores (first 20):" << std::endl;
  //int score_count = std::min(20, static_cast<int>(detection_results.scores.size()));
  //for (int i = 0; i < score_count; ++i) {
  //  std::cout << detection_results.scores[i] << " ";
  //  if ((i + 1) % 10 == 0) std::cout << std::endl;
  //}
  //std::cout << std::endl;

  //std::cout << "Labels (first 20):" << std::endl;
  //int label_count = std::min(20, static_cast<int>(detection_results.labels.size()));
  //for (int i = 0; i < label_count; ++i) {
  //  std::cout << detection_results.labels[i] << " ";
  //  if ((i + 1) % 10 == 0) std::cout << std::endl;
  //}
  //std::cout << std::endl;

  //std::cout << "\n=== Demo completed successfully! ===" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
