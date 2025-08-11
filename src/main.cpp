#include <iostream>
#include <string>

// OpenCV includes
#include <opencv2/opencv.hpp>

// local header files
#include "fcos_trt_backend/fcos_trt_backend.hpp"
#include "fcos_trt_backend/post_processor.hpp"
#include "fcos_trt_backend/exception.hpp"


int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
    std::cerr << "Example: ./fcos_inference engines/fcos_resnet50_fpn_374x1238.engine test/image_000.png" << std::endl;
    return -1;
  }

  std::string engine_path = argv[1];
  std::string image_path = argv[2];

  try {
    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Error: Cannot load image: " << image_path << std::endl;
      return -1;
    }

    cv::Size original_size(image.cols, image.rows);
    std::cout << "Loaded image: " << image_path << " (height, width) = (" << image.rows << ", " << image.cols << ")" << std::endl;

    // Initialize FCOS TensorRT inference
    fcos_trt_backend::FCOSTrtBackend::Config trt_config;
    fcos_trt_backend::FCOSTrtBackend fcos_trt(engine_path, trt_config);

    // Initialize post-processor
    fcos_trt_backend::FCOSPostProcessor::Config postprocess_config;
    postprocess_config.score_thresh = 0.2f;
    postprocess_config.nms_thresh = 0.6f;
    postprocess_config.detections_per_img = 100;
    postprocess_config.topk_candidates = 1000;
    fcos_trt_backend::FCOSPostProcessor postprocessor(postprocess_config);

    // Run inference
    std::cout << "\nRunning TensorRT inference..." << std::endl;
    auto raw_results = fcos_trt.infer(image);

    // Print raw results
    fcos_trt.print_results(raw_results);

    // Run post-processing
    std::cout << "\nRunning post-processing..." << std::endl;
    cv::Size processed_size(trt_config.width, trt_config.height);
    auto final_results = postprocessor.postprocess(raw_results, original_size, processed_size);

    // Print final detection results
    std::cout << "\n=== FINAL DETECTION RESULTS ===" << std::endl;
    std::cout << "Number of detections: " << final_results.detections.size() << std::endl;

    for (size_t i = 0; i < final_results.detections.size(); ++i) {
      const auto& detection = final_results.detections[i];
      std::cout << "Detection " << (i + 1) << ":" << std::endl;
      std::cout << "  Class ID: " << detection.label << std::endl;
      std::cout << "  Score: " << detection.score << std::endl;
      std::cout << "  Box: [" << detection.box.x << ", " << detection.box.y
        << ", " << (detection.box.x + detection.box.width)
        << ", " << (detection.box.y + detection.box.height) << "]" << std::endl;
    }

    std::cout << "\n✓ Inference and post-processing completed successfully!" << std::endl;

  } catch (const fcos_trt_backend::TensorRTException& e) {
    std::cerr << "TensorRT Error: " << e.what() << std::endl;
    return -1;
  } catch (const fcos_trt_backend::CudaException& e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
