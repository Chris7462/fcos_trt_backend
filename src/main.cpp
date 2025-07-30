#include <iostream>
#include <string>

// OpenCV includes
#include <opencv2/opencv.hpp>

// local header files
#include "fcos_trt_backend/fcos_trt_backend.hpp"
#include "fcos_trt_backend/exception.hpp"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
    std::cerr << "Example: ./fcos_inference engines/fcos_resnet50_fpn_374x1238.engine script/image_000.png" << std::endl;
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

    std::cout << "Loaded image: " << image_path << " (height, width) = (" << image.rows << ", " << image.cols << ")" << std::endl;

    // Initialize FCOS TensorRT inference
    fcos_trt_backend::FCOSTrtBackend fcos_trt(engine_path);

    // Run inference
    std::cout << "\nRunning inference..." << std::endl;
    auto results = fcos_trt.infer(image);

    // Print results
    fcos_trt.print_results(results);

    std::cout << "\n✓ Inference completed successfully!" << std::endl;

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
