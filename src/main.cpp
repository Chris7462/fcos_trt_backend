#include <iostream>
#include <chrono>

#include "fcos_trt_backend/fcos_trt_backend.hpp"


void printOutputShapes(const fcos_trt_backend::FCOSTrtBackend::FCOSOutputs& outputs) {
    std::cout << "\n=== Output Tensor Shapes ===" << std::endl;

    std::cout << "cls_logits: [";
    for (size_t i = 0; i < outputs.cls_logits_dims.size(); ++i) {
        std::cout << outputs.cls_logits_dims[i] << (i < outputs.cls_logits_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.cls_logits.size() << " elements" << std::endl;

    std::cout << "bbox_regression: [";
    for (size_t i = 0; i < outputs.bbox_regression_dims.size(); ++i) {
        std::cout << outputs.bbox_regression_dims[i] << (i < outputs.bbox_regression_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.bbox_regression.size() << " elements" << std::endl;

    std::cout << "bbox_ctrness: [";
    for (size_t i = 0; i < outputs.bbox_ctrness_dims.size(); ++i) {
        std::cout << outputs.bbox_ctrness_dims[i] << (i < outputs.bbox_ctrness_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.bbox_ctrness.size() << " elements" << std::endl;

    std::cout << "anchors: [";
    for (size_t i = 0; i < outputs.anchors_dims.size(); ++i) {
        std::cout << outputs.anchors_dims[i] << (i < outputs.anchors_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.anchors.size() << " elements" << std::endl;

    std::cout << "image_sizes: [";
    for (size_t i = 0; i < outputs.image_sizes_dims.size(); ++i) {
        std::cout << outputs.image_sizes_dims[i] << (i < outputs.image_sizes_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.image_sizes.size() << " elements" << std::endl;

    std::cout << "original_image_sizes: [";
    for (size_t i = 0; i < outputs.original_image_sizes_dims.size(); ++i) {
        std::cout << outputs.original_image_sizes_dims[i] << (i < outputs.original_image_sizes_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.original_image_sizes.size() << " elements" << std::endl;

    std::cout << "num_anchors_per_level: [";
    for (size_t i = 0; i < outputs.num_anchors_per_level_dims.size(); ++i) {
        std::cout << outputs.num_anchors_per_level_dims[i] << (i < outputs.num_anchors_per_level_dims.size() - 1 ? ", " : "");
    }
    std::cout << "] - " << outputs.num_anchors_per_level.size() << " elements" << std::endl;
}

void printSampleOutputValues(const fcos_trt_backend::FCOSTrtBackend::FCOSOutputs& outputs) {
    std::cout << "\n=== Sample Output Values ===" << std::endl;

    // Print first few values of each output tensor
    constexpr int sample_size = 5;

    std::cout << "cls_logits (first " << sample_size << " values): ";
    for (int i = 0; i < std::min(sample_size, static_cast<int>(outputs.cls_logits.size())); ++i) {
        std::cout << outputs.cls_logits[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "bbox_regression (first " << sample_size << " values): ";
    for (int i = 0; i < std::min(sample_size, static_cast<int>(outputs.bbox_regression.size())); ++i) {
        std::cout << outputs.bbox_regression[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "bbox_ctrness (first " << sample_size << " values): ";
    for (int i = 0; i < std::min(sample_size, static_cast<int>(outputs.bbox_ctrness.size())); ++i) {
        std::cout << outputs.bbox_ctrness[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "anchors (first " << sample_size << " values): ";
    for (int i = 0; i < std::min(sample_size, static_cast<int>(outputs.anchors.size())); ++i) {
        std::cout << outputs.anchors[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "image_sizes: ";
    for (size_t i = 0; i < outputs.image_sizes.size(); ++i) {
        std::cout << outputs.image_sizes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "original_image_sizes: ";
    for (size_t i = 0; i < outputs.original_image_sizes.size(); ++i) {
        std::cout << outputs.original_image_sizes[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "num_anchors_per_level: ";
    for (size_t i = 0; i < outputs.num_anchors_per_level.size(); ++i) {
        std::cout << outputs.num_anchors_per_level[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " engines/fcos_resnet50_fpn_374x1238.engine test_image.jpg" << std::endl;
        return -1;
    }

    const std::string engine_path = argv[1];
    const std::string image_path = argv[2];

    std::cout << "=== FCOS TensorRT Inference Test ===" << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Image: " << image_path << std::endl;

    try {
        // Configure the inferencer
        fcos_trt_backend::FCOSTrtBackend::Config config;
        config.height = 374;
        config.width = 1238;
        config.warmup_iterations = 2;
        config.log_level = fcos_trt_backend::Logger::Severity::kINFO;

        // Initialize TensorRT engine
        std::cout << "\nInitializing FCOS TensorRT backend..." << std::endl;
        auto fcos_backend = std::make_unique<fcos_trt_backend::FCOSTrtBackend>(engine_path, config);
        std::cout << "✓ Engine loaded successfully" << std::endl;

        // Load test image
        std::cout << "\nLoading test image..." << std::endl;
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        std::cout << "✓ Image loaded successfully" << std::endl;
        std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;
        //std::cout << "Model input size: " << fcos_backend->getInputWidth() << "x" << fcos_backend->getInputHeight() << std::endl;

        // Run inference
        std::cout << "\nRunning inference..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto outputs = fcos_backend->infer(image);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "✓ Inference completed successfully" << std::endl;
        std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

        // Print output information
        printOutputShapes(outputs);
        printSampleOutputValues(outputs);

        std::cout << "\n=== Test Complete ===" << std::endl;
        std::cout << "Raw FCOS outputs are ready for post-processing!" << std::endl;
        std::cout << "Next step: Implement FCOSPostProcessor in C++ to convert these" << std::endl;
        std::cout << "raw outputs into final detection results (boxes, scores, labels)." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
