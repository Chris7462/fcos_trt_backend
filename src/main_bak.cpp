#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only print warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class FCOSTensorRT {
private:
    Logger logger;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;

    // Input/Output bindings
    void* buffers[4]; // input + 3 outputs
    int input_index;
    int cls_logits_index;
    int bbox_regression_index;
    int bbox_ctrness_index;

    // Input dimensions
    int input_height;
    int input_width;
    int input_channels;

    // Output dimensions (will be determined at runtime)
    std::vector<int> cls_logits_dims;
    std::vector<int> bbox_regression_dims;
    std::vector<int> bbox_ctrness_dims;

    cudaStream_t stream;

public:
    FCOSTensorRT() : engine(nullptr), context(nullptr), stream(nullptr) {
        // Initialize buffers to nullptr
        for (int i = 0; i < 4; i++) {
            buffers[i] = nullptr;
        }
    }

    ~FCOSTensorRT() {
        cleanup();
    }

    void cleanup() {
        // Free CUDA buffers
        for (int i = 0; i < 4; i++) {
            if (buffers[i]) {
                cudaFree(buffers[i]);
                buffers[i] = nullptr;
            }
        }

        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    bool loadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Error: Cannot open engine file: " << engine_path << std::endl;
            return false;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Read engine data
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();

        // Create TensorRT runtime
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
            return false;
        }

        // Deserialize engine
        engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
        if (!engine) {
            std::cerr << "Error: Failed to deserialize engine" << std::endl;
            return false;
        }

        // Create execution context
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Error: Failed to create execution context" << std::endl;
            return false;
        }

        // Create CUDA stream
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            std::cerr << "Error: Failed to create CUDA stream" << std::endl;
            return false;
        }

        // Get binding indices and dimensions
        if (!setupBindings()) {
            return false;
        }

        // Allocate GPU memory
        if (!allocateBuffers()) {
            return false;
        }

        std::cout << "✓ TensorRT engine loaded successfully" << std::endl;
        return true;
    }

private:
    bool setupBindings() {
        int num_bindings = engine->getNbIOTensors();
        std::cout << "Number of bindings: " << num_bindings << std::endl;

        for (int i = 0; i < num_bindings; i++) {
            const char* name = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(name);
            auto shape = engine->getTensorShape(name);

            std::cout << "Binding " << i << ": " << name;
            std::cout << " Shape: (";
            for (int j = 0; j < shape.nbDims; j++) {
                std::cout << shape.d[j];
                if (j < shape.nbDims - 1) std::cout << ", ";
            }
            std::cout << ")";
            std::cout << " Mode: " << (mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;

            if (mode == TensorIOMode::kINPUT) {
                input_index = i;
                // Handle different input shapes
                if (shape.nbDims == 4) {
                    // Batch format: (N, C, H, W)
                    input_ = shape.d[1];
                    input_height = shape.d[2];
                    input_width = shape.d[3];
                } else if (shape.nbDims == 3) {
                    // No batch dimension: (C, H, W)
                    input_ = shape.d[0];
                    input_height = shape.d[1];
                    input_width = shape.d[2];
                } else {
                    std::cerr << "Error: Unexpected input shape dimensions: " << shape.nbDims << std::endl;
                    return false;
                }
            } else {
                std::string tensor_name(name);
                if (tensor_name == "cls_logits") {
                    cls_logits_index = i;
                    for (int j = 0; j < shape.nbDims; j++) {
                        cls_logits_dims.push_back(shape.d[j]);
                    }
                } else if (tensor_name == "bbox_regression") {
                    bbox_regression_index = i;
                    for (int j = 0; j < shape.nbDims; j++) {
                        bbox_regression_dims.push_back(shape.d[j]);
                    }
                } else if (tensor_name == "bbox_ctrness") {
                    bbox_ctrness_index = i;
                    for (int j = 0; j < shape.nbDims; j++) {
                        bbox_ctrness_dims.push_back(shape.d[j]);
                    }
                }
            }
        }

        std::cout << "Input dimensions: " << input_channels << "x" << input_height << "x" << input_width << std::endl;
        return true;
    }

    bool allocateBuffers() {
        // Calculate sizes
        size_t input_size = input_channels * input_height * input_width * sizeof(float);

        size_t cls_size = 1;
        for (int dim : cls_logits_dims) cls_size *= dim;
        cls_size *= sizeof(float);

        size_t bbox_size = 1;
        for (int dim : bbox_regression_dims) bbox_size *= dim;
        bbox_size *= sizeof(float);

        size_t ctr_size = 1;
        for (int dim : bbox_ctrness_dims) ctr_size *= dim;
        ctr_size *= sizeof(float);

        std::cout << "Buffer sizes:" << std::endl;
        std::cout << "  Input: " << input_size << " bytes (" << input_channels << "x" << input_height << "x" << input_width << ")" << std::endl;
        std::cout << "  cls_logits: " << cls_size << " bytes" << std::endl;
        std::cout << "  bbox_regression: " << bbox_size << " bytes" << std::endl;
        std::cout << "  bbox_ctrness: " << ctr_size << " bytes" << std::endl;

        // Allocate GPU memory
        if (cudaMalloc(&buffers[input_index], input_size) != cudaSuccess) {
            std::cerr << "Error: Failed to allocate input buffer" << std::endl;
            return false;
        }

        if (cudaMalloc(&buffers[cls_logits_index], cls_size) != cudaSuccess) {
            std::cerr << "Error: Failed to allocate cls_logits buffer" << std::endl;
            return false;
        }

        if (cudaMalloc(&buffers[bbox_regression_index], bbox_size) != cudaSuccess) {
            std::cerr << "Error: Failed to allocate bbox_regression buffer" << std::endl;
            return false;
        }

        if (cudaMalloc(&buffers[bbox_ctrness_index], ctr_size) != cudaSuccess) {
            std::cerr << "Error: Failed to allocate bbox_ctrness buffer" << std::endl;
            return false;
        }

        std::cout << "✓ GPU buffers allocated successfully" << std::endl;
        return true;
    }

public:
    cv::Mat preprocessImage(const cv::Mat& image) {
        cv::Mat processed;

        // Convert BGR to RGB
        cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);

        // Resize to input dimensions
        cv::resize(processed, processed, cv::Size(input_width, input_height));

        // Convert to float and normalize to [0, 1]
        processed.convertTo(processed, CV_32F, 1.0 / 255.0);

        return processed;
    }

    bool runInference(const cv::Mat& image) {
        // Preprocess image
        cv::Mat processed_image = preprocessImage(image);

        // Convert to CHW format and copy to GPU
        std::vector<float> input_data(input_channels * input_height * input_width);

        // OpenCV image is HWC, we need CHW
        for (int c = 0; c < input_channels; c++) {
            for (int h = 0; h < input_height; h++) {
                for (int w = 0; w < input_width; w++) {
                    int chw_idx = c * input_height * input_width + h * input_width + w;
                    input_data[chw_idx] = processed_image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // Copy input data to GPU
        if (cudaMemcpyAsync(buffers[input_index], input_data.data(),
                           input_data.size() * sizeof(float),
                           cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            std::cerr << "Error: Failed to copy input data to GPU" << std::endl;
            return false;
        }

        // Set tensor addresses for the new TensorRT API
        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* name = engine->getIOTensorName(i);
            context->setTensorAddress(name, buffers[i]);
        }

        // Run inference
        if (!context->enqueueV3(stream)) {
            std::cerr << "Error: Failed to run inference" << std::endl;
            return false;
        }

        // Wait for completion
        cudaStreamSynchronize(stream);

        return true;
    }

    void printResults() {
        // Calculate output sizes
        size_t cls_size = 1;
        for (int dim : cls_logits_dims) cls_size *= dim;

        size_t bbox_size = 1;
        for (int dim : bbox_regression_dims) bbox_size *= dim;

        size_t ctr_size = 1;
        for (int dim : bbox_ctrness_dims) ctr_size *= dim;

        // Allocate host memory for results
        std::vector<float> cls_logits(cls_size);
        std::vector<float> bbox_regression(bbox_size);
        std::vector<float> bbox_ctrness(ctr_size);

        // Copy results from GPU to CPU
        cudaMemcpy(cls_logits.data(), buffers[cls_logits_index],
                   cls_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bbox_regression.data(), buffers[bbox_regression_index],
                   bbox_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(bbox_ctrness.data(), buffers[bbox_ctrness_index],
                   ctr_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Print results
        std::cout << "\n=== INFERENCE RESULTS ===" << std::endl;

        std::cout << "\nCLS_LOGITS (first 20 values):" << std::endl;
        for (int i = 0; i < std::min(20, (int)cls_size); i++) {
            std::cout << cls_logits[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << std::endl;
        }
        if (cls_size > 20) std::cout << "... (total size: " << cls_size << ")" << std::endl;

        std::cout << "\nBBOX_REGRESSION (first 20 values):" << std::endl;
        for (int i = 0; i < std::min(20, (int)bbox_size); i++) {
            std::cout << bbox_regression[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << std::endl;
        }
        if (bbox_size > 20) std::cout << "... (total size: " << bbox_size << ")" << std::endl;

        std::cout << "\nBBOX_CTRNESS (first 20 values):" << std::endl;
        for (int i = 0; i < std::min(20, (int)ctr_size); i++) {
            std::cout << bbox_ctrness[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << std::endl;
        }
        if (ctr_size > 20) std::cout << "... (total size: " << ctr_size << ")" << std::endl;

        // Print statistics
        auto calc_stats = [](const std::vector<float>& data, const std::string& name) {
            float min_val = *std::min_element(data.begin(), data.end());
            float max_val = *std::max_element(data.begin(), data.end());
            float sum = std::accumulate(data.begin(), data.end(), 0.0f);
            float mean = sum / data.size();

            std::cout << "\n" << name << " Statistics:" << std::endl;
            std::cout << "  Min: " << min_val << std::endl;
            std::cout << "  Max: " << max_val << std::endl;
            std::cout << "  Mean: " << mean << std::endl;
            std::cout << "  Size: " << data.size() << std::endl;
        };

        calc_stats(cls_logits, "CLS_LOGITS");
        calc_stats(bbox_regression, "BBOX_REGRESSION");
        calc_stats(bbox_ctrness, "BBOX_CTRNESS");
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        std::cerr << "Example: ./fcos_inference engines/fcos_resnet50_fpn_374x1238.engine script/image_000.png" << std::endl;
        return -1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];

    // Load test image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Loaded image: " << image_path << " (height, width) = (" << image.rows << ", " << image.cols << ")" << std::endl;

    // Initialize TensorRT inference
    FCOSTensorRT fcos_trt;

    if (!fcos_trt.loadEngine(engine_path)) {
        std::cerr << "Error: Failed to load TensorRT engine" << std::endl;
        return -1;
    }

    // Run inference
    std::cout << "\nRunning inference..." << std::endl;
    if (!fcos_trt.runInference(image)) {
        std::cerr << "Error: Inference failed" << std::endl;
        return -1;
    }

    // Print results
    fcos_trt.printResults();

    std::cout << "\n✓ Inference completed successfully!" << std::endl;
    return 0;
}
