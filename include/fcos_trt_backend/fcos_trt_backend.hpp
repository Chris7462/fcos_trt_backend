#pragma once

// C++ standard library includes
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/opencv.hpp>

namespace fcos_trt_backend
{

// Custom exception classes
class TensorRTException : public std::runtime_error
{
public:
  explicit TensorRTException(const std::string & message)
  : std::runtime_error("TensorRT Error: " + message) {}
};

class CudaException : public std::runtime_error
{
public:
  explicit CudaException(const std::string & message, cudaError_t error)
  : std::runtime_error("CUDA Error: " + message + " (" + cudaGetErrorString(error) + ")") {}
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      throw CudaException(#call, error); \
    } \
  } while(0)

// ImageNet normalization constants
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

// TensorRT Logger with configurable severity
class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(Severity min_severity = Severity::kWARNING)
  : min_severity_(min_severity) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= min_severity_) {
      std::cout << "[TensorRT] " << msg << std::endl;
    }
  }

private:
  Severity min_severity_;
};

// FCOS TensorRT inference class
class FCOSTrtBackend
{
public:
  struct Config
  {
    /**
     * @brief Input image height
     */
    int height;

    /**
     * @brief Input image width
     */
    int width;

    /**
     * @brief Number of warmup iterations before timing starts
     */
    int warmup_iterations;

    /**
     * @brief Log level for TensorRT messages
     */
    Logger::Severity log_level;

    /**
     * @brief Default constructor
     */
    Config()
    : height(374), width(1238), warmup_iterations(2),
      log_level(Logger::Severity::kWARNING) {}
  };

  struct FCOSOutputs
  {
    std::vector<float> cls_logits;
    std::vector<float> bbox_regression;
    std::vector<float> bbox_ctrness;
    std::vector<float> anchors;
    std::vector<float> image_sizes;
    std::vector<float> original_image_sizes;
    std::vector<float> num_anchors_per_level;

    // Tensor dimensions for each output
    std::vector<int> cls_logits_dims;
    std::vector<int> bbox_regression_dims;
    std::vector<int> bbox_ctrness_dims;
    std::vector<int> anchors_dims;
    std::vector<int> image_sizes_dims;
    std::vector<int> original_image_sizes_dims;
    std::vector<int> num_anchors_per_level_dims;
  };

  // Constructor with configuration
  explicit FCOSTrtBackend(const std::string & engine_path, const Config & config = Config());

  // Destructor
  ~FCOSTrtBackend();

  // Disable copy and move semantics
  FCOSTrtBackend(const FCOSTrtBackend &) = delete;
  FCOSTrtBackend & operator=(const FCOSTrtBackend &) = delete;
  FCOSTrtBackend(FCOSTrtBackend &&) = delete;
  FCOSTrtBackend & operator=(FCOSTrtBackend &&) = delete;

  // Main inference method
  /**
   * @brief Run FCOS inference and return raw outputs
   * @param image Input image
   * @return FCOSOutputs containing all raw model outputs
   */
  FCOSOutputs infer(const cv::Mat & image);

private:
  // Initialization methods
  void initialize_engine(const std::string & engine_path);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();
  void warmup_engine();

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void preprocess_image(const cv::Mat & image, float * output, cudaStream_t stream) const;

  // Output tensor info
  struct OutputTensorInfo
  {
    std::string name;
    std::vector<int> dims;
    size_t size;
    size_t element_count;
  };

private:
  // Configuration
  Config config_;

  // TensorRT objects
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Tensor information
  std::string input_name_;
  std::vector<OutputTensorInfo> output_tensors_;
  size_t input_size_;

  // Memory buffers
  struct MemoryBuffers
  {
    float * pinned_input;
    float * device_input;
    std::vector<float *> device_outputs;
    std::vector<float *> pinned_outputs;

    MemoryBuffers()
    : pinned_input(nullptr), device_input(nullptr) {}
  } buffers_;

  // CUDA stream for operations
  cudaStream_t stream_;
};

} // namespace fcos_trt_backend
