#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <memory>
#include <string>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/core.hpp>


namespace fcos_trt_backend
{
// TensorRT Logger with configurable severity
class Logger : public nvinfer1::ILogger
{
public:
  explicit Logger(Severity min_severity = Severity::kWARNING)
  : min_severity_(min_severity) {}

  void log(Severity severity, const char * msg) noexcept override;

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
     * @brief Number of input channels
     */
    int channels;

    /**
     * @brief Log level for TensorRT messages
     * @details This controls the verbosity of TensorRT logging.
     */
    Logger::Severity log_level;

    /**
     * @brief Default constructor
     * @details Initializes the configuration with default values.
     */
    Config()
    : height(374), width(1238), channels(3),
      log_level(Logger::Severity::kWARNING) {}
  };

  struct DetectionResults
  {
    std::vector<float> cls_logits;
    std::vector<float> bbox_regression;
    std::vector<float> bbox_ctrness;
  };

  // Constructor with configuration
  explicit FCOSTrtBackend(const std::string & engine_path, const Config & config = Config());

  // Destructor
  ~FCOSTrtBackend();

  // Disable copy and move semantics - use std::unique_ptr for ownership transfer
  FCOSTrtBackend(const FCOSTrtBackend &) = delete;
  FCOSTrtBackend & operator=(const FCOSTrtBackend &) = delete;
  FCOSTrtBackend(FCOSTrtBackend &&) = delete;
  FCOSTrtBackend & operator=(FCOSTrtBackend &&) = delete;

  // Main inference method
  /**
   * @brief Run inference on input image and return detection results
   * @param image Input image
   * @return Detection results containing cls_logits, bbox_regression, and bbox_ctrness
   */
  DetectionResults infer(const cv::Mat & image);

  // Utility function to print results
  void print_results(const DetectionResults & results);

private:
  // Initialization methods
  void initialize_engine(const std::string & engine_path);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  //cv::Mat preprocess_image(const cv::Mat & image) const;
  void preprocess_image(const cv::Mat & image, float * output, cudaStream_t stream) const;

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
  std::string cls_logits_name_;
  std::string bbox_regression_name_;
  std::string bbox_ctrness_name_;
  size_t input_size_;
  size_t cls_logits_size_;
  size_t bbox_regression_size_;
  size_t bbox_ctrness_size_;

  // Memory buffers
  struct MemoryBuffers
  {
    float * pinned_input;
    float * device_input;
    float * device_cls_logits;
    float * device_bbox_regression;
    float * device_bbox_ctrness;

    MemoryBuffers()
    : pinned_input(nullptr), device_input(nullptr),
      device_cls_logits(nullptr), device_bbox_regression(nullptr),
      device_bbox_ctrness(nullptr) {}
  } buffers_;

  // CUDA stream
  cudaStream_t stream_;
};

} // namespace fcos_trt_backend
