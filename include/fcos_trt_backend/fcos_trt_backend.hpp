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
     * @brief Number of warmup iterations before timing starts
     * @details This is used to ensure that the CUDA kernels and GPU resources are properly initialized
     * and cached before actual inference timing begins. This helps to avoid cold start penalties.
     * - The first iteration initializes CUDA kernels and allocates any lazy GPU resources.
     * - The second iteration ensures everything is properly warmed up and gives more consistent timing.
     * - Set to 0 to disable warmup iterations.
     */
    int warmup_iterations;

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
    : height(374), width(1238), warmup_iterations(2),
      log_level(Logger::Severity::kWARNING) {}
  };

  struct HeadOutputs
  {
    std::vector<float> cls_logits;
    std::vector<float> bbox_regression;
    std::vector<float> bbox_ctrness;
    std::vector<float> anchors;
    std::vector<int64_t> image_sizes;
    std::vector<int64_t> num_anchors_per_level;
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
   * @brief Run inference on input image and return head outputs from the model
   * @param image Input image
   * @return Head outputs containing cls_logits, bbox_regression, and bbox_ctrness
   */
  HeadOutputs infer(const cv::Mat & image);

  // Utility function to print results
  //void print_results(const HeadOutputs & results);

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
  cv::Mat preprocess_image(const cv::Mat & image) const;

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
  std::string anchors_name_;
  std::string image_sizes_name_;
  std::string num_anchors_per_level_name_;

  size_t input_size_;
  size_t cls_logits_size_;
  size_t bbox_regression_size_;
  size_t bbox_ctrness_size_;
  size_t anchors_size_;
  size_t image_sizes_size_;
  size_t num_anchors_per_level_size_;

  // Memory buffers
  struct MemoryBuffers
  {
    float * pinned_input;
    float * device_input;
    float * device_cls_logits;
    float * device_bbox_regression;
    float * device_bbox_ctrness;
    float * device_anchors;
    int64_t * device_image_sizes;
    int64_t * device_num_anchors_per_level;
    float * device_temp_buffer; // For img preprocessing

    MemoryBuffers()
    : pinned_input(nullptr), device_input(nullptr),
      device_cls_logits(nullptr), device_bbox_regression(nullptr),
      device_bbox_ctrness(nullptr), device_anchors(nullptr),
      device_image_sizes(nullptr), device_num_anchors_per_level(nullptr),
      device_temp_buffer(nullptr) {}
  } buffers_;

  // CUDA stream
  cudaStream_t stream_;
};

} // namespace fcos_trt_backend
