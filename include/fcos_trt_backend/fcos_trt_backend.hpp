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

// Local includes
#include "fcos_trt_backend/types.hpp"
#include "fcos_trt_backend/postprocessor.hpp"

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

// Optimized FCOS TensorRT inference class
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
     * @brief Number of object classes (COCO: 80, Pascal VOC: 20)
     * @details This should match the number of classes in your model.
     * Note: This is the number of object classes, not including background.
     */
    int num_classes;

    /**
     * @brief Number of warmup iterations before timing starts
     * @details This is used to ensure that the CUDA kernels and GPU resources are properly initialized
     * and cached before actual inference timing begins. This helps to avoid cold start penalties.
     */
    int warmup_iterations;

    /**
     * @brief Log level for TensorRT messages
     */
    Logger::Severity log_level;

    /**
     * @brief Post-processing configuration
     */
    FCOSPostProcessor::Config postprocess_config;

    /**
     * @brief Default constructor
     * @details Initializes the configuration with default values for COCO dataset.
     */
    Config()
    : height(374), width(1238), num_classes(80), warmup_iterations(2),
      log_level(Logger::Severity::kWARNING) {}
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
   * @brief Perform object detection on input image
   * @param image Input image (BGR format)
   * @return Vector of detected objects with bounding boxes, scores, and class labels
   */
  Detections detect(const cv::Mat & image);

  /**
   * @brief Create visualization of detections on the original image
   * @param image Original image
   * @param detections Detection results
   * @param confidence_threshold Minimum confidence to display (default: 0.5)
   * @param draw_labels Whether to draw class labels (default: true)
   * @return Image with drawn bounding boxes and labels
   */
  static cv::Mat visualize_detections(
    const cv::Mat & image,
    const Detections & detections,
    float confidence_threshold = 0.5f,
    bool draw_labels = true);

private:
  // Initialization methods
  void initialize_engine(const std::string & engine_path);
  void find_tensor_names();
  void initialize_memory();
  void initialize_streams();
  void initialize_constants();
  void warmup_engine();

  // Memory management
  void cleanup() noexcept;

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  void preprocess_image(const cv::Mat & image, float * output, cudaStream_t stream) const;

  // Extract raw outputs from TensorRT buffers
  RawOutputs extract_raw_outputs() const;

private:
  // Configuration
  Config config_;

  // TensorRT objects
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // Post-processor
  std::unique_ptr<FCOSPostProcessor> postprocessor_;

  // Tensor information
  struct TensorInfo
  {
    std::string cls_logits_name;
    std::string bbox_regression_name;
    std::string bbox_ctrness_name;
    std::string anchors_name;
    std::string image_sizes_name;
    std::string original_image_sizes_name;
    std::string num_anchors_per_level_name;
    std::string input_name;
  } tensor_names_;

  // Memory sizes
  struct MemorySizes
  {
    size_t input_size;
    size_t cls_logits_size;
    size_t bbox_regression_size;
    size_t bbox_ctrness_size;
    size_t anchors_size;
    size_t image_sizes_size;
    size_t original_image_sizes_size;
    size_t num_anchors_per_level_size;

    MemorySizes() : input_size(0), cls_logits_size(0), bbox_regression_size(0),
                    bbox_ctrness_size(0), anchors_size(0), image_sizes_size(0),
                    original_image_sizes_size(0), num_anchors_per_level_size(0) {}
  } memory_sizes_;

  // Memory buffers
  struct MemoryBuffers
  {
    // Host pinned memory
    float * pinned_input;

    // Device memory - input
    float * device_input;
    float * device_temp_buffer; // For preprocessing

    // Device memory - outputs
    float * device_cls_logits;
    float * device_bbox_regression;
    float * device_bbox_ctrness;
    float * device_anchors;
    int32_t * device_image_sizes;
    int32_t * device_original_image_sizes;
    int32_t * device_num_anchors_per_level;

    // Host memory for output copying
    float * host_cls_logits;
    float * host_bbox_regression;
    float * host_bbox_ctrness;
    float * host_anchors;
    int32_t * host_image_sizes;
    int32_t * host_original_image_sizes;
    int32_t * host_num_anchors_per_level;

    MemoryBuffers() : pinned_input(nullptr), device_input(nullptr),
                      device_temp_buffer(nullptr), device_cls_logits(nullptr),
                      device_bbox_regression(nullptr), device_bbox_ctrness(nullptr),
                      device_anchors(nullptr), device_image_sizes(nullptr),
                      device_original_image_sizes(nullptr), device_num_anchors_per_level(nullptr),
                      host_cls_logits(nullptr), host_bbox_regression(nullptr),
                      host_bbox_ctrness(nullptr), host_anchors(nullptr),
                      host_image_sizes(nullptr), host_original_image_sizes(nullptr),
                      host_num_anchors_per_level(nullptr) {}
  } buffers_;

  // CUDA streams for pipelining
  cudaStream_t stream_;

  // Model dimensions (determined at runtime)
  int num_anchors_;
  int num_levels_;
};

} // namespace fcos_trt_backend
