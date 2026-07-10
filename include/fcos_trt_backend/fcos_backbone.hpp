#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <memory>
#include <mutex>
#include <string>

// OpenCV includes
#include <opencv2/core.hpp>

// Local header
#include "fcos_trt_backend/fcos_types.hpp"


namespace fcos_trt_backend
{

// Public log level enum, decoupled from nvinfer1::ILogger::Severity.
// Values match the existing "log_level" ROS2 parameter convention
// (0: Internal Error, 1: Error, 2: Warning, 3: Info, 4: Verbose).
enum class LogLevel
{
  kInternalError = 0,
  kError = 1,
  kWarning = 2,
  kInfo = 3,
  kVerbose = 4
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
    LogLevel log_level;

    /**
     * @brief Default constructor
     * @details Initializes the configuration with default values.
     */
    Config()
    : height(374), width(1238), warmup_iterations(2),
      log_level(LogLevel::kWarning) {}
  };

  // Constructor with configuration
  explicit FCOSTrtBackend(const std::string & engine_path, const Config & config = Config());

  // Destructor (defined in .cpp: required since Impl is incomplete here)
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

private:
  // Configuration (plain data - no TensorRT/CUDA types, safe to keep as a direct member)
  Config config_;

  // Opaque implementation - hides TensorRT (NvInfer.h) and CUDA (cuda_runtime.h)
  // types from consumers of this header.
  class Impl;
  std::unique_ptr<Impl> impl_;

  // Thread safety
  mutable std::mutex infer_mutex_;
};

} // namespace fcos_trt_backend
