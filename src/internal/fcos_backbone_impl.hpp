#pragma once

// C++ standard library includes
#include <memory>
#include <string>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// TensorRT includes
#include <NvInfer.h>

// OpenCV includes
#include <opencv2/core.hpp>

// local headers
#include "fcos_trt_backend/fcos_backbone.hpp"
#include "internal/trt_logger.hpp"
#include "internal/cuda_raii.hpp"

// NOTE: This is a private implementation header. It lives under src/internal/
// and is never installed - it must not be included by any consumer of the
// fcos_trt_backend public API. It completes the FCOSTrtBackend::Impl type that
// the public header only forward-declares.

namespace fcos_trt_backend
{

// FCOSTrtBackend::Impl - holds everything that needs NvInfer.h / cuda_runtime.h.
// Public surface is intentionally narrow: initialize() and infer() are the
// only two operations FCOSTrtBackend needs to drive. Everything else -
// engine setup, memory allocation, warmup, cleanup - is an internal
// implementation detail and stays private.
class FCOSTrtBackend::Impl
{
public:
  ~Impl();

  void initialize(const std::string & engine_path, const FCOSTrtBackend::Config & config);
  HeadOutputs infer(const cv::Mat & image, const FCOSTrtBackend::Config & config);

private:
  // Initialization steps
  void initialize_engine(const std::string & engine_path, const FCOSTrtBackend::Config & config);
  void find_tensor_names();
  void initialize_memory(const FCOSTrtBackend::Config & config);
  void initialize_streams();
  void warmup_engine(const FCOSTrtBackend::Config & config);

  // Helper methods
  std::vector<uint8_t> load_engine_file(const std::string & engine_path) const;
  cv::Mat preprocess_image(const cv::Mat & image, const FCOSTrtBackend::Config & config) const;

private:
  // TensorRT objects
  std::unique_ptr<internal::Logger> logger_;
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

  size_t input_size_ = 0;
  size_t cls_logits_size_ = 0;
  size_t bbox_regression_size_ = 0;
  size_t bbox_ctrness_size_ = 0;
  size_t anchors_size_ = 0;
  size_t image_sizes_size_ = 0;
  size_t num_anchors_per_level_size_ = 0;

  // Memory buffers. Ownership via RAII smart pointers (see cuda_raii.hpp) -
  // no manual cleanup() bookkeeping needed; destruction order below (and
  // stream_ declared last) ensures buffers_ is torn down before the stream
  // that operations on it were queued against.
  struct MemoryBuffers
  {
    internal::DevPtr device_input;
    internal::DevPtr device_cls_logits;
    internal::DevPtr device_bbox_regression;
    internal::DevPtr device_bbox_ctrness;
    internal::DevPtr device_anchors;
    internal::DevPtr device_image_sizes;
    internal::DevPtr device_num_anchors_per_level;
  } buffers_;

  // CUDA stream
  internal::StreamPtr stream_;
};

} // namespace fcos_trt_backend
