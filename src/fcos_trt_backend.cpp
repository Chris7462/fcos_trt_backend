#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files: This project includes local header files.
#include "fcos_trt_backend/exception.hpp"
#include "fcos_trt_backend/fcos_trt_backend.hpp"


namespace fcos_trt_backend
{

// Logger implementation
void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= min_severity_) {
    const char * severity_str;
    switch (severity) {
      case Severity::kINTERNAL_ERROR: severity_str = "INTERNAL_ERROR"; break;
      case Severity::kERROR: severity_str = "ERROR"; break;
      case Severity::kWARNING: severity_str = "WARNING"; break;
      case Severity::kINFO: severity_str = "INFO"; break;
      case Severity::kVERBOSE: severity_str = "VERBOSE"; break;
      default: severity_str = "UNKNOWN"; break;
    }
    std::cerr << "[TensorRT " << severity_str << "] " << msg << std::endl;
  }
}

// FCOSTrtBackend implementation
FCOSTrtBackend::FCOSTrtBackend(const std::string & engine_path, const Config & config)
: config_(config)
{
  try {
    initialize_engine(engine_path);
    find_tensor_names();
    initialize_memory();
    initialize_streams();
    warmup_engine();
  } catch (const std::exception & e) {
    cleanup();
    throw TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

FCOSTrtBackend::~FCOSTrtBackend()
{
  cleanup();
}

void FCOSTrtBackend::initialize_engine(const std::string & engine_path)
{
  // Initialize logger
  logger_ = std::make_unique<Logger>(config_.log_level);

  auto engine_data = load_engine_file(engine_path);

  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw TensorRTException("Failed to create TensorRT runtime");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw TensorRTException("Failed to deserialize CUDA engine");
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine_->createExecutionContext());
  if (!context_) {
    throw TensorRTException("Failed to create execution context");
  }
}

std::vector<uint8_t> FCOSTrtBackend::load_engine_file(
  const std::string & engine_path) const
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read engine file: " + engine_path);
  }

  return buffer;
}

void FCOSTrtBackend::find_tensor_names()
{
  bool found_input = false;
  bool found_cls_logits = false;
  bool found_bbox_regression = false;
  bool found_bbox_ctrness = false;
  bool found_anchors = false;
  bool found_image_sizes = false;
  bool found_num_anchors_per_level = false;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;
      found_input = true;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      std::string name_str(tensor_name);
      if (name_str == "cls_logits") {
        cls_logits_name_ = tensor_name;
        found_cls_logits = true;
      } else if (name_str == "bbox_regression") {
        bbox_regression_name_ = tensor_name;
        found_bbox_regression = true;
      } else if (name_str == "bbox_ctrness") {
        bbox_ctrness_name_ = tensor_name;
        found_bbox_ctrness = true;
      } else if (name_str == "anchors") {
        anchors_name_ = tensor_name;
        found_anchors = true;
      } else if (name_str == "image_sizes") {
        image_sizes_name_ = tensor_name;
        found_image_sizes = true;
      } else if (name_str == "num_anchors_per_level") {
        num_anchors_per_level_name_ = tensor_name;
        found_num_anchors_per_level = true;
      }
    }
  }

  if (!found_input || !found_cls_logits || !found_bbox_regression ||
      !found_bbox_ctrness || !found_anchors || !found_image_sizes ||
      !found_num_anchors_per_level) {
    throw TensorRTException("Failed to find required input/output tensors");
  }
}

void FCOSTrtBackend::initialize_memory()
{
  // Calculate buffer sizes based on config (not engine shape)
  input_size_ = 1 * 3 * config_.height * config_.width * sizeof(float);

  // For FCOS, we need to get the actual output sizes from the engine
  // since they depend on the FPN levels and anchor points
  auto cls_shape = engine_->getTensorShape(cls_logits_name_.c_str());
  auto bbox_shape = engine_->getTensorShape(bbox_regression_name_.c_str());
  auto ctr_shape = engine_->getTensorShape(bbox_ctrness_name_.c_str());
  auto anchors_shape = engine_->getTensorShape(anchors_name_.c_str());
  auto image_sizes_shape = engine_->getTensorShape(image_sizes_name_.c_str());
  auto num_anchors_shape = engine_->getTensorShape(num_anchors_per_level_name_.c_str());

  cls_logits_size_ = 1;
  for (int i = 0; i < cls_shape.nbDims; ++i) {
    cls_logits_size_ *= cls_shape.d[i];
  }
  cls_logits_size_ *= sizeof(float);

  bbox_regression_size_ = 1;
  for (int i = 0; i < bbox_shape.nbDims; ++i) {
    bbox_regression_size_ *= bbox_shape.d[i];
  }
  bbox_regression_size_ *= sizeof(float);

  bbox_ctrness_size_ = 1;
  for (int i = 0; i < ctr_shape.nbDims; ++i) {
    bbox_ctrness_size_ *= ctr_shape.d[i];
  }
  bbox_ctrness_size_ *= sizeof(float);

  anchors_size_ = 1;
  for (int i = 0; i < anchors_shape.nbDims; ++i) {
    anchors_size_ *= anchors_shape.d[i];
  }
  anchors_size_ *= sizeof(float);

  image_sizes_size_ = 1;
  for (int i = 0; i < image_sizes_shape.nbDims; ++i) {
    image_sizes_size_ *= image_sizes_shape.d[i];
  }
  image_sizes_size_ *= sizeof(int64_t);

  num_anchors_per_level_size_ = 1;
  for (int i = 0; i < num_anchors_shape.nbDims; ++i) {
    num_anchors_per_level_size_ *= num_anchors_shape.d[i];
  }
  num_anchors_per_level_size_ *= sizeof(int64_t);

  // Allocate pinned host memory
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_input, input_size_));

  // Allocate GPU memory
  CUDA_CHECK(cudaMalloc(&buffers_.device_input, input_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_cls_logits, cls_logits_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_bbox_regression, bbox_regression_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_bbox_ctrness, bbox_ctrness_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_anchors, anchors_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_image_sizes, image_sizes_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_num_anchors_per_level, num_anchors_per_level_size_));
  CUDA_CHECK(cudaMalloc(&buffers_.device_temp_buffer, input_size_));

  // Set tensor addresses for the new TensorRT API
  if (!context_->setTensorAddress(input_name_.c_str(),
    static_cast<void *>(buffers_.device_input))) {
    throw TensorRTException("Failed to set input tensor address");
  }
  if (!context_->setTensorAddress(cls_logits_name_.c_str(),
    static_cast<void *>(buffers_.device_cls_logits))) {
    throw TensorRTException("Failed to set cls_logits tensor address");
  }
  if (!context_->setTensorAddress(bbox_regression_name_.c_str(),
    static_cast<void *>(buffers_.device_bbox_regression))) {
    throw TensorRTException("Failed to set bbox_regression tensor address");
  }
  if (!context_->setTensorAddress(bbox_ctrness_name_.c_str(),
    static_cast<void *>(buffers_.device_bbox_ctrness))) {
    throw TensorRTException("Failed to set bbox_ctrness tensor address");
  }
  if (!context_->setTensorAddress(anchors_name_.c_str(),
    static_cast<void *>(buffers_.device_anchors))) {
    throw TensorRTException("Failed to set anchors tensor address");
  }
  if (!context_->setTensorAddress(image_sizes_name_.c_str(),
    static_cast<void *>(buffers_.device_image_sizes))) {
    throw TensorRTException("Failed to set image_sizes tensor address");
  }
  if (!context_->setTensorAddress(num_anchors_per_level_name_.c_str(),
    static_cast<void *>(buffers_.device_num_anchors_per_level))) {
    throw TensorRTException("Failed to set num_anchors_per_level tensor address");
  }
}

void FCOSTrtBackend::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
  if (!stream_) {
    throw TensorRTException("Failed to create CUDA stream");
  }
}

void FCOSTrtBackend::warmup_engine()
{
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input, 0, input_size_, stream_));

  for (int i = 0; i < config_.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_)) {
      throw TensorRTException("Failed to enqueue warmup inference");
    }

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  std::cout << "Engine warmed up with " << config_.warmup_iterations << " iterations" << std::endl;
}

void FCOSTrtBackend::cleanup() noexcept
{
  // Free pinned host memory
  if (buffers_.pinned_input) {
    cudaFreeHost(buffers_.pinned_input);
  }

  // Free device memory
  if (buffers_.device_input) {
    cudaFree(buffers_.device_input);
  }

  if (buffers_.device_cls_logits) {
    cudaFree(buffers_.device_cls_logits);
  }

  if (buffers_.device_bbox_regression) {
    cudaFree(buffers_.device_bbox_regression);
  }

  if (buffers_.device_bbox_ctrness) {
    cudaFree(buffers_.device_bbox_ctrness);
  }

  if (buffers_.device_anchors) {
    cudaFree(buffers_.device_anchors);
  }

  if (buffers_.device_image_sizes) {
    cudaFree(buffers_.device_image_sizes);
  }

  if (buffers_.device_num_anchors_per_level) {
    cudaFree(buffers_.device_num_anchors_per_level);
  }

  if (buffers_.device_temp_buffer) {
    cudaFree(buffers_.device_temp_buffer);
  }

  // Reset all pointers to nullptr
  buffers_ = MemoryBuffers{};

  // Destroy streams safely
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

cv::Mat FCOSTrtBackend::preprocess_image(const cv::Mat & image) const
{
  cv::Mat processed;

  // Convert BGR to RGB
  cv::cvtColor(image, processed, cv::COLOR_BGR2RGB);

  // Resize to input dimensions
  cv::resize(processed, processed, cv::Size(config_.width, config_.height));

  // Convert to float and normalize to [0, 1]
  processed.convertTo(processed, CV_32F, 1.0 / 255.0);

  return processed;
}

FCOSTrtBackend::HeadOutputs FCOSTrtBackend::infer(const cv::Mat & image)
{
  // Preprocess image
  cv::Mat processed_image = preprocess_image(image);

  // Convert to CHW format and copy to GPU
  std::vector<float> input_data(3 * config_.height * config_.width);

  // OpenCV image is HWC, we need CHW
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < config_.height; h++) {
      for (int w = 0; w < config_.width; w++) {
        int chw_idx = c * config_.height * config_.width + h * config_.width + w;
        input_data[chw_idx] = processed_image.at<cv::Vec3f>(h, w)[c];
      }
    }
  }

  // Copy input data to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_input, input_data.data(),
                            input_data.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream_));

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw TensorRTException("Failed to run inference");
  }

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Prepare results
  HeadOutputs results;

  // Calculate output sizes
  size_t cls_size = cls_logits_size_ / sizeof(float);
  size_t bbox_size = bbox_regression_size_ / sizeof(float);
  size_t ctr_size = bbox_ctrness_size_ / sizeof(float);
  size_t anchors_size = anchors_size_ / sizeof(float);
  size_t image_sizes_size = image_sizes_size_ / sizeof(int64_t);
  size_t num_anchors_size = num_anchors_per_level_size_ / sizeof(int64_t);

  // Resize result vectors
  results.cls_logits.resize(cls_size);
  results.bbox_regression.resize(bbox_size);
  results.bbox_ctrness.resize(ctr_size);
  results.anchors.resize(anchors_size);
  results.image_sizes.resize(image_sizes_size);
  results.num_anchors_per_level.resize(num_anchors_size);

  // Copy results from GPU to CPU
  CUDA_CHECK(cudaMemcpy(results.cls_logits.data(), buffers_.device_cls_logits,
                        cls_logits_size_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results.bbox_regression.data(), buffers_.device_bbox_regression,
                        bbox_regression_size_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results.bbox_ctrness.data(), buffers_.device_bbox_ctrness,
                        bbox_ctrness_size_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results.anchors.data(), buffers_.device_anchors,
                        anchors_size_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results.image_sizes.data(), buffers_.device_image_sizes,
                        image_sizes_size_, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(results.num_anchors_per_level.data(), buffers_.device_num_anchors_per_level,
                        num_anchors_per_level_size_, cudaMemcpyDeviceToHost));

  return results;
}

} // namespace fcos_trt_backend
