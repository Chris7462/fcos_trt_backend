#include <algorithm>
#include <numeric>

#include "fcos_trt_backend/fcos_trt_backend.hpp"


namespace fcos_trt_backend
{

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
  // Create logger
  logger_ = std::make_unique<Logger>(config_.log_level);

  // Load engine file
  auto engine_data = load_engine_file(engine_path);

  // Create runtime
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw TensorRTException("Failed to create TensorRT runtime");
  }

  // Deserialize engine
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    throw TensorRTException("Failed to deserialize TensorRT engine");
  }

  // Create execution context
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    throw TensorRTException("Failed to create execution context");
  }
}

void FCOSTrtBackend::find_tensor_names()
{
  const int32_t num_io_tensors = engine_->getNbIOTensors();

  // Expected output names from ONNX export
  const std::vector<std::string> expected_outputs = {
    "cls_logits", "bbox_regression", "bbox_ctrness",
    "anchors", "image_sizes", "original_image_sizes", "num_anchors_per_level"
  };

  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
    nvinfer1::Dims dims = engine_->getTensorShape(tensor_name);

    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = tensor_name;

      // Calculate input size
      input_size_ = 1;
      for (int32_t j = 0; j < dims.nbDims; ++j) {
        input_size_ *= dims.d[j];
      }
      input_size_ *= sizeof(float);

      std::cout << "Input tensor: " << tensor_name << " [";
      for (int32_t j = 0; j < dims.nbDims; ++j) {
        std::cout << dims.d[j] << (j < dims.nbDims - 1 ? ", " : "");
      }
      std::cout << "]" << std::endl;
    }
    else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      OutputTensorInfo info;
      info.name = tensor_name;

      // Convert dims to vector and calculate size
      info.element_count = 1;
      for (int32_t j = 0; j < dims.nbDims; ++j) {
        info.dims.push_back(dims.d[j]);
        info.element_count *= dims.d[j];
      }
      info.size = info.element_count * sizeof(float);

      output_tensors_.push_back(info);

      std::cout << "Output tensor: " << tensor_name << " [";
      for (int32_t j = 0; j < dims.nbDims; ++j) {
        std::cout << dims.d[j] << (j < dims.nbDims - 1 ? ", " : "");
      }
      std::cout << "]" << std::endl;
    }
  }

  if (input_name_.empty()) {
    throw TensorRTException("No input tensor found");
  }

  if (output_tensors_.size() != expected_outputs.size()) {
    throw TensorRTException("Expected " + std::to_string(expected_outputs.size()) + 
                          " output tensors, found " + std::to_string(output_tensors_.size()));
  }
}

void FCOSTrtBackend::initialize_memory()
{
  // Allocate pinned input memory
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_input, input_size_));

  // Allocate device input memory
  CUDA_CHECK(cudaMalloc(&buffers_.device_input, input_size_));

  // Allocate output memories
  buffers_.device_outputs.resize(output_tensors_.size());
  buffers_.pinned_outputs.resize(output_tensors_.size());

  for (size_t i = 0; i < output_tensors_.size(); ++i) {
    // Allocate device memory for outputs
    CUDA_CHECK(cudaMalloc(&buffers_.device_outputs[i], output_tensors_[i].size));

    // Allocate pinned host memory for outputs
    CUDA_CHECK(cudaMallocHost(&buffers_.pinned_outputs[i], output_tensors_[i].size));

    // Set tensor addresses in context
    context_->setTensorAddress(output_tensors_[i].name.c_str(), buffers_.device_outputs[i]);
  }

  // Set input tensor address
  context_->setTensorAddress(input_name_.c_str(), buffers_.device_input);
}

void FCOSTrtBackend::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

void FCOSTrtBackend::warmup_engine()
{
  // Create dummy input image for warmup
  cv::Mat dummy_image = cv::Mat::zeros(config_.height, config_.width, CV_8UC3);

  for (int i = 0; i < config_.warmup_iterations; ++i) {
    FCOSOutputs dummy_outputs = infer(dummy_image);
  }

  // Synchronize to ensure warmup is complete
  CUDA_CHECK(cudaStreamSynchronize(stream_));
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

void FCOSTrtBackend::preprocess_image(const cv::Mat & image, float * output, cudaStream_t stream) const
{
  cv::Mat processed_image;

  // Resize to model input size
  cv::resize(image, processed_image, cv::Size(config_.width, config_.height));

  // Convert BGR to RGB
  cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);

  // Convert to float and normalize to [0, 1]
  processed_image.convertTo(processed_image, CV_32F, 1.0/255.0);

  // Apply ImageNet normalization
  std::vector<cv::Mat> channels;
  cv::split(processed_image, channels);

  for (int i = 0; i < 3; ++i) {
    channels[i] = (channels[i] - MEAN[i]) / STDDEV[i];
  }

  cv::merge(channels, processed_image);

  // Convert HWC to CHW format and copy to output buffer
  const int height = processed_image.rows;
  const int width = processed_image.cols;
  const int channels_count = processed_image.channels();

  std::vector<cv::Mat> channel_mats;
  cv::split(processed_image, channel_mats);

  float* ptr = output;
  for (int c = 0; c < channels_count; ++c) {
    std::memcpy(ptr, channel_mats[c].ptr<float>(), height * width * sizeof(float));
    ptr += height * width;
  }
}

FCOSTrtBackend::FCOSOutputs FCOSTrtBackend::infer(const cv::Mat & image)
{
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  // Preprocess image on CPU
  preprocess_image(image, buffers_.pinned_input, stream_);

  // Copy input to device
  CUDA_CHECK(cudaMemcpyAsync(
    buffers_.device_input,
    buffers_.pinned_input,
    input_size_,
    cudaMemcpyHostToDevice,
    stream_
  ));

  // Execute inference
  bool success = context_->enqueueV3(stream_);
  if (!success) {
    throw TensorRTException("Failed to execute inference");
  }

  // Copy outputs back to host
  for (size_t i = 0; i < output_tensors_.size(); ++i) {
    CUDA_CHECK(cudaMemcpyAsync(
      buffers_.pinned_outputs[i],
      buffers_.device_outputs[i],
      output_tensors_[i].size,
      cudaMemcpyDeviceToHost,
      stream_
    ));
  }

  // Wait for operations to complete
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Prepare outputs
  FCOSOutputs outputs;

  for (size_t i = 0; i < output_tensors_.size(); ++i) {
    const auto& tensor_info = output_tensors_[i];
    std::vector<float> data(tensor_info.element_count);
    std::memcpy(data.data(), buffers_.pinned_outputs[i], tensor_info.size);

    // Store outputs based on tensor name
    if (tensor_info.name == "cls_logits") {
      outputs.cls_logits = std::move(data);
      outputs.cls_logits_dims = tensor_info.dims;
    } else if (tensor_info.name == "bbox_regression") {
      outputs.bbox_regression = std::move(data);
      outputs.bbox_regression_dims = tensor_info.dims;
    } else if (tensor_info.name == "bbox_ctrness") {
      outputs.bbox_ctrness = std::move(data);
      outputs.bbox_ctrness_dims = tensor_info.dims;
    } else if (tensor_info.name == "anchors") {
      outputs.anchors = std::move(data);
      outputs.anchors_dims = tensor_info.dims;
    } else if (tensor_info.name == "image_sizes") {
      outputs.image_sizes = std::move(data);
      outputs.image_sizes_dims = tensor_info.dims;
    } else if (tensor_info.name == "original_image_sizes") {
      outputs.original_image_sizes = std::move(data);
      outputs.original_image_sizes_dims = tensor_info.dims;
    } else if (tensor_info.name == "num_anchors_per_level") {
      outputs.num_anchors_per_level = std::move(data);
      outputs.num_anchors_per_level_dims = tensor_info.dims;
    }
  }

  return outputs;
}

void FCOSTrtBackend::cleanup() noexcept
{
  try {
    // Free pinned memory
    if (buffers_.pinned_input) {
      cudaFreeHost(buffers_.pinned_input);
      buffers_.pinned_input = nullptr;
    }

    for (auto& pinned_output : buffers_.pinned_outputs) {
      if (pinned_output) {
        cudaFreeHost(pinned_output);
        pinned_output = nullptr;
      }
    }

    // Free device memory
    if (buffers_.device_input) {
      cudaFree(buffers_.device_input);
      buffers_.device_input = nullptr;
    }

    for (auto& device_output : buffers_.device_outputs) {
      if (device_output) {
        cudaFree(device_output);
        device_output = nullptr;
      }
    }

    // Destroy stream
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  } catch (...) {
    // Ignore exceptions in cleanup
  }
}

} // namespace fcos_trt_backend
