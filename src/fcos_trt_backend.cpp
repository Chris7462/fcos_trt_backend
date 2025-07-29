#include <iostream>
#include <fstream>
#include <algorithm>

// OpenCV includes
#include <opencv2/imgproc.hpp>

// local header files: This project includes local header files.
#include "fcos_trt_backend/config.hpp"
#include "fcos_trt_backend/exception.hpp"
#include "fcos_trt_backend/fcos_trt_backend.hpp"
#include "fcos_trt_backend/normalize_kernel.hpp"

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
: config_(config), num_anchors_(0), num_levels_(0)
{
  try {
    initialize_engine(engine_path);
    find_tensor_names();
    initialize_memory();
    initialize_streams();
    initialize_constants();

    // Initialize post-processor
    postprocessor_ = std::make_unique<FCOSPostProcessor>(config_.postprocess_config);

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
  // Expected tensor names based on ONNX export
  const std::vector<std::string> expected_outputs = {
    "cls_logits", "bbox_regression", "bbox_ctrness", "anchors",
    "image_sizes", "original_image_sizes", "num_anchors_per_level"
  };

  bool found_input = false;
  int outputs_found = 0;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      tensor_names_.input_name = tensor_name;
      found_input = true;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      std::string name_str(tensor_name);

      if (name_str == "cls_logits") {
        tensor_names_.cls_logits_name = tensor_name;
        outputs_found++;
      } else if (name_str == "bbox_regression") {
        tensor_names_.bbox_regression_name = tensor_name;
        outputs_found++;
      } else if (name_str == "bbox_ctrness") {
        tensor_names_.bbox_ctrness_name = tensor_name;
        outputs_found++;
      } else if (name_str == "anchors") {
        tensor_names_.anchors_name = tensor_name;
        outputs_found++;
      } else if (name_str == "image_sizes") {
        tensor_names_.image_sizes_name = tensor_name;
        outputs_found++;
      } else if (name_str == "original_image_sizes") {
        tensor_names_.original_image_sizes_name = tensor_name;
        outputs_found++;
      } else if (name_str == "num_anchors_per_level") {
        tensor_names_.num_anchors_per_level_name = tensor_name;
        outputs_found++;
      }
    }
  }

  if (!found_input) {
    throw TensorRTException("Failed to find input tensor");
  }

  if (outputs_found != 7) {
    throw TensorRTException("Failed to find all required output tensors. Found: " +
                           std::to_string(outputs_found) + "/7");
  }
}

void FCOSTrtBackend::initialize_memory()
{
  // Get tensor dimensions to calculate memory sizes
  auto get_tensor_size = [this](const std::string& name) -> size_t {
    auto dims = engine_->getTensorShape(name.c_str());
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
      size *= dims.d[i];
    }
    return size;
  };

  // Calculate memory sizes for known tensors
  memory_sizes_.input_size = 1 * 3 * config_.height * config_.width * sizeof(float);
  memory_sizes_.cls_logits_size = get_tensor_size(tensor_names_.cls_logits_name) * sizeof(float);
  memory_sizes_.bbox_regression_size = get_tensor_size(tensor_names_.bbox_regression_name) * sizeof(float);
  memory_sizes_.bbox_ctrness_size = get_tensor_size(tensor_names_.bbox_ctrness_name) * sizeof(float);
  memory_sizes_.anchors_size = get_tensor_size(tensor_names_.anchors_name) * sizeof(float);
  memory_sizes_.image_sizes_size = get_tensor_size(tensor_names_.image_sizes_name) * sizeof(int32_t);
  memory_sizes_.original_image_sizes_size = get_tensor_size(tensor_names_.original_image_sizes_name) * sizeof(int32_t);
  memory_sizes_.num_anchors_per_level_size = get_tensor_size(tensor_names_.num_anchors_per_level_name) * sizeof(int32_t);

  // Calculate model dimensions
  num_anchors_ = static_cast<int>(memory_sizes_.cls_logits_size / sizeof(float) / config_.num_classes);
  num_levels_ = static_cast<int>(memory_sizes_.num_anchors_per_level_size / sizeof(int32_t));

  // Allocate pinned host memory
  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_input, memory_sizes_.input_size));

  // Allocate device memory for known tensors
  CUDA_CHECK(cudaMalloc(&buffers_.device_input, memory_sizes_.input_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_temp_buffer, memory_sizes_.input_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_cls_logits, memory_sizes_.cls_logits_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_bbox_regression, memory_sizes_.bbox_regression_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_bbox_ctrness, memory_sizes_.bbox_ctrness_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_anchors, memory_sizes_.anchors_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_image_sizes, memory_sizes_.image_sizes_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_original_image_sizes, memory_sizes_.original_image_sizes_size));
  CUDA_CHECK(cudaMalloc(&buffers_.device_num_anchors_per_level, memory_sizes_.num_anchors_per_level_size));

  // Allocate host memory for outputs
  CUDA_CHECK(cudaMallocHost(&buffers_.host_cls_logits, memory_sizes_.cls_logits_size));
  CUDA_CHECK(cudaMallocHost(&buffers_.host_bbox_regression, memory_sizes_.bbox_regression_size));
  CUDA_CHECK(cudaMallocHost(&buffers_.host_bbox_ctrness, memory_sizes_.bbox_ctrness_size));
  CUDA_CHECK(cudaMallocHost(&buffers_.host_anchors, memory_sizes_.anchors_size));
  CUDA_CHECK(cudaMallocHost(&buffers_.host_image_sizes, memory_sizes_.image_sizes_size));
  CUDA_CHECK(cudaMallocHost(&buffers_.host_original_image_sizes, memory_sizes_.original_image_sizes_size));
  CUDA_CHECK(cudaMallocHost(&buffers_.host_num_anchors_per_level, memory_sizes_.num_anchors_per_level_size));

  // Set tensor addresses for ALL tensors (both known and unknown)

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);
    std::string name_str(tensor_name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      if (name_str == tensor_names_.input_name) {
        if (!context_->setTensorAddress(tensor_name, buffers_.device_input)) {
          throw TensorRTException("Failed to set input tensor address: " + name_str);
        }
      }
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      void* tensor_address = nullptr;

      // Set addresses for known output tensors
      if (name_str == tensor_names_.cls_logits_name) {
        tensor_address = buffers_.device_cls_logits;
      } else if (name_str == tensor_names_.bbox_regression_name) {
        tensor_address = buffers_.device_bbox_regression;
      } else if (name_str == tensor_names_.bbox_ctrness_name) {
        tensor_address = buffers_.device_bbox_ctrness;
      } else if (name_str == tensor_names_.anchors_name) {
        tensor_address = buffers_.device_anchors;
      } else if (name_str == tensor_names_.image_sizes_name) {
        tensor_address = buffers_.device_image_sizes;
      } else if (name_str == tensor_names_.original_image_sizes_name) {
        tensor_address = buffers_.device_original_image_sizes;
      } else if (name_str == tensor_names_.num_anchors_per_level_name) {
        tensor_address = buffers_.device_num_anchors_per_level;
      } else {
        // Handle unknown output tensors by allocating dummy buffers
        std::cout << "Warning: Unknown output tensor '" << name_str
                  << "' - allocating dummy buffer" << std::endl;

        size_t tensor_size = get_tensor_size(name_str);
        auto dims = engine_->getTensorShape(tensor_name);
        nvinfer1::DataType data_type = engine_->getTensorDataType(tensor_name);

        size_t element_size = 4; // Default to float
        if (data_type == nvinfer1::DataType::kINT32) {
          element_size = sizeof(int32_t);
        } else if (data_type == nvinfer1::DataType::kFLOAT) {
          element_size = sizeof(float);
        } else if (data_type == nvinfer1::DataType::kHALF) {
          element_size = sizeof(uint16_t);
        }

        size_t buffer_size = tensor_size * element_size;
        void* dummy_buffer;
        CUDA_CHECK(cudaMalloc(&dummy_buffer, buffer_size));
        buffers_.additional_device_buffers.push_back(dummy_buffer);
        tensor_address = dummy_buffer;
      }

      if (!context_->setTensorAddress(tensor_name, tensor_address)) {
        throw TensorRTException("Failed to set output tensor address: " + name_str);
      }
    }
  }

  // Log information about additional buffers
  if (!buffers_.additional_device_buffers.empty()) {
    std::cout << "Allocated " << buffers_.additional_device_buffers.size()
              << " additional buffers for unknown tensors" << std::endl;
  }
}

//void FCOSTrtBackend::initialize_memory()
//{
//  // Get tensor dimensions to calculate memory sizes
//  auto get_tensor_size = [this](const std::string& name) -> size_t {
//    auto dims = engine_->getTensorShape(name.c_str());
//    size_t size = 1;
//    for (int i = 0; i < dims.nbDims; ++i) {
//      size *= dims.d[i];
//    }
//    return size;
//  };

//  // Calculate memory sizes
//  memory_sizes_.input_size = 1 * 3 * config_.height * config_.width * sizeof(float);
//  memory_sizes_.cls_logits_size = get_tensor_size(tensor_names_.cls_logits_name) * sizeof(float);
//  memory_sizes_.bbox_regression_size = get_tensor_size(tensor_names_.bbox_regression_name) * sizeof(float);
//  memory_sizes_.bbox_ctrness_size = get_tensor_size(tensor_names_.bbox_ctrness_name) * sizeof(float);
//  memory_sizes_.anchors_size = get_tensor_size(tensor_names_.anchors_name) * sizeof(float);
//  memory_sizes_.image_sizes_size = get_tensor_size(tensor_names_.image_sizes_name) * sizeof(int32_t);
//  memory_sizes_.original_image_sizes_size = get_tensor_size(tensor_names_.original_image_sizes_name) * sizeof(int32_t);
//  memory_sizes_.num_anchors_per_level_size = get_tensor_size(tensor_names_.num_anchors_per_level_name) * sizeof(int32_t);

//  // Calculate model dimensions
//  num_anchors_ = static_cast<int>(memory_sizes_.cls_logits_size / sizeof(float) / config_.num_classes);
//  num_levels_ = static_cast<int>(memory_sizes_.num_anchors_per_level_size / sizeof(int32_t));

//  // Allocate pinned host memory
//  CUDA_CHECK(cudaMallocHost(&buffers_.pinned_input, memory_sizes_.input_size));

//  // Allocate device memory
//  CUDA_CHECK(cudaMalloc(&buffers_.device_input, memory_sizes_.input_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_temp_buffer, memory_sizes_.input_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_cls_logits, memory_sizes_.cls_logits_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_bbox_regression, memory_sizes_.bbox_regression_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_bbox_ctrness, memory_sizes_.bbox_ctrness_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_anchors, memory_sizes_.anchors_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_image_sizes, memory_sizes_.image_sizes_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_original_image_sizes, memory_sizes_.original_image_sizes_size));
//  CUDA_CHECK(cudaMalloc(&buffers_.device_num_anchors_per_level, memory_sizes_.num_anchors_per_level_size));

//  // Allocate host memory for outputs
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_cls_logits, memory_sizes_.cls_logits_size));
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_bbox_regression, memory_sizes_.bbox_regression_size));
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_bbox_ctrness, memory_sizes_.bbox_ctrness_size));
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_anchors, memory_sizes_.anchors_size));
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_image_sizes, memory_sizes_.image_sizes_size));
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_original_image_sizes, memory_sizes_.original_image_sizes_size));
//  CUDA_CHECK(cudaMallocHost(&buffers_.host_num_anchors_per_level, memory_sizes_.num_anchors_per_level_size));

//  // Set tensor addresses for ALL tensors (both known and unknown)
//  std::vector<void*> additional_buffers; // To track additional allocations

//  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
//    const char* tensor_name = engine_->getIOTensorName(i);
//    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);
//    std::string name_str(tensor_name);

//    if (mode == nvinfer1::TensorIOMode::kINPUT) {
//      if (name_str == tensor_names_.input_name) {
//        if (!context_->setTensorAddress(tensor_name, buffers_.device_input)) {
//          throw TensorRTException("Failed to set input tensor address: " + name_str);
//        }
//      }
//    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
//      void* tensor_address = nullptr;

//      // Set addresses for known output tensors
//      if (name_str == tensor_names_.cls_logits_name) {
//        tensor_address = buffers_.device_cls_logits;
//      } else if (name_str == tensor_names_.bbox_regression_name) {
//        tensor_address = buffers_.device_bbox_regression;
//      } else if (name_str == tensor_names_.bbox_ctrness_name) {
//        tensor_address = buffers_.device_bbox_ctrness;
//      } else if (name_str == tensor_names_.anchors_name) {
//        tensor_address = buffers_.device_anchors;
//      } else if (name_str == tensor_names_.image_sizes_name) {
//        tensor_address = buffers_.device_image_sizes;
//      } else if (name_str == tensor_names_.original_image_sizes_name) {
//        tensor_address = buffers_.device_original_image_sizes;
//      } else if (name_str == tensor_names_.num_anchors_per_level_name) {
//        tensor_address = buffers_.device_num_anchors_per_level;
//      } else {
//        // Handle unknown output tensors by allocating dummy buffers
//        std::cout << "Warning: Unknown output tensor '" << name_str
//                  << "' - allocating dummy buffer" << std::endl;

//        size_t tensor_size = get_tensor_size(name_str);
//        auto dims = engine_->getTensorShape(tensor_name);
//        nvinfer1::DataType data_type = engine_->getTensorDataType(tensor_name);

//        size_t element_size = 4; // Default to float
//        if (data_type == nvinfer1::DataType::kINT32) {
//          element_size = sizeof(int32_t);
//        } else if (data_type == nvinfer1::DataType::kFLOAT) {
//          element_size = sizeof(float);
//        } else if (data_type == nvinfer1::DataType::kHALF) {
//          element_size = sizeof(uint16_t);
//        }

//        size_t buffer_size = tensor_size * element_size;
//        void* dummy_buffer;
//        CUDA_CHECK(cudaMalloc(&dummy_buffer, buffer_size));
//        additional_buffers.push_back(dummy_buffer);
//        tensor_address = dummy_buffer;
//      }

//      if (!context_->setTensorAddress(tensor_name, tensor_address)) {
//        throw TensorRTException("Failed to set output tensor address: " + name_str);
//      }
//    }
//  }

//  // Store additional buffers for cleanup
//  if (!additional_buffers.empty()) {
//    // You'll need to add this to your class to track additional buffers for cleanup
//    // For now, we'll accept the small memory leak since this is initialization code
//    std::cout << "Allocated " << additional_buffers.size()
//              << " additional buffers for unknown tensors" << std::endl;
//  }
//}

void FCOSTrtBackend::initialize_streams()
{
  CUDA_CHECK(cudaStreamCreate(&stream_));
}

void FCOSTrtBackend::initialize_constants()
{
  // Initialize CUDA constant memory for normalization
  initialize_mean_std_constants();
}

void FCOSTrtBackend::warmup_engine()
{
  // Zero the input buffer
  CUDA_CHECK(cudaMemsetAsync(buffers_.device_input, 0, memory_sizes_.input_size, stream_));

  for (int i = 0; i < config_.warmup_iterations; ++i) {
    // Run inference pipeline once to initialize CUDA kernels
    if (!context_->enqueueV3(stream_)) {
      throw TensorRTException("Failed to enqueue warmup inference");
    }

    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  std::cout << "FCOS Engine warmed up with " << config_.warmup_iterations << " iterations" << std::endl;
}

void FCOSTrtBackend::cleanup() noexcept
{
  // Free pinned host memory
  if (buffers_.pinned_input) cudaFreeHost(buffers_.pinned_input);
  if (buffers_.host_cls_logits) cudaFreeHost(buffers_.host_cls_logits);
  if (buffers_.host_bbox_regression) cudaFreeHost(buffers_.host_bbox_regression);
  if (buffers_.host_bbox_ctrness) cudaFreeHost(buffers_.host_bbox_ctrness);
  if (buffers_.host_anchors) cudaFreeHost(buffers_.host_anchors);
  if (buffers_.host_image_sizes) cudaFreeHost(buffers_.host_image_sizes);
  if (buffers_.host_original_image_sizes) cudaFreeHost(buffers_.host_original_image_sizes);
  if (buffers_.host_num_anchors_per_level) cudaFreeHost(buffers_.host_num_anchors_per_level);

  // Free device memory
  if (buffers_.device_input) cudaFree(buffers_.device_input);
  if (buffers_.device_temp_buffer) cudaFree(buffers_.device_temp_buffer);
  if (buffers_.device_cls_logits) cudaFree(buffers_.device_cls_logits);
  if (buffers_.device_bbox_regression) cudaFree(buffers_.device_bbox_regression);
  if (buffers_.device_bbox_ctrness) cudaFree(buffers_.device_bbox_ctrness);
  if (buffers_.device_anchors) cudaFree(buffers_.device_anchors);
  if (buffers_.device_image_sizes) cudaFree(buffers_.device_image_sizes);
  if (buffers_.device_original_image_sizes) cudaFree(buffers_.device_original_image_sizes);
  if (buffers_.device_num_anchors_per_level) cudaFree(buffers_.device_num_anchors_per_level);

  // Free additional device buffers
  for (void* buffer : buffers_.additional_device_buffers) {
    if (buffer) cudaFree(buffer);
  }
  buffers_.additional_device_buffers.clear();

  // Reset all pointers
  buffers_ = MemoryBuffers{};

  // Destroy streams
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

Detections FCOSTrtBackend::detect(const cv::Mat & image)
{
  // Store original image dimensions
  int original_width = image.cols;
  int original_height = image.rows;

  // Preprocess image
  preprocess_image(image, buffers_.device_input, stream_);

  // Run inference
  if (!context_->enqueueV3(stream_)) {
    throw TensorRTException("Failed to enqueue inference");
  }

  // Copy outputs from device to host
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_cls_logits, buffers_.device_cls_logits,
                            memory_sizes_.cls_logits_size, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_bbox_regression, buffers_.device_bbox_regression,
                            memory_sizes_.bbox_regression_size, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_bbox_ctrness, buffers_.device_bbox_ctrness,
                            memory_sizes_.bbox_ctrness_size, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_anchors, buffers_.device_anchors,
                            memory_sizes_.anchors_size, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_image_sizes, buffers_.device_image_sizes,
                            memory_sizes_.image_sizes_size, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_original_image_sizes, buffers_.device_original_image_sizes,
                            memory_sizes_.original_image_sizes_size, cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaMemcpyAsync(buffers_.host_num_anchors_per_level, buffers_.device_num_anchors_per_level,
                            memory_sizes_.num_anchors_per_level_size, cudaMemcpyDeviceToHost, stream_));

  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // Extract raw outputs
  RawOutputs raw_outputs = extract_raw_outputs();

  // Post-process to get final detections
  return postprocessor_->postprocess(raw_outputs, original_width, original_height);
}

RawOutputs FCOSTrtBackend::extract_raw_outputs() const
{
  RawOutputs outputs;

  outputs.cls_logits = buffers_.host_cls_logits;
  outputs.bbox_regression = buffers_.host_bbox_regression;
  outputs.bbox_ctrness = buffers_.host_bbox_ctrness;
  outputs.anchors = buffers_.host_anchors;
  outputs.image_sizes = buffers_.host_image_sizes;
  outputs.original_image_sizes = buffers_.host_original_image_sizes;
  outputs.num_anchors_per_level = buffers_.host_num_anchors_per_level;

  outputs.num_anchors = num_anchors_;
  outputs.num_classes = config_.num_classes;
  outputs.num_levels = num_levels_;

  return outputs;
}

cv::Mat FCOSTrtBackend::visualize_detections(
  const cv::Mat & image,
  const Detections & detections,
  float confidence_threshold,
  bool draw_labels)
{
  cv::Mat result = image.clone();

  for (const auto& detection : detections) {
    if (detection.score < confidence_threshold) {
      continue;
    }

    // Draw bounding box
    cv::Point2i pt1(static_cast<int>(detection.x1), static_cast<int>(detection.y1));
    cv::Point2i pt2(static_cast<int>(detection.x2), static_cast<int>(detection.y2));

    // Get color for this class
    auto color = cv::Scalar(
        config::COCO_COLORS[detection.label % config::COCO_COLORS.size()][0],
        config::COCO_COLORS[detection.label % config::COCO_COLORS.size()][1],
        config::COCO_COLORS[detection.label % config::COCO_COLORS.size()][2]
    );

    cv::rectangle(result, pt1, pt2, color, 2);

    if (draw_labels) {
      // Draw label and confidence
      std::string label_text = config::COCO_CLASS_NAMES[detection.label] +
                              ": " + std::to_string(detection.score).substr(0, 4);

      int baseline = 0;
      cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      cv::Point2i label_pos(pt1.x, pt1.y - 10);
      cv::rectangle(result,
                   cv::Point2i(label_pos.x, label_pos.y - text_size.height - baseline),
                   cv::Point2i(label_pos.x + text_size.width, label_pos.y + baseline),
                   color, -1);

      cv::putText(result, label_text, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                 cv::Scalar(255, 255, 255), 1);
    }
  }

  return result;
}

void FCOSTrtBackend::preprocess_image(
  const cv::Mat & image, float * output, cudaStream_t stream) const
{
  // Step 1: Resize image using OpenCV (on CPU)
  cv::Mat img_wrapper(config_.height, config_.width, CV_32FC3, buffers_.pinned_input);
  cv::resize(image, img_wrapper, cv::Size(config_.width, config_.height));

  // Step 2: Convert to float (on CPU)
  img_wrapper.convertTo(img_wrapper, CV_32FC3, 1.0f / 255.0f);

  // Step 3: Upload resized float image to GPU
  CUDA_CHECK(cudaMemcpyAsync(buffers_.device_temp_buffer, img_wrapper.data,
    memory_sizes_.input_size, cudaMemcpyHostToDevice, stream));

  // Step 4: Launch normalization kernel
  launch_normalize_kernel(
    buffers_.device_temp_buffer,
    output,
    config_.width, config_.height,
    stream);
}

} // namespace fcos_trt_backend
