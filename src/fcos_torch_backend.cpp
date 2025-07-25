#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "fcos_torch_backend/config.hpp"
#include "fcos_torch_backend/fcos_torch_backend.hpp"


namespace fcos_torch_backend
{

FCOSTorchBackend::FCOSTorchBackend(const std::string & model_path)
: device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
  try {
    if (model_path.empty()) {
      throw std::runtime_error("Model path cannot be empty.");
    }
    // Load the TorchScript model
    model_ = torch::jit::load(model_path);
    model_.to(device_);
    model_.eval();
    std::cout << "Model loaded successfully on " <<
      (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
  } catch (const c10::Error & e) {
    throw std::runtime_error("PyTorch error loading model: " + std::string(e.what()));
  } catch (const std::exception & e) {
    throw std::runtime_error("Error loading model: " + std::string(e.what()));
  }
}

// Convert OpenCV Mat to PyTorch tensor
torch::Tensor FCOSTorchBackend::mat_to_tensor(const cv::Mat & image)
{
  cv::Mat float_image;
  image.convertTo(float_image, CV_32F, 1.0 / 255.0);

  // Convert from HWC to CHW
  auto tensor = torch::from_blob(float_image.data, {image.rows, image.cols, 3}, torch::kFloat);
  tensor = tensor.permute({2, 0, 1}); // HWC to CHW

  return tensor.to(device_);
}

// Run inference
std::tuple<torch::Tensor, torch::Tensor,
  torch::Tensor> FCOSTorchBackend::predict(const cv::Mat & image)
{
  torch::NoGradGuard no_grad;

  // Convert image to tensor (remove batch dimension since we'll create a list)
  auto tensor = mat_to_tensor(image);

  // Create a list of tensors (this is what FCOS expects)
  std::vector<torch::Tensor> image_list = {tensor};

  // Run inference
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(image_list);

  auto output = model_.forward(inputs);

  // FCOS returns a tuple: (Losses, Detections) in scripting mode
  auto output_tuple = output.toTuple();

  // Get the detections (second element of the tuple)
  auto detections = output_tuple->elements()[1];
  auto detection_list = detections.toList();
  auto detection_dict = detection_list.get(0).toGenericDict();

  // Extract predictions
  auto boxes = detection_dict.at("boxes").toTensor().to(torch::kCPU);
  auto scores = detection_dict.at("scores").toTensor().to(torch::kCPU);
  auto labels = detection_dict.at("labels").toTensor().to(torch::kCPU);

  return std::make_tuple(boxes, scores, labels);
}

// Draw predictions on image
void FCOSTorchBackend::draw_predictions(
  cv::Mat & image, const torch::Tensor & boxes, const torch::Tensor & scores,
  const torch::Tensor & labels, float confidence_threshold)
{
  auto boxes_a = boxes.accessor<float, 2>();
  auto scores_a = scores.accessor<float, 1>();
  auto labels_a = labels.accessor<long, 1>();

  int num_detections = scores.size(0);

  for (int i = 0; i < num_detections; ++i) {
    float score = scores_a[i];

    if (score >= confidence_threshold) {
      // Get bounding box coordinates
      int x1 = static_cast<int>(boxes_a[i][0]);
      int y1 = static_cast<int>(boxes_a[i][1]);
      int x2 = static_cast<int>(boxes_a[i][2]);
      int y2 = static_cast<int>(boxes_a[i][3]);

      // Get class label
      size_t label_idx = labels_a[i];
      std::string class_name = (label_idx <
        config::COCO_CLASSES.size()) ? config::COCO_CLASSES[label_idx] : "unknown";

      // Draw bounding box
      cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);

      // Draw label and score
      std::string label_text = class_name + ": " + std::to_string(score).substr(0, 4);
      int baseline;
      cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      cv::rectangle(image,
        cv::Point(x1, y1 - text_size.height - baseline),
        cv::Point(x1 + text_size.width, y1),
        cv::Scalar(0, 0, 255), -1);

      cv::putText(image, label_text, cv::Point(x1, y1 - baseline),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
  }
}

} // namespace fcos_torch_backend
