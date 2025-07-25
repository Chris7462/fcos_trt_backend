#include <string>
#include <opencv2/core.hpp>
#include <torch/torch.h>
#include <torch/script.h>


namespace fcos_torch_backend
{
class FCOSTorchBackend
{
public:
  FCOSTorchBackend(const std::string & model_path);

  // Run inference
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict(const cv::Mat & image);

  void draw_predictions(
    cv::Mat & image, const torch::Tensor & boxes, const torch::Tensor & scores,
    const torch::Tensor & labels, float confidence_threshold = 0.6f);

private:
  // Convert OpenCV Mat to PyTorch tensor
  torch::Tensor mat_to_tensor(const cv::Mat & image);

private:
  torch::jit::script::Module model_;
  torch::Device device_;
};

} // namespace fcos_torch_backend
