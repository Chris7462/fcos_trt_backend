#include "fcos_trt_backend/fcos_backbone.hpp"
#include "internal/fcos_backbone_impl.hpp"
#include "internal/cuda_check.hpp"


namespace fcos_trt_backend
{

FCOSTrtBackend::FCOSTrtBackend(const std::string & engine_path, const Config & config)
: config_(config), impl_(std::make_unique<Impl>())
{
  try {
    impl_->initialize(engine_path, config_);
  } catch (const std::exception & e) {
    impl_.reset();
    throw internal::TensorRTException("Initialization failed: " + std::string(e.what()));
  }
}

FCOSTrtBackend::~FCOSTrtBackend() = default;

HeadOutputs FCOSTrtBackend::infer(const cv::Mat & image)
{
  std::lock_guard<std::mutex> lock(infer_mutex_);
  return impl_->infer(image, config_);
}

} // namespace fcos_trt_backend
