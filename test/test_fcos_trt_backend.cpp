//// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#define private public
#include "fcos_trt_backend/fcos_backbone.hpp"
#include "fcos_trt_backend/fcos_post_processor.hpp"
#undef private

#include "fcos_trt_backend/detection_utils.hpp"


class FCOSTrtBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Configure the inferencer
    fcos_trt_backend::FCOSBackbone::Config config;

    // Initialize postprocessor
    const float score_thresh = 0.2f;
    const float nms_thresh = 0.6f;
    const int detections_per_img = 100;
    const int topk_candidates = 1000;

    try {
      backbone = std::make_unique<fcos_trt_backend::FCOSBackbone>(engine_path_, config);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize backbone: " << e.what();
    }

    try {
      postprocessor = std::make_unique<fcos_trt_backend::FCOSPostProcessor>(
        score_thresh, nms_thresh, detections_per_img, topk_candidates);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize postprocessor: " << e.what();
    }
  }

  void TearDown() override
  {
  }

  cv::Mat load_test_image(std::string image_path)
  {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path);
    }
    return image;
  }

  std::shared_ptr<fcos_trt_backend::FCOSBackbone> backbone;
  std::shared_ptr<fcos_trt_backend::FCOSPostProcessor> postprocessor;

private:
  const std::string engine_path_ = "fcos_resnet50_fpn_374x1238.engine";
};


TEST_F(FCOSTrtBackendTest, TestBasicInference)
{
  const std::string image_path = "image_001.png";
  cv::Mat image = load_test_image(image_path);

  int original_height = image.rows;
  int original_width = image.cols;

  EXPECT_EQ(image.cols, 1238);
  EXPECT_EQ(image.rows, 374);
  EXPECT_EQ(image.type(), CV_8UC3);

  auto start1 = std::chrono::high_resolution_clock::now();
  auto head_outputs = backbone->infer(image);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration<double, std::milli>(end1 - start1);
  std::cout << "Backbone infer time: " << duration1.count() << " ms" << std::endl;

  auto start2 = std::chrono::high_resolution_clock::now();
  auto detection_results = postprocessor->postprocess_detections(
    head_outputs,
    original_height,
    original_width
  );
  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration<double, std::milli>(end2 - start2);
  std::cout << "PostProcessing time: " << duration2.count() << " ms" << std::endl;

  fcos_trt_backend::utils::plot_detections(
    image_path,
    detection_results,
    0.5f,  // confidence threshold
    "fcos_trt_detections.png"
  );

  fcos_trt_backend::utils::print_detection_results(detection_results, 9);
  // Validate output
  //EXPECT_EQ(segmentation.rows, segmentor->config_.height);
  //EXPECT_EQ(segmentation.cols, segmentor->config_.width);
  //EXPECT_EQ(segmentation.type(), CV_8UC3);
}
