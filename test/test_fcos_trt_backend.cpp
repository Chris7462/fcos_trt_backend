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

    try {
      detector = std::make_unique<fcos_trt_backend::FCOSBackbone>(engine_path_, config);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize TensorRT inferencer: " << e.what();
    }
  }

  void TearDown() override
  {
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  void save_results(
    const cv::Mat & original, const cv::Mat & segmentation,
    const cv::Mat & overlay, const std::string & suffix = "")
  {
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_segmentation" + suffix + ".png", segmentation);
    cv::imwrite("test_output_overlay" + suffix + ".png", overlay);
  }

  std::shared_ptr<fcos_trt_backend::FCOSBackbone> detector;

public:
  const int input_width_ = 1238;
  const int input_height_ = 374;
  const int num_classes_ = 21;

private:
  const std::string engine_path_ = "fcn_resnet101_374x1238.engine";
  const std::string image_path_ = "image_000.png";
};

//TEST_F(FCNTrtBackendTest, TestBasicInference)
//{
//  cv::Mat image = load_test_image();
//  EXPECT_EQ(image.cols, 1238);
//  EXPECT_EQ(image.rows, 374);
//  EXPECT_EQ(image.type(), CV_8UC3);

//  auto start = std::chrono::high_resolution_clock::now();
//  cv::Mat segmentation = segmentor->infer(image);
//  auto end = std::chrono::high_resolution_clock::now();

//  auto duration = std::chrono::duration<double, std::milli>(end - start);
//  std::cout << "GPU infer with decode: " << duration.count() << " ms" << std::endl;

//  // Validate output
//  EXPECT_EQ(segmentation.rows, segmentor->config_.height);
//  EXPECT_EQ(segmentation.cols, segmentor->config_.width);
//  EXPECT_EQ(segmentation.type(), CV_8UC3);

//  // Create overlay
//  cv::Mat overlay = fcn_trt_backend::utils::create_overlay(image, segmentation, 0.5f);
//  EXPECT_EQ(overlay.size(), image.size());
//  EXPECT_EQ(overlay.type(), CV_8UC3);

//  // Save results for visual inspection
//  save_results(image, segmentation, overlay, "_gpu_optimized");

//  // Optional: Display results (comment out for automated testing)
//  /*
//  cv::imshow("Original", image);
//  cv::imshow("Segmentation", segmentation);
//  cv::imshow("Overlay", overlay);
//  cv::waitKey(0);
//  cv::destroyAllWindows();
//  */
//}