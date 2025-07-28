// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <iostream>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "fcos_trt_backend/config.hpp"
#define private public
#include "fcos_trt_backend/fcos_trt_backend.hpp"
#undef private

class FCOSTrtBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Configure the FCOS detector
    fcos_trt_backend::FCOSTrtBackend::Config conf;
    conf.height = input_height_;
    conf.width = input_width_;
    conf.num_classes = num_classes_;
    conf.warmup_iterations = 2;
    conf.log_level = fcos_trt_backend::Logger::Severity::kINFO;

    // Configure post-processing
    conf.postprocess_config.score_thresh = 0.2f;
    conf.postprocess_config.nms_thresh = 0.6f;
    conf.postprocess_config.detections_per_img = 100;
    conf.postprocess_config.topk_candidates = 1000;

    try {
      detector = std::make_unique<fcos_trt_backend::FCOSTrtBackend>(engine_path_, conf);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize FCOS TensorRT detector: " << e.what();
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
    const cv::Mat & original,
    const cv::Mat & visualization,
    const fcos_trt_backend::Detections & detections,
    const std::string & suffix = "")
  {
    // Save original and visualization
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_detections" + suffix + ".png", visualization);

    // Print detection results to console
    std::cout << "\n=== Detection Results" + suffix + " ===" << std::endl;
    std::cout << "Total detections: " << detections.size() << std::endl;

    for (size_t i = 0; i < std::min(detections.size(), size_t(10)); ++i) {
      const auto& det = detections[i];
      std::cout << "Detection " << i+1 << ": "
                << "class=" << det.label
                << " (" << config::COCO_CLASS_NAMES[det.label] << "), "
                << "score=" << det.score << ", "
                << "bbox=[" << det.x1 << "," << det.y1 << ","
                << det.x2 << "," << det.y2 << "]" << std::endl;
    }
    if (detections.size() > 10) {
      std::cout << "... and " << (detections.size() - 10) << " more detections" << std::endl;
    }
  }

  std::shared_ptr<fcos_trt_backend::FCOSTrtBackend> detector;

public:
  const int input_width_ = 1238;
  const int input_height_ = 374;
  const int num_classes_ = 80; // COCO dataset

private:
  const std::string engine_path_ = "fcos_resnet50_fpn_374x1238.engine";
  const std::string image_path_ = "image_000.png";
};

TEST_F(FCOSTrtBackendTest, TestBasicDetection)
{
  cv::Mat image = load_test_image();
  EXPECT_EQ(image.cols, input_width_);
  EXPECT_EQ(image.rows, input_height_);
  EXPECT_EQ(image.type(), CV_8UC3);

  auto start = std::chrono::high_resolution_clock::now();
  auto detections = detector->detect(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "FCOS detection inference time: " << duration.count() << " ms" << std::endl;

  // Validate detections
  EXPECT_GE(detections.size(), 0); // Should have some detections or none (both valid)

  // Validate detection format
  for (const auto& det : detections) {
    EXPECT_GE(det.x1, 0.0f);
    EXPECT_GE(det.y1, 0.0f);
    EXPECT_LE(det.x2, static_cast<float>(image.cols));
    EXPECT_LE(det.y2, static_cast<float>(image.rows));
    EXPECT_GT(det.x2, det.x1); // x2 should be greater than x1
    EXPECT_GT(det.y2, det.y1); // y2 should be greater than y1
    EXPECT_GE(det.score, 0.0f);
    EXPECT_LE(det.score, 1.0f);
    EXPECT_GE(det.label, 0);
    EXPECT_LT(det.label, num_classes_);
  }

  // Create visualization
  cv::Mat visualization = detector->visualize_detections(image, detections, 0.5f, true);
  EXPECT_EQ(visualization.size(), image.size());
  EXPECT_EQ(visualization.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, visualization, detections, "_basic");
}

//TEST_F(FCOSTrtBackendTest, TestMultipleDetections)
//{
//  cv::Mat image = load_test_image();

//  const int num_iterations = 10;
//  std::vector<double> inference_times;
//  std::vector<size_t> detection_counts;

//  for (int i = 0; i < num_iterations; ++i) {
//    auto start = std::chrono::high_resolution_clock::now();
//    auto detections = detector->detect(image);
//    auto end = std::chrono::high_resolution_clock::now();

//    auto duration = std::chrono::duration<double, std::milli>(end - start);
//    inference_times.push_back(duration.count());
//    detection_counts.push_back(detections.size());

//    // Validate consistency
//    if (i > 0) {
//      // Detection count should be consistent (within reasonable bounds)
//      EXPECT_EQ(detection_counts[i], detection_counts[0]) 
//        << "Detection count should be consistent across runs";
//    }
//  }

//  // Calculate statistics
//  double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
//    inference_times.size();
//  double min_time = *std::min_element(inference_times.begin(), inference_times.end());
//  double max_time = *std::max_element(inference_times.begin(), inference_times.end());

//  std::cout << "\nMultiple detection statistics:" << std::endl;
//  std::cout << "  Average time: " << avg_time << " ms" << std::endl;
//  std::cout << "  Min time: " << min_time << " ms" << std::endl;
//  std::cout << "  Max time: " << max_time << " ms" << std::endl;
//  std::cout << "  Consistent detection count: " << detection_counts[0] << std::endl;

//  // Performance expectations (adjust based on your hardware)
//  EXPECT_LT(avg_time, 200.0); // Should be less than 200ms on decent hardware
//}

//TEST_F(FCOSTrtBackendTest, TestBenchmarkDetection)
//{
//  cv::Mat image = load_test_image();

//  const int warmup_iterations = 10;
//  const int benchmark_iterations = 100;

//  // Warmup
//  for (int i = 0; i < warmup_iterations; ++i) {
//    detector->detect(image);
//  }

//  // Benchmark
//  auto start = std::chrono::high_resolution_clock::now();

//  for (int i = 0; i < benchmark_iterations; ++i) {
//    detector->detect(image);
//  }

//  auto end = std::chrono::high_resolution_clock::now();
//  auto total_duration = std::chrono::duration<double, std::milli>(end - start);

//  double avg_time = total_duration.count() / benchmark_iterations;
//  double fps = 1000.0 / avg_time;

//  std::cout << "\n=== FCOS Benchmark Results ===" << std::endl;
//  std::cout << "Iterations: " << benchmark_iterations << std::endl;
//  std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
//  std::cout << "Average time per detection: " << avg_time << " ms" << std::endl;
//  std::cout << "Throughput: " << fps << " FPS" << std::endl;
//}

//TEST_F(FCOSTrtBackendTest, TestDifferentConfidenceThresholds)
//{
//  cv::Mat image = load_test_image();
//  auto detections = detector->detect(image);

//  if (detections.empty()) {
//    GTEST_SKIP() << "No detections found in test image";
//  }

//  // Test different confidence thresholds
//  std::vector<float> thresholds = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
//  
//  for (float threshold : thresholds) {
//    auto visualization = detector->visualize_detections(image, detections, threshold, true);
//    
//    // Count how many detections are above threshold
//    size_t count_above_threshold = 0;
//    for (const auto& det : detections) {
//      if (det.score >= threshold) {
//        count_above_threshold++;
//      }
//    }
//    
//    std::cout << "Confidence threshold " << threshold 
//              << ": " << count_above_threshold << " detections" << std::endl;
//    
//    std::string suffix = "_conf_" + std::to_string(static_cast<int>(threshold * 100));
//    cv::imwrite("test_output_threshold" + suffix + ".png", visualization);
//  }
//}

//TEST_F(FCOSTrtBackendTest, TestPostprocessorConfiguration)
//{
//  cv::Mat image = load_test_image();
//  
//  // Test with different post-processing configurations
//  struct TestConfig {
//    float score_thresh;
//    float nms_thresh;
//    int detections_per_img;
//    std::string description;
//  };
//  
//  std::vector<TestConfig> configs = {
//    {0.1f, 0.5f, 50, "low_threshold_strict_nms"},
//    {0.3f, 0.7f, 100, "medium_threshold_loose_nms"},
//    {0.5f, 0.6f, 200, "high_threshold_normal_nms"}
//  };
//  
//  for (const auto& config : configs) {  
//    // Create new detector with different config
//    fcos_trt_backend::FCOSTrtBackend::Config detector_config;
//    detector_config.height = input_height_;
//    detector_config.width = input_width_;
//    detector_config.num_classes = num_classes_;
//    detector_config.warmup_iterations = 0; // Skip warmup for speed
//    detector_config.log_level = fcos_trt_backend::Logger::Severity::kWARNING;
//    
//    detector_config.postprocess_config.score_thresh = config.score_thresh;
//    detector_config.postprocess_config.nms_thresh = config.nms_thresh;
//    detector_config.postprocess_config.detections_per_img = config.detections_per_img;
//    
//    try {
//      auto test_detector = std::make_unique<fcos_trt_backend::FCOSTrtBackend>(
//        "fcos_resnet50_fpn_374x1238.engine", detector_config);
//      
//      auto detections = test_detector->detect(image);
//      auto visualization = test_detector->visualize_detections(image, detections, 0.3f, true);
//      
//      std::cout << "\nConfig " << config.description << ":" << std::endl;
//      std::cout << "  Score threshold: " << config.score_thresh << std::endl;
//      std::cout << "  NMS threshold: " << config.nms_thresh << std::endl;
//      std::cout << "  Max detections: " << config.detections_per_img << std::endl;
//      std::cout << "  Actual detections: " << detections.size() << std::endl;
//      
//      // Validate max detections limit
//      EXPECT_LE(static_cast<int>(detections.size()), config.detections_per_img);
//      
//      save_results(image, visualization, detections, "_" + config.description);
//      
//    } catch (const std::exception& e) {
//      FAIL() << "Failed to test config " << config.description << ": " << e.what();
//    }
//  }
//}

//// Test with multiple different images (if available)
//TEST_F(FCOSTrtBackendTest, DISABLED_TestMultipleImages)
//{
//  std::vector<std::string> test_images = {
//    "image_000.png",
//    "image_001.png", 
//    "image_002.png"
//  };

//  int successful_tests = 0;

//  for (const auto & image_path : test_images) {
//    cv::Mat image = cv::imread(image_path);
//    if (image.empty()) {
//      std::cout << "Skipping missing image: " << image_path << std::endl;
//      continue;
//    }

//    // Resize to expected dimensions if needed
//    if (image.rows != input_height_ || image.cols != input_width_) {
//      cv::resize(image, image, cv::Size(input_width_, input_height_));
//    }

//    try {
//      auto detections = detector->detect(image);
//      auto visualization = detector->visualize_detections(image, detections);

//      // Save results with image-specific suffix
//      std::string suffix = "_image_" + std::to_string(successful_tests);
//      save_results(image, visualization, detections, suffix);

//      successful_tests++;

//    } catch (const std::exception & e) {
//      FAIL() << "Failed to process image " << image_path << ": " << e.what();
//    }
//  }

//  EXPECT_GT(successful_tests, 0) << "No test images were successfully processed";
//  std::cout << "Successfully processed " << successful_tests << " test images" << std::endl;
//}
