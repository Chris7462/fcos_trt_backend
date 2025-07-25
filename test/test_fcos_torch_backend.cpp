// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <set>
#include <algorithm>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Torch includes
#include <torch/torch.h>

// Local includes
#include "fcos_torch_backend/config.hpp"
#define private public
#include "fcos_torch_backend/fcos_torch_backend.hpp"
#undef private


class FCOSTorchBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    try {
      detector_ = std::make_unique<fcos_torch_backend::FCOSTorchBackend>(model_path_);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize FCOS detector: " << e.what();
    }
  }

  void TearDown() override
  {
    // Clean up if needed
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  void save_detection_results(
    const cv::Mat & image_with_boxes, const std::string & suffix = "")
  {
    cv::imwrite("test_detection_output" + suffix + ".png", image_with_boxes);
  }

  void validate_detection_tensors(
    const torch::Tensor & boxes,
    const torch::Tensor & scores,
    const torch::Tensor & labels)
  {
    // Check tensor shapes
    EXPECT_EQ(boxes.dim(), 2) << "Boxes should be 2D tensor";
    EXPECT_EQ(scores.dim(), 1) << "Scores should be 1D tensor";
    EXPECT_EQ(labels.dim(), 1) << "Labels should be 1D tensor";

    // Check tensor sizes are consistent
    EXPECT_EQ(boxes.size(0), scores.size(0)) << "Number of boxes and scores should match";
    EXPECT_EQ(boxes.size(0), labels.size(0)) << "Number of boxes and labels should match";
    EXPECT_EQ(boxes.size(1), 4) << "Each box should have 4 coordinates";

    // Check data types
    EXPECT_TRUE(boxes.dtype() == torch::kFloat32 || boxes.dtype() == torch::kFloat64)
      << "Boxes should be float tensor";
    EXPECT_TRUE(scores.dtype() == torch::kFloat32 || scores.dtype() == torch::kFloat64)
      << "Scores should be float tensor";
    EXPECT_TRUE(labels.dtype() == torch::kInt64 || labels.dtype() == torch::kLong)
      << "Labels should be integer tensor";
  }

  void validate_bounding_boxes(const torch::Tensor & boxes, const cv::Size & image_size)
  {
    auto boxes_a = boxes.accessor<float, 2>();

    for (int i = 0; i < boxes.size(0); ++i) {
      float x1 = boxes_a[i][0];
      float y1 = boxes_a[i][1];
      float x2 = boxes_a[i][2];
      float y2 = boxes_a[i][3];

      // Check coordinate validity
      EXPECT_GE(x1, 0.0f) << "x1 should be >= 0";
      EXPECT_GE(y1, 0.0f) << "y1 should be >= 0";
      EXPECT_LT(x2, static_cast<float>(image_size.width)) << "x2 should be < image width";
      EXPECT_LT(y2, static_cast<float>(image_size.height)) << "y2 should be < image height";

      // Check box validity (x2 > x1, y2 > y1)
      EXPECT_GT(x2, x1) << "x2 should be greater than x1";
      EXPECT_GT(y2, y1) << "y2 should be greater than y1";
    }
  }

  void validate_scores_and_labels(const torch::Tensor & scores, const torch::Tensor & labels)
  {
    auto scores_a = scores.accessor<float, 1>();
    auto labels_a = labels.accessor<long, 1>();

    for (int i = 0; i < scores.size(0); ++i) {
      // Check score range [0, 1]
      EXPECT_GE(scores_a[i], 0.0f) << "Score should be >= 0";
      EXPECT_LE(scores_a[i], 1.0f) << "Score should be <= 1";

      // Check label validity (should be valid COCO class indices)
      EXPECT_GE(labels_a[i], 0) << "Label should be >= 0";
      EXPECT_LT(labels_a[i], static_cast<long>(config::COCO_CLASSES.size()))
        << "Label should be < number of COCO classes";
    }
  }

  std::unique_ptr<fcos_torch_backend::FCOSTorchBackend> detector_;

private:
  const std::string model_path_ = "fcos_resnet50_fpn_374x1238.pt";
  const std::string image_path_ = "image_000.png";
};


TEST_F(FCOSTorchBackendTest, TestBasicInference)
{
  cv::Mat image = load_test_image();

  // Validate input image
  EXPECT_FALSE(image.empty());
  EXPECT_EQ(image.type(), CV_8UC3);
  EXPECT_GT(image.rows, 0);
  EXPECT_GT(image.cols, 0);

  std::cout << "Input image size: " << image.cols << "x" << image.rows << std::endl;
  std::cout << "Using device: " << (detector_->device_.is_cuda() ? "CUDA" : "CPU") << std::endl;

  // Convert to RGB for inference
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

  auto start = std::chrono::high_resolution_clock::now();
  auto [boxes, scores, labels] = detector_->predict(image_rgb);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

  // Validate output tensors
  validate_detection_tensors(boxes, scores, labels);

  if (boxes.size(0) > 0) {
    validate_bounding_boxes(boxes, image.size());
    validate_scores_and_labels(scores, labels);

    std::cout << "Found " << boxes.size(0) << " detections" << std::endl;
  } else {
    std::cout << "No detections found" << std::endl;
  }

  // Test drawing functionality
  cv::Mat image_with_boxes = image.clone();
  detector_->draw_predictions(image_with_boxes, boxes, scores, labels, 0.5f);

  EXPECT_EQ(image_with_boxes.size(), image.size());
  EXPECT_EQ(image_with_boxes.type(), CV_8UC3);

  // Save results for visual inspection
  std::string device_suffix = detector_->device_.is_cuda() ? "_gpu" : "_cpu";
  save_detection_results(image_with_boxes, device_suffix);

  // Optional: Display results (comment out for automated testing)
  /*
  cv::imshow("Original", image);
  cv::imshow("Detections", image_with_boxes);
  cv::waitKey(0);
  cv::destroyAllWindows();
  */
}

//TEST_F(FCOSTorchBackendTest, TestMultipleInferences)
//{
//  cv::Mat image = create_test_image();
//  cv::Mat image_rgb;
//  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

//  const int num_iterations = 10;
//  std::vector<double> inference_times;
//  torch::Tensor first_boxes, first_scores, first_labels;

//  for (int i = 0; i < num_iterations; ++i) {
//    auto start = std::chrono::high_resolution_clock::now();
//    auto [boxes, scores, labels] = detector_->predict(image_rgb);
//    auto end = std::chrono::high_resolution_clock::now();

//    auto duration = std::chrono::duration<double, std::milli>(end - start);
//    inference_times.push_back(duration.count());

//    // Validate output consistency
//    validate_detection_tensors(boxes, scores, labels);

//    // Store first result for consistency check
//    if (i == 0) {
//      first_boxes = boxes.clone();
//      first_scores = scores.clone();
//      first_labels = labels.clone();
//    } else {
//      // Check if results are consistent (should be identical for same input)
//      EXPECT_EQ(boxes.size(0), first_boxes.size(0))
//        << "Number of detections should be consistent";

//      if (boxes.size(0) > 0 && first_boxes.size(0) > 0) {
//        // Check if boxes are approximately equal (allowing for small numerical differences)
//        torch::Tensor box_diff = torch::abs(boxes - first_boxes);
//        float max_box_diff = torch::max(box_diff).item<float>();
//        EXPECT_LT(max_box_diff, 1e-5f) << "Bounding boxes should be consistent";

//        // Check scores consistency
//        torch::Tensor score_diff = torch::abs(scores - first_scores);
//        float max_score_diff = torch::max(score_diff).item<float>();
//        EXPECT_LT(max_score_diff, 1e-5) << "Scores should be consistent";

//        // Check labels consistency (should be exact)
//        bool labels_equal = torch::equal(labels, first_labels);
//        EXPECT_TRUE(labels_equal) << "Labels should be identical";
//      }
//    }
//  }

//  // Calculate statistics
//  double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) /
//    inference_times.size();
//  double min_time = *std::min_element(inference_times.begin(), inference_times.end());
//  double max_time = *std::max_element(inference_times.begin(), inference_times.end());

//  std::cout << "Multiple inference statistics (" << (detector_->device_.is_cuda() ? "GPU" : "CPU") << "):" << std::endl;
//  std::cout << "  Average: " << avg_time << " ms" << std::endl;
//  std::cout << "  Min: " << min_time << " ms" << std::endl;
//  std::cout << "  Max: " << max_time << " ms" << std::endl;

//  // Performance expectations (adjust based on your hardware and model complexity)
//  if (detector_->device_.is_cuda()) {
//    EXPECT_LT(avg_time, 500.0) << "GPU inference should be reasonably fast";
//  } else {
//    EXPECT_LT(avg_time, 2000.0) << "CPU inference should complete within reasonable time";
//  }
//}

//TEST_F(FCOSTorchBackendTest, TestBenchmarkInference)
//{
//  cv::Mat image = create_test_image();
//  cv::Mat image_rgb;
//  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

//  const int warmup_iterations = 5;
//  const int benchmark_iterations = 50;

//  std::cout << "Running benchmark on " << (detector_->device_.is_cuda() ? "GPU" : "CPU") << std::endl;

//  // Warmup
//  std::cout << "Warming up..." << std::endl;
//  for (int i = 0; i < warmup_iterations; ++i) {
//    detector_->predict(image_rgb);
//  }

//  // Benchmark
//  std::cout << "Running benchmark..." << std::endl;
//  auto start = std::chrono::high_resolution_clock::now();

//  for (int i = 0; i < benchmark_iterations; ++i) {
//    detector_->predict(image_rgb);
//  }

//  auto end = std::chrono::high_resolution_clock::now();
//  auto total_duration = std::chrono::duration<double, std::milli>(end - start);

//  double avg_time = total_duration.count() / benchmark_iterations;
//  double fps = 1000.0 / avg_time;

//  std::cout << "Benchmark Results (" << (detector_->device_.is_cuda() ? "GPU" : "CPU") << "):" << std::endl;
//  std::cout << "Iterations: " << benchmark_iterations << std::endl;
//  std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
//  std::cout << "Average time per inference: " << avg_time << " ms" << std::endl;
//  std::cout << "Throughput: " << fps << " FPS" << std::endl;

//  // Basic performance check
//  EXPECT_GT(fps, 0.1) << "Throughput should be at least 0.1 FPS";
//}

//TEST_F(FCOSTorchBackendTest, TestInputValidation)
//{
//  // Test with empty image
//  cv::Mat empty_image;
//  EXPECT_THROW(detector_->predict(empty_image), std::exception);

//  // Test with different image sizes
//  std::vector<cv::Size> test_sizes = {
//    cv::Size(320, 240),
//    cv::Size(640, 480),
//    cv::Size(1024, 768),
//    cv::Size(100, 100),
//    cv::Size(1920, 1080)
//  };

//  for (const auto& size : test_sizes) {
//    cv::Mat test_image = create_test_image(size);
//    cv::Mat test_image_rgb;
//    cv::cvtColor(test_image, test_image_rgb, cv::COLOR_BGR2RGB);

//    EXPECT_NO_THROW({
//      auto [boxes, scores, labels] = detector_->predict(test_image_rgb);
//      validate_detection_tensors(boxes, scores, labels);

//      if (boxes.size(0) > 0) {
//        validate_bounding_boxes(boxes, size);
//        validate_scores_and_labels(scores, labels);
//      }
//    }) << "Failed for image size: " << size.width << "x" << size.height;
//  }
//}

//TEST_F(FCOSTorchBackendTest, TestDrawingFunction)
//{
//  cv::Mat image = create_test_image();
//  cv::Mat image_rgb;
//  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

//  auto [boxes, scores, labels] = detector_->predict(image_rgb);

//  // Test drawing with different confidence thresholds
//  std::vector<float> thresholds = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

//  for (float threshold : thresholds) {
//    cv::Mat image_copy = image.clone();

//    EXPECT_NO_THROW({
//      detector_->draw_predictions(image_copy, boxes, scores, labels, threshold);
//    }) << "Drawing failed for threshold: " << threshold;

//    EXPECT_EQ(image_copy.size(), image.size());
//    EXPECT_EQ(image_copy.type(), CV_8UC3);

//    // Count how many detections should be drawn
//    int expected_detections = 0;
//    if (scores.size(0) > 0) {
//      auto scores_a = scores.accessor<float, 1>();
//      for (int i = 0; i < scores.size(0); ++i) {
//        if (scores_a[i] >= threshold) {
//          expected_detections++;
//        }
//      }
//    }

//    std::cout << "Threshold " << threshold << ": " << expected_detections
//              << " detections should be drawn" << std::endl;
//  }

//  // Test with empty detections
//  torch::Tensor empty_boxes = torch::empty({0, 4}, torch::kFloat32);
//  torch::Tensor empty_scores = torch::empty({0}, torch::kFloat32);
//  torch::Tensor empty_labels = torch::empty({0}, torch::kInt64);

//  cv::Mat image_empty = image.clone();
//  EXPECT_NO_THROW({
//    detector_->draw_predictions(image_empty, empty_boxes, empty_scores, empty_labels);
//  }) << "Drawing should handle empty detections gracefully";
//}

//TEST_F(FCOSTorchBackendTest, TestTensorConversion)
//{
//  // Test the private mat_to_tensor method
//  cv::Mat test_image = create_test_image();

//  EXPECT_NO_THROW({
//    torch::Tensor tensor = detector_->mat_to_tensor(test_image);

//    // Check tensor properties
//    EXPECT_EQ(tensor.dim(), 4) << "Tensor should be 4D (batch, channels, height, width)";
//    EXPECT_EQ(tensor.size(0), 1) << "Batch size should be 1";
//    EXPECT_EQ(tensor.size(1), 3) << "Should have 3 channels";
//    EXPECT_EQ(tensor.size(2), test_image.rows) << "Height should match image height";
//    EXPECT_EQ(tensor.size(3), test_image.cols) << "Width should match image width";

//    // Check tensor is on correct device
//    EXPECT_EQ(tensor.device(), detector_->device_) << "Tensor should be on same device as model";

//    // Check tensor dtype
//    EXPECT_TRUE(tensor.dtype() == torch::kFloat32 || tensor.dtype() == torch::kFloat64)
//      << "Tensor should be float type";

//    // Check tensor values are in [0, 1] range (normalized)
//    float min_val = torch::min(tensor).item<float>();
//    float max_val = torch::max(tensor).item<float>();
//    EXPECT_GE(min_val, 0.0f) << "Tensor values should be >= 0";
//    EXPECT_LE(max_val, 1.0f) << "Tensor values should be <= 1";

//  }) << "mat_to_tensor conversion failed";
//}

//TEST_F(FCOSTorchBackendTest, TestCOCOClassLabels)
//{
//  // Test that COCO class labels are properly defined
//  EXPECT_GT(config::COCO_CLASSES.size(), 0) << "COCO classes should not be empty";
//  EXPECT_EQ(config::COCO_CLASSES[0], "__background__") << "First class should be background";

//  // Check some common classes exist
//  auto it_person = std::find(config::COCO_CLASSES.begin(), config::COCO_CLASSES.end(), "person");
//  EXPECT_NE(it_person, config::COCO_CLASSES.end()) << "Should contain 'person' class";

//  auto it_car = std::find(config::COCO_CLASSES.begin(), config::COCO_CLASSES.end(), "car");
//  EXPECT_NE(it_car, config::COCO_CLASSES.end()) << "Should contain 'car' class";

//  // Test with actual predictions to ensure labels are valid
//  cv::Mat image = create_test_image();
//  cv::Mat image_rgb;
//  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

//  auto [boxes, scores, labels] = detector_->predict(image_rgb);

//  if (labels.size(0) > 0) {
//    auto labels_a = labels.accessor<long, 1>();
//    for (int i = 0; i < labels.size(0); ++i) {
//      size_t label_idx = labels_a[i];
//      EXPECT_LT(label_idx, config::COCO_CLASSES.size())
//        << "Label index should be valid COCO class index";

//      std::string class_name = config::COCO_CLASSES[label_idx];
//      EXPECT_FALSE(class_name.empty()) << "Class name should not be empty";

//      std::cout << "Detection " << i << ": " << class_name
//                << " (confidence: " << scores.accessor<float, 1>()[i] << ")" << std::endl;
//    }
//  }
//}

//// Test with real image file if available
//TEST_F(FCOSTorchBackendTest, DISABLED_TestRealImage)
//{
//  cv::Mat image;
//  try {
//    image = load_test_image();
//  } catch (const std::exception& e) {
//    GTEST_SKIP() << "Real test image not available: " << e.what();
//  }

//  cv::Mat image_rgb;
//  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

//  auto start = std::chrono::high_resolution_clock::now();
//  auto [boxes, scores, labels] = detector_->predict(image_rgb);
//  auto end = std::chrono::high_resolution_clock::now();

//  auto duration = std::chrono::duration<double, std::milli>(end - start);
//  std::cout << "Real image inference time: " << duration.count() << " ms" << std::endl;

//  validate_detection_tensors(boxes, scores, labels);

//  if (boxes.size(0) > 0) {
//    validate_bounding_boxes(boxes, image.size());
//    validate_scores_and_labels(scores, labels);

//    // Draw and save results
//    cv::Mat image_with_detections = image.clone();
//    detector_->draw_predictions(image_with_detections, boxes, scores, labels, 0.5f);
//    save_detection_results(image_with_detections, "_real_image");

//    std::cout << "Found " << boxes.size(0) << " detections in real image" << std::endl;
//  }
//}

//// Test device switching if both CPU and GPU are available
//TEST_F(FCOSTorchBackendTest, DISABLED_TestDeviceSwitching)
//{
//  if (!torch::cuda::is_available()) {
//    GTEST_SKIP() << "CUDA not available, skipping device switching test";
//  }

//  cv::Mat image = create_test_image();
//  cv::Mat image_rgb;
//  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

//  // Note: This would require modifying the constructor to accept device parameter
//  // or creating separate models for CPU and GPU testing

//  std::cout << "Device switching test would require constructor modifications" << std::endl;
//  std::cout << "Current device: " << (detector_->device_.is_cuda() ? "CUDA" : "CPU") << std::endl;

//  // For now, just verify current device works
//  auto [boxes, scores, labels] = detector_->predict(image_rgb);
//  validate_detection_tensors(boxes, scores, labels);
//}
