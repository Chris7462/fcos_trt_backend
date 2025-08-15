#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <vector>

// OpenCV includes
#include <opencv2/core.hpp>


namespace fcos_trt_backend
{
/**
 * @brief Detection results structure
 * @details Contains bounding boxes, confidence scores, and class labels
 *          for object detection results. This is the common interface
 *          between postprocessor and utility functions.
 */
struct Detections
{
  std::vector<cv::Rect2f> boxes;  ///< Bounding boxes in image coordinates
  std::vector<float> scores;      ///< Confidence scores [0.0, 1.0]
  std::vector<int> labels;        ///< Class labels (COCO category IDs)

///**
// * @brief Get number of detections
// */
//size_t size() const noexcept {
//  return boxes.size();
//}

///**
// * @brief Check if detection results are empty
// */
//bool empty() const noexcept {
//  return boxes.empty();
//}

///**
// * @brief Reserve space for expected number of detections
// * @param capacity Expected number of detections
// */
//void reserve(size_t capacity) {
//  boxes.reserve(capacity);
//  scores.reserve(capacity);
//  labels.reserve(capacity);
//}

///**
// * @brief Clear all detection results
// */
//void clear() noexcept {
//  boxes.clear();
//  scores.clear();
//  labels.clear();
//}

///**
// * @brief Add a single detection
// * @param box Bounding box
// * @param score Confidence score
// * @param label Class label
// */
//void add_detection(const cv::Rect2f& box, float score, int label) {
//  boxes.push_back(box);
//  scores.push_back(score);
//  labels.push_back(label);
//}

///**
// * @brief Validate that all vectors have the same size
// * @return true if valid, false otherwise
// */
//bool is_valid() const noexcept {
//  return (boxes.size() == scores.size()) && (scores.size() == labels.size());
//}
};

} // namespace fcos_trt_backend
