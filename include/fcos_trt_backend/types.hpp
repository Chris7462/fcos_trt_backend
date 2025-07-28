#pragma once

#include <vector>
#include <cstdint>

namespace fcos_trt_backend
{

// Detection result structure
struct Detection
{
    float x1, y1, x2, y2;  // Bounding box coordinates
    float score;           // Confidence score
    int32_t label;         // Class label

    Detection() : x1(0), y1(0), x2(0), y2(0), score(0), label(0) {}

    Detection(float x1_, float y1_, float x2_, float y2_, float score_, int32_t label_)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(score_), label(label_) {}
};

// Collection of detections for one image
using Detections = std::vector<Detection>;

// Raw model outputs structure (matches Python FCOSBackbone output)
struct RawOutputs
{
    float* cls_logits;           // [N, num_anchors, num_classes]
    float* bbox_regression;      // [N, num_anchors, 4]
    float* bbox_ctrness;         // [N, num_anchors, 1]
    float* anchors;              // [num_anchors, 4]
    int32_t* image_sizes;        // [N, 2] - processed image sizes
    int32_t* original_image_sizes; // [N, 2] - original image sizes
    int32_t* num_anchors_per_level; // [num_levels] - anchors per pyramid level

    int32_t num_anchors;
    int32_t num_classes;
    int32_t num_levels;

    RawOutputs() : cls_logits(nullptr), bbox_regression(nullptr), bbox_ctrness(nullptr),
                   anchors(nullptr), image_sizes(nullptr), original_image_sizes(nullptr),
                   num_anchors_per_level(nullptr), num_anchors(0), num_classes(0), num_levels(0) {}
};

} // namespace fcos_trt_backend
