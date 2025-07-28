#pragma once

#include <cuda_runtime.h>

namespace fcos_trt_backend
{

/**
 * @brief Initialize constant memory for ImageNet normalization
 * This should be called once during initialization
 */
void initialize_mean_std_constants();

/**
 * @brief Launch CUDA kernel for image normalization
 *
 * Performs ImageNet normalization: (pixel/255 - mean) / std
 * Input: HWC format (height, width, channels)
 * Output: CHW format (channels, height, width)
 *
 * @param input Input image data in HWC format (range 0-1)
 * @param output Output normalized data in CHW format
 * @param width Image width
 * @param height Image height
 * @param stream CUDA stream for async execution
 */
void launch_normalize_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    cudaStream_t stream);

} // namespace fcos_trt_backend
