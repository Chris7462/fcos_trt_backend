#include <stdexcept>

#include "fcos_trt_backend/normalize_kernel.hpp"
#include "fcos_trt_backend/config.hpp"

namespace fcos_trt_backend
{

// Constant memory for normalization parameters
__constant__ float d_mean[3];
__constant__ float d_std[3];

void initialize_mean_std_constants()
{
    // Copy ImageNet normalization constants to GPU constant memory
    cudaMemcpyToSymbol(d_mean, config::MEAN.data(), 3 * sizeof(float));
    cudaMemcpyToSymbol(d_std, config::STDDEV.data(), 3 * sizeof(float));
}

__global__ void normalize_kernel(
    const float* input,
    float* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int pixel_idx = y * width + x;

    // Input is HWC format, output is CHW format
    for (int c = 0; c < 3; ++c) {
        int input_idx = pixel_idx * 3 + c;  // HWC: (H*W*C)
        int output_idx = c * height * width + pixel_idx;  // CHW: (C*H*W)

        // Normalize: (pixel - mean) / std
        // Input is already in range [0, 1] from preprocessing
        output[output_idx] = (input[input_idx] - d_mean[c]) / d_std[c];
    }
}

void launch_normalize_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    cudaStream_t stream)
{
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // Launch kernel
    normalize_kernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, width, height
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

} // namespace fcos_trt_backend
