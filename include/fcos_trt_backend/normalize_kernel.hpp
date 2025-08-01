#pragma once

#include <cuda_runtime.h>


namespace fcos_trt_backend
{
void initialize_mean_std_constants();

// Host function wrapper for CUDA kernel
void launch_normalize_kernel(
  const float * input_data,
  float * output_data,
  int width, int height,
  cudaStream_t stream);

} // namespace fcos_trt_backend
