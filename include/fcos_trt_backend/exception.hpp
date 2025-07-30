#pragma once

// C++ standard library version: This project uses the C++17 standard library.
#include <stdexcept>

// CUDA includes
#include <cuda_runtime.h>


namespace fcos_trt_backend
{

// Custom exception classes
class TensorRTException : public std::runtime_error
{
public:
  explicit TensorRTException(const std::string & message)
  : std::runtime_error("TensorRT Error: " + message) {}
};

class CudaException : public std::runtime_error
{
public:
  explicit CudaException(const std::string & message, cudaError_t error)
  : std::runtime_error("CUDA Error: " + message + " (" + cudaGetErrorString(error) + ")") {}
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      throw CudaException(#call, error); \
    } \
  } while(0)

} // namespace fcos_trt_backend
