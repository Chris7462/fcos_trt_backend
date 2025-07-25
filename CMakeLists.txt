cmake_minimum_required(VERSION 3.16)
project(FCOSTensorRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set TensorRT paths (adjust these paths according to your installation)
set(TENSORRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT installation directory")

# Find TensorRT libraries
find_library(TENSORRT_LIBRARY_NVINFER nvinfer
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
  PATH_SUFFIXES lib lib64
)

find_library(TENSORRT_LIBRARY_NVINFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
  PATH_SUFFIXES lib lib64
)

find_library(TENSORRT_LIBRARY_NVONNXPARSER nvonnxparser
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
  PATH_SUFFIXES lib lib64
)

# Check if TensorRT libraries were found
if(NOT TENSORRT_LIBRARY_NVINFER)
  message(FATAL_ERROR "Could not find nvinfer library")
endif()

if(NOT TENSORRT_LIBRARY_NVINFER_PLUGIN)
  message(FATAL_ERROR "Could not find nvinfer_plugin library")
endif()

if(NOT TENSORRT_LIBRARY_NVONNXPARSER)
  message(FATAL_ERROR "Could not find nvonnxparser library")
endif()

# Include directories
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_ROOT}/include
)

# Create executable
add_executable(fcos_tensorrt
  src/fcos_trt_backend.cpp
)

# Link libraries
target_link_libraries(fcos_tensorrt
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_NVINFER}
  ${TENSORRT_LIBRARY_NVINFER_PLUGIN}
  ${TENSORRT_LIBRARY_NVONNXPARSER}
  cudart
)

# Set CUDA properties
set_target_properties(fcos_tensorrt PROPERTIES
  CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90"
)

# Compiler flags
target_compile_options(fcos_tensorrt PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Print information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "TensorRT root: ${TENSORRT_ROOT}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_LIBRARY_NVINFER}")
message(STATUS "TensorRT nvonnxparser: ${TENSORRT_LIBRARY_NVONNXPARSER}")
