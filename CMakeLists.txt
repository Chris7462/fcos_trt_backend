cmake_minimum_required(VERSION 3.12)
project(FCOS_TensorRT LANGUAGES CXX CUDA)

# Set C++ standard to match your code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set TensorRT paths (adjust these paths according to your installation)
set(TENSORRT_ROOT "/usr/local/TensorRT-10.9.0.34" CACHE PATH "TensorRT installation directory")

# Alternative common paths - uncomment the one that matches your installation
# set(TENSORRT_ROOT "/opt/tensorrt" CACHE PATH "TensorRT installation directory")
# set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation directory")

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_PARSER nvonnxparser
  HINTS ${TENSORRT_ROOT} ${TENSORRT_ROOT}/lib ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

# Check if TensorRT was found
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER OR NOT TENSORRT_LIBRARY_PARSER)
  message(FATAL_ERROR "TensorRT not found! Please set TENSORRT_ROOT to your TensorRT installation directory")
endif()

message(STATUS "Found TensorRT:")
message(STATUS "  Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  Libraries: ${TENSORRT_LIBRARY_INFER}")
message(STATUS "             ${TENSORRT_LIBRARY_PARSER}")

# Create executable with correct source paths
add_executable(fcos_inference
  src/fcos_trt_backend.cpp
  src/normalize_kernel.cu
  src/main.cpp
)

# Include directories
target_include_directories(fcos_inference
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(fcos_inference
  ${OpenCV_LIBS}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_INFER_PLUGIN}
  ${TENSORRT_LIBRARY_PARSER}
  ${CUDA_LIBRARIES}
  CUDA::cudart
)

# Compiler-specific options
target_compile_definitions(fcos_inference PRIVATE API_EXPORTS)

# Set properties for both C++ and CUDA
set_target_properties(fcos_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "50;60;70;75;80;86"  # Adjust based on your GPU
)

# CUDA-specific compiler flags (optional, for optimization)
set_property(TARGET fcos_inference PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Print build information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Installation (optional)
install(TARGETS fcos_inference DESTINATION bin)

# Print usage instructions
message(STATUS "")
message(STATUS "Build instructions:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./fcos_inference <engine_path> <image_path>")
message(STATUS "  Example: ./fcos_inference ../engines/fcos_resnet50_fpn_374x1238.engine ../script/image_000.png")
message(STATUS "")
