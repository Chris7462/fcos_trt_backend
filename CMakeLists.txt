cmake_minimum_required(VERSION 3.18)
project(fcos_trt_backend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT paths - adjust these according to your TensorRT installation
set(TENSORRT_ROOT "/usr/local/TensorRT-10.9.0.34" CACHE PATH "TensorRT root directory")

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib
    PATH_SUFFIXES lib lib64)

find_library(TENSORRT_ONNX_PARSER nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib
    PATH_SUFFIXES lib lib64)

# Check if all required libraries are found
if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT include directory not found. Please set TENSORRT_ROOT correctly.")
endif()

if(NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT library not found. Please set TENSORRT_ROOT correctly.")
endif()

if(NOT TENSORRT_ONNX_PARSER)
    message(FATAL_ERROR "TensorRT ONNX parser library not found. Please set TENSORRT_ROOT correctly.")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create library for FCOS TensorRT backend
add_library(fcos_trt_backend
    src/fcos_trt_backend.cpp
)

# Specify include directories for the target
target_include_directories(fcos_trt_backend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TENSORRT_INCLUDE_DIR}  # Make this PUBLIC so downstream packages can use it
    ${OpenCV_INCLUDE_DIRS}
    #${CUDAToolkit_INCLUDE_DIRS}  # Not needed as CUDA::cudart in the target_link_libraries will provide CUDA includes for Modern CMake
)

target_link_libraries(fcos_trt_backend
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER}
    ${CUDA_LIBRARIES}
    cudart
)

# Add main executable
add_executable(fcos_tensorrt
  src/fcos_trt_backend.cpp
  src/main.cpp)

# Specify include directories for the target
target_include_directories(fcos_tensorrt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${TENSORRT_INCLUDE_DIR}  # Make this PUBLIC so downstream packages can use it
    ${OpenCV_INCLUDE_DIRS}
    #${CUDAToolkit_INCLUDE_DIRS}  # Not needed as CUDA::cudart in the target_link_libraries will provide CUDA includes for Modern CMake
)

# Link main executable with the library
target_link_libraries(fcos_tensorrt
    fcos_trt_backend
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_PARSER}
    ${CUDA_LIBRARIES}
    cudart
)

# Compiler-specific options
target_compile_options(fcos_trt_backend PRIVATE
    -Wall
    -Wextra
    -O3
    -DWITH_OPENCV
)

target_compile_options(fcos_tensorrt PRIVATE
    -Wall
    -Wextra
    -O3
    -DWITH_OPENCV
)

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "TensorRT include dir: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "TensorRT ONNX parser: ${TENSORRT_ONNX_PARSER}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

# # Optional: Enable testing if desired
# option(BUILD_TESTS "Build test executables" OFF)
# if(BUILD_TESTS)
#     enable_testing()
#     find_package(GTest REQUIRED)

#     add_executable(test_fcos_trt_backend test_fcos_trt_backend.cpp)
#     target_link_libraries(test_fcos_trt_backend
#         fcos_trt_backend
#         ${GTEST_LIBRARIES}
#         pthread
#     )

#     add_test(NAME fcos_backend_test COMMAND test_fcos_trt_backend)
# endif()

# # Create a simple test target for manual testing
# add_test(NAME fcos_inference_test
#     COMMAND fcos_tensorrt engines/fcos_resnet50_fpn_374x1238.engine test_image.jpg
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
