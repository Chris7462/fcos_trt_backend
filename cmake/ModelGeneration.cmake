# ModelGeneration.cmake
# This file handles automatic ONNX and TensorRT engine generation

# Include shared model configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

# Create directories for models and engines
file(MAKE_DIRECTORY ${ONNXS_DIR})
file(MAKE_DIRECTORY ${ENGINES_DIR})

# Find Python3
find_program(PYTHON3_EXECUTABLE python3 REQUIRED)
if(NOT PYTHON3_EXECUTABLE)
  message(FATAL_ERROR "Python3 not found. Please install Python3.")
endif()

# Find trtexec
find_program(TRTEXEC_EXECUTABLE trtexec
  HINTS ${TENSORRT_ROOT}/bin ${CUDA_TOOLKIT_ROOT_DIR}/bin
  PATHS /usr/local/bin /usr/bin)
if(NOT TRTEXEC_EXECUTABLE)
  message(FATAL_ERROR "trtexec not found. Please ensure TensorRT is properly installed and trtexec is in PATH.")
endif()

# Custom target to generate ONNX model
add_custom_command(
  OUTPUT ${ONNX_PATH}
  COMMAND ${PYTHON3_EXECUTABLE} ${EXPORT_SCRIPT_PATH} --output-dir ${ONNXS_DIR}
  DEPENDS ${EXPORT_SCRIPT_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating ONNX format: ${ONNX_FILE}..."
  VERBATIM
)

# Custom target to generate TensorRT engine
add_custom_command(
  OUTPUT ${ENGINE_PATH}
  COMMAND ${TRTEXEC_EXECUTABLE} --onnx=${ONNX_PATH} --saveEngine=${ENGINE_PATH}
          --memPoolSize=workspace:4096 --fp16 --verbose
  DEPENDS ${ONNX_PATH}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating TensorRT engine: ${ENGINE_FILE}..."
  VERBATIM
)

# Create custom targets that can be built
add_custom_target(generate_engine
  DEPENDS ${ENGINE_PATH}
)

# Function to add model generation dependency to a target
function(add_model_generation_dependency target_name)
  add_dependencies(${target_name} generate_engine)
endfunction()

# Install generated models and engines
# Check if directories exist before installing
if(EXISTS ${ONNXS_DIR})
  install(DIRECTORY ${ONNXS_DIR}/
    DESTINATION share/${PROJECT_NAME}/models
    FILES_MATCHING PATTERN "*.onnx")
endif()

if(EXISTS ${ENGINES_DIR})
  install(DIRECTORY ${ENGINES_DIR}/
    DESTINATION share/${PROJECT_NAME}/engines
    FILES_MATCHING PATTERN "*.engine")
endif()
