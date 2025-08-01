# Config.cmake
# Shared configuration for model generation and testing
# This file contains all model-related configuration variables

# Model configuration - change these variables to use different models
set(MODEL_NAME "fcos_resnet50_fpn_374x1238" CACHE STRING "Base name of the model")
set(EXPORT_SCRIPT "export_fcos_to_onnx.py" CACHE STRING "Python script for ONNX export")

# Derived file names (automatically generated from MODEL_NAME)
set(ONNX_FILE "${MODEL_NAME}.onnx")
set(ENGINE_FILE "${MODEL_NAME}.engine")

# Common directory paths
set(ONNXS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/onnxs)
set(ENGINES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/engines)
set(SCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/script)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# Full file paths
set(ONNX_PATH ${ONNXS_DIR}/${ONNX_FILE})
set(ENGINE_PATH ${ENGINES_DIR}/${ENGINE_FILE})
set(EXPORT_SCRIPT_PATH ${SCRIPTS_DIR}/${EXPORT_SCRIPT})

# Test configuration
set(TEST_IMAGE_FILES
  image_000.png
  image_001.png
  image_002.png
  CACHE STRING "List of test image files")
