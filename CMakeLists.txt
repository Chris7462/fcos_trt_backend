cmake_minimum_required(VERSION 3.8)
project(fcos_torch_backend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

add_library(fcos_torch_backend
  src/fcos_torch_backend.cpp)

target_include_directories(fcos_torch_backend
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(fcos_torch_backend
  PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    -Wl,--no-as-needed TorchVision::TorchVision)

target_compile_features(fcos_torch_backend PUBLIC cxx_std_17)

# Optionally install the library
install(TARGETS fcos_torch_backend
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV Torch)

# Export include directories for downstream packages
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_fcos_torch_backend
    test/test_fcos_torch_backend.cpp)

  target_link_libraries(test_fcos_torch_backend
    fcos_torch_backend
    gtest_main)

endif()

ament_package()
