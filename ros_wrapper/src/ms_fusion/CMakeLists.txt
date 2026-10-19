cmake_minimum_required(VERSION 3.8)
project(ms_fusion)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Precompiled headers (target_precompile_headers requires CMake 3.16).
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
  option(MSF_USE_PCH "Use precompiled headers for ms_fusion" ON)
else()
  set(MSF_USE_PCH OFF CACHE BOOL "Use precompiled headers for ms_fusion")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_srvs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(msf_core REQUIRED)
find_package(glog REQUIRED)

# Axera IVPS 共享内存取图（板端默认开启，可通过 -DMSF_USE_AXERA_IVPS=OFF 关闭）
if(NOT DEFINED MSF_USE_AXERA_IVPS)
  if(EXISTS "/soc/lib/libax_ivps.so")
    set(MSF_USE_AXERA_IVPS ON)
  else()
    set(MSF_USE_AXERA_IVPS OFF)
  endif()
endif()
option(MSF_USE_AXERA_IVPS "Enable IVPS shared-memory image grab via third_party/axera" ${MSF_USE_AXERA_IVPS})

set(MSF_AXERA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/axera")

set(dependencies
  rclcpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
  std_srvs)

if(NOT MSF_USE_AXERA_IVPS)
  find_package(message_filters REQUIRED)
  list(APPEND dependencies message_filters)
endif()

message(STATUS "MSF motion source: sensor_msgs/Imu + nav_msgs/Odometry")

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  )

add_library(${PROJECT_NAME}
  src/ms_fusion_node.cpp
  src/matched_point_cloud.cpp
  src/msf_ivps_grabber.cpp)

if(MSF_USE_PCH)
  target_precompile_headers(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/ms_fusion_pch.hpp>")
endif()

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen ${OpenCV_LIBS} msf_core::msf_core yaml-cpp glog::glog)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})

if(MSF_USE_AXERA_IVPS)
  if(NOT EXISTS "${MSF_AXERA_ROOT}/include/grab_images_api.h")
    message(FATAL_ERROR "MSF_USE_AXERA_IVPS=ON but third_party/axera not found at ${MSF_AXERA_ROOT}")
  endif()
  message(STATUS "MSF Axera IVPS image grab: enabled")
  target_compile_definitions(${PROJECT_NAME} PUBLIC MSF_USE_AXERA_IVPS)
  # ## PUBLIC：ms_fusion_node 可执行文件（app/ms_fusion_app.cpp）通过
  # ## ms_fusion_node.h 间接 include 到 msf_core 的 ivps_frame_handle.h，
  # ## 同样需要 third_party/axera/include，PRIVATE 只对本库自己的源文件生效。
  target_include_directories(${PROJECT_NAME} PUBLIC ${MSF_AXERA_ROOT}/include)
  target_link_directories(${PROJECT_NAME} PUBLIC /soc/lib)
  target_link_options(${PROJECT_NAME} PUBLIC "LINKER:-rpath-link,/soc/lib")
  target_link_libraries(${PROJECT_NAME} ax_sys ax_ivps)
else()
  message(STATUS "MSF Axera IVPS image grab: disabled (topic mode only)")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_executable(ms_fusion_node app/ms_fusion_app.cpp)
target_link_libraries(ms_fusion_node ${PROJECT_NAME})

if(MSF_USE_AXERA_IVPS OR MSF_USE_AXERA_SUPERPOINT)
  set_target_properties(ms_fusion_node PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib:\$ORIGIN/..")
  ament_environment_hooks(
    "${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/axera_lib_paths.sh.in")
endif()

install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS ms_fusion_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY include launch
  DESTINATION share/${PROJECT_NAME})

# Runtime uses a single install path: share/ms_fusion/config/msf_robot.yaml
# Per-device / per-dataset profiles live in repo calib_files/<name>/ and are copied
# onto this path before running (not installed from source).
install(
  FILES config/msf_robot.yaml
  DESTINATION share/${PROJECT_NAME}/config)

# Optional alternate platform config (i2nav_robot.launch.py)
install(
  FILES config/i2nav_robot.yaml
  DESTINATION share/${PROJECT_NAME}/config)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/axera/asset/models/
  DESTINATION share/${PROJECT_NAME}/models
  FILES_MATCHING
  PATTERN "*.axmodel")

if(USE_ONNX)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/onnxruntime/models/
    DESTINATION share/${PROJECT_NAME}/models
    FILES_MATCHING PATTERN "*.onnx")
endif()

# DBoW3 vocab: put *.dbow3 under ms_fusion_loop/voc/, colcon installs to
#   install/ms_fusion/share/ms_fusion/models/
# Config examples:
#   loop.voc_path: ../models/xfeat_voc_robot_0729.dbow3
#   loop.voc_path: ../models/xfeat_voc_stereoimu_0729.dbow3
set(MSF_LOOP_VOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ms_fusion_loop/voc")
if(EXISTS "${MSF_LOOP_VOC_DIR}")
  install(
    DIRECTORY ${MSF_LOOP_VOC_DIR}/
    DESTINATION share/${PROJECT_NAME}/models
    FILES_MATCHING
    PATTERN "*.dbow3")
endif()

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
endif()

ament_package()
