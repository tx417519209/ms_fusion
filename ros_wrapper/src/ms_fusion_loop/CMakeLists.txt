cmake_minimum_required(VERSION 3.8)
project(ms_fusion_loop)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(msf_core REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DBoW3/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DUtils
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

set(DBOW3_SOURCES
  src/ThirdParty/DBoW3/src/BowVector.cpp
  src/ThirdParty/DBoW3/src/Database.cpp
  src/ThirdParty/DBoW3/src/DescManip.cpp
  src/ThirdParty/DBoW3/src/FeatureVector.cpp
  src/ThirdParty/DBoW3/src/QueryResults.cpp
  src/ThirdParty/DBoW3/src/ScoringObject.cpp
  src/ThirdParty/DBoW3/src/Vocabulary.cpp
  src/ThirdParty/DBoW3/src/quicklz.c
  src/ThirdParty/DUtils/Random.cpp
  src/ThirdParty/DUtils/Timestamp.cpp
)

add_library(ms_fusion_dbow3 STATIC ${DBOW3_SOURCES})
target_include_directories(ms_fusion_dbow3 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DBoW3/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DUtils)
target_link_libraries(ms_fusion_dbow3 PUBLIC ${OpenCV_LIBS})

add_executable(ms_fusion_loop_node
  src/pose_graph_node.cpp
  src/pose_graph.cpp
  src/keyframe.cpp
  src/parameters.cpp
  src/utility/CameraPoseVisualization.cpp
  src/utility/utility.cpp
  src/utility/loop_timing.cpp
)

target_compile_definitions(ms_fusion_loop_node PRIVATE
  MSFUSION_LOOP_PKG_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

ament_target_dependencies(ms_fusion_loop_node
  rclcpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
)

target_link_libraries(ms_fusion_loop_node
  ms_fusion_dbow3
  msf_core::msf_core
  Eigen3::Eigen
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  yaml-cpp
)

install(TARGETS ms_fusion_loop_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch voc
  DESTINATION share/${PROJECT_NAME})
install(FILES tools/README.md
  DESTINATION share/${PROJECT_NAME}/tools)

# ---- Optional offline image collection / vocabulary training tools ----
option(MSF_BUILD_LOOP_TOOLS "Build ms_fusion_loop offline collection/training tools" OFF)
if(MSF_BUILD_LOOP_TOOLS)
set(MSF_AXERA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party_axera")
if(NOT DEFINED MSF_USE_AXERA_SUPERPOINT)
  if(EXISTS "/soc/lib/libax_engine.so")
    set(MSF_USE_AXERA_SUPERPOINT ON)
  else()
    set(MSF_USE_AXERA_SUPERPOINT OFF)
  endif()
endif()
option(MSF_USE_AXERA_SUPERPOINT "Build Axera-based vocab training tools" ${MSF_USE_AXERA_SUPERPOINT})

# Collect gray PNGs from IVPS (no NPU); same SoC deploy path as ms_fusion
if(NOT DEFINED MSF_USE_AXERA_IVPS)
  if(EXISTS "/soc/lib/libax_ivps.so")
    set(MSF_USE_AXERA_IVPS ON)
  else()
    set(MSF_USE_AXERA_IVPS OFF)
  endif()
endif()
option(MSF_USE_AXERA_IVPS "Build IVPS collect_gray_images tool" ${MSF_USE_AXERA_IVPS})

if(MSF_USE_AXERA_IVPS AND EXISTS "${MSF_AXERA_ROOT}/include/grab_images_api.h")
  message(STATUS "ms_fusion_loop: building collect_gray_images (IVPS ON)")
  add_executable(collect_gray_images tools/collect_gray_images.cpp)
  target_include_directories(collect_gray_images PRIVATE ${MSF_AXERA_ROOT}/include)
  target_link_directories(collect_gray_images PRIVATE /soc/lib)
  target_link_options(collect_gray_images PRIVATE "LINKER:-rpath-link,/soc/lib")
  target_link_libraries(collect_gray_images
    # 日志宏 MSF_*LOG（runtime_log.h）在 msf_core 里实现
    msf_core::msf_core
    ${OpenCV_LIBS}
    ax_sys
    ax_ivps)
  set_target_properties(collect_gray_images PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib:\$ORIGIN/..")
  install(TARGETS collect_gray_images
    DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "ms_fusion_loop: skip collect_gray_images (need SoC Axera IVPS)")
endif()

if(MSF_USE_AXERA_SUPERPOINT)
  message(STATUS "ms_fusion_loop: building train_superpoint_voc / train_xfeat_voc (Axera ON)")

  add_executable(train_superpoint_voc
    tools/train_superpoint_voc.cpp
    tools/train_voc_common.cpp)
  target_include_directories(train_superpoint_voc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DBoW3/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DUtils)
  target_compile_definitions(train_superpoint_voc PRIVATE MSF_USE_AXERA_SUPERPOINT)
  target_link_libraries(train_superpoint_voc
    ms_fusion_dbow3
    msf_core::msf_core
    Eigen3::Eigen
    ${OpenCV_LIBS})
  set_target_properties(train_superpoint_voc PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib:\$ORIGIN/..")

  add_executable(train_xfeat_voc
    tools/train_xfeat_voc.cpp
    tools/train_voc_common.cpp)
  target_include_directories(train_xfeat_voc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DBoW3/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ThirdParty/DUtils)
  target_compile_definitions(train_xfeat_voc PRIVATE MSF_USE_AXERA_SUPERPOINT)
  target_link_libraries(train_xfeat_voc
    ms_fusion_dbow3
    msf_core::msf_core
    Eigen3::Eigen
    ${OpenCV_LIBS})
  set_target_properties(train_xfeat_voc PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib:\$ORIGIN/..")

  install(TARGETS train_superpoint_voc train_xfeat_voc
    DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "ms_fusion_loop: skip vocab training tools (need SoC Axera: MSF_USE_AXERA_SUPERPOINT=ON)")
endif()
else()
  message(STATUS "ms_fusion_loop: offline collection/training tools disabled (MSF_BUILD_LOOP_TOOLS=OFF)")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
