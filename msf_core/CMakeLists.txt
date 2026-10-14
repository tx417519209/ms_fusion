cmake_minimum_required(VERSION 3.8)
project(msf_core)

find_package(ament_cmake REQUIRED)

add_compile_options(-std=c++20 -fPIC)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Ceres REQUIRED)
find_package(fmt REQUIRED)
# 系统 spdlog（与 ROS rcl_logging_spdlog 同一份），日志宏后端，见 runtime_log.cpp
find_package(spdlog REQUIRED)

message(STATUS "Ceres version: ${CERES_VERSION}")

add_subdirectory(third_party/Sophus EXCLUDE_FROM_ALL)

# Axera SuperPoint 支持（板端默认开启，可通过 -DMSF_USE_AXERA_SUPERPOINT=OFF 关闭）
if(NOT DEFINED MSF_USE_AXERA_SUPERPOINT)
  if(EXISTS "/soc/lib/libax_engine.so")
    set(MSF_USE_AXERA_SUPERPOINT ON)
  else()
    set(MSF_USE_AXERA_SUPERPOINT OFF)
  endif()
endif()
option(MSF_USE_AXERA_SUPERPOINT "Enable Axera NPU SuperPoint feature detector" ${MSF_USE_AXERA_SUPERPOINT})

# Axera IVPS 零拷贝取图句柄（ImageMessage::image0/1 用 IvpsFramePtr 而非 cv::Mat）。
# 与 ros_wrapper/src/ms_fusion/CMakeLists.txt 里的同名检测逻辑保持一致：msf_core
# 被 ms_fusion 用 add_subdirectory 内嵌构建时，这里的 option() 会直接复用 ms_fusion
# 那边已经缓存好的值；msf_core 独立被 colcon 构建（如给 ms_fusion_loop 用）时，
# 靠同样的板端探测逻辑独立算出一致的默认值，避免两边对 ImageMessage 的内存布局
# 编译出不一致的结果（ODR 风险）。
if(NOT DEFINED MSF_USE_AXERA_IVPS)
  if(EXISTS "/soc/lib/libax_ivps.so")
    set(MSF_USE_AXERA_IVPS ON)
  else()
    set(MSF_USE_AXERA_IVPS OFF)
  endif()
endif()
option(MSF_USE_AXERA_IVPS "Enable IVPS zero-copy frame handle in ImageMessage" ${MSF_USE_AXERA_IVPS})

# 必须规范化成不含 ".." 的绝对路径：IVPS 开启时它会进入导出的
# INTERFACE_INCLUDE_DIRECTORIES，CMake 按字符串前缀判断，"msf_core/../" 会被当成
# 源码目录内路径，生成阶段直接报错。
get_filename_component(MSF_AXERA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party_axera" ABSOLUTE)
if(MSF_USE_AXERA_SUPERPOINT)
  if(NOT EXISTS "${MSF_AXERA_ROOT}/include/infer_api.h")
    message(FATAL_ERROR "MSF_USE_AXERA_SUPERPOINT=ON but third_party_axera not found at ${MSF_AXERA_ROOT}")
  endif()
  message(STATUS "MSF Axera SuperPoint: enabled")
else()
  message(STATUS "MSF Axera SuperPoint: disabled")
endif()

include_directories(
  ${TBB_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/interface/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/manager/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vision/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/preintegration/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/optimization/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/factors/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Precompiled headers（target_precompile_headers 需要 CMake 3.16）
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16)
  option(MSF_CORE_USE_PCH "Use precompiled headers for msf_core" ON)
else()
  set(MSF_CORE_USE_PCH OFF CACHE BOOL "Use precompiled headers for msf_core")
endif()
if(MSF_CORE_USE_PCH)
  target_precompile_headers(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/src/msf_core_pch.hpp>")
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${OpenCV_LIBS} ${LAPACK_LIBRARIES}  ${BLAS_LIBRARIES}
  TBB::tbb Eigen3::Eigen yaml-cpp Ceres::ceres Sophus::Sophus
  # 日志后端只在 runtime_log.cpp 使用，公开头文件不暴露 spdlog
  spdlog::spdlog)

if(MSF_USE_AXERA_SUPERPOINT)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MSF_USE_AXERA_SUPERPOINT)
  target_include_directories(${PROJECT_NAME} PRIVATE ${MSF_AXERA_ROOT}/include)
  target_link_directories(${PROJECT_NAME} PUBLIC /soc/lib)
  target_link_options(${PROJECT_NAME} PUBLIC "LINKER:-rpath-link,/soc/lib")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    ax_sys
    ax_engine
    ax_interpreter
    ax_ivps)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib")
endif()

if(MSF_USE_AXERA_IVPS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MSF_USE_AXERA_IVPS)
  # ## PUBLIC 而不是 PRIVATE：IvpsFrameHandle 的完整定义（含 ax_sys_api.h/
  # ## ax_global_type.h 里的类型）直接嵌在 ImageMessage 等公开头文件里，任何
  # ## 链接 msf_core 的下游目标（比如 ms_fusion_node 可执行文件自己的
  # ## app/ms_fusion_app.cpp）编译时也需要能找到这些头文件，PRIVATE 只对
  # ## msf_core 自己的源文件生效，会导致下游目标报 ax_sys_api.h 找不到。
  target_include_directories(${PROJECT_NAME} PUBLIC ${MSF_AXERA_ROOT}/include)
  target_link_directories(${PROJECT_NAME} PUBLIC /soc/lib)
  target_link_options(${PROJECT_NAME} PUBLIC "LINKER:-rpath-link,/soc/lib")
  target_link_libraries(${PROJECT_NAME} PRIVATE ax_sys)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib")
endif()

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/interface>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Export one canonical core target. Consumers must use this installed target rather than embedding
# this directory with add_subdirectory(), otherwise Ceres and all core sources compile once
# per consumer package.
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES
  interface/config_params.h
  interface/eigen_utils.hpp
  interface/ms_fusion_interface.h
  interface/ms_fusion_message.h
  DESTINATION include)

# 单测：只依赖 Eigen 的纯数学模块直接编进测试，不链接 msf_core（避免拉进厂商 SDK）。
# 找不到 ament_cmake_gtest 时跳过，不影响正常构建。
if(BUILD_TESTING)
  find_package(ament_cmake_gtest QUIET)
  if(ament_cmake_gtest_FOUND)
    ament_add_gtest(test_upright_p2_solver
      test/test_upright_p2_solver.cpp
      src/vision/upright_p2_solver.cpp)
    target_include_directories(test_upright_p2_solver PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_upright_p2_solver Eigen3::Eigen)
  endif()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 OpenCV yaml-cpp Ceres)
ament_package()
