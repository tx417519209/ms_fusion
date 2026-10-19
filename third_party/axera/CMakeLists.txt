cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ax_test)

set(CMAKE_CXX_STANDARD 17)

# 防止在源码目录编译
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "FATAL: Building inplace is not allowed. Please build in a separate directory.")
endif()

# 设置安装目录
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# 构建类型（默认Release）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# 目标平台配置（可按需修改）
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 编译器（默认使用宿主机gcc/g++，可改为交叉编译工具链）
#set(CMAKE_C_COMPILER aarch64-none-linux-gnu-gcc)
#set(CMAKE_CXX_COMPILER aarch64-none-linux-gnu-g++)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

# 编译优化参数
if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0 -Wall -fPIC")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -Wall -fPIC -fdata-sections -ffunction-sections")
endif()

# 包含头文件目录（按需添加）
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ROS2 依赖（仅 superpoint_msg_app 目标需要，找不到时跳过该目标）
find_package(rclcpp QUIET)
find_package(sensor_msgs QUIET)
find_package(std_msgs QUIET)
find_package(rcl_interfaces QUIET)

# 源文件
file(GLOB SOURCES_SUPERPOINT_MSG "example/test_infer_superpoint_from_msg.cpp")
file(GLOB SOURCES_SUPERPOINT_IVPS "example/test_infer_superpoint_from_ivps.cpp")
add_executable(example example/example.cpp)
if(rclcpp_FOUND AND sensor_msgs_FOUND AND std_msgs_FOUND)
    add_executable(superpoint_msg_app ${SOURCES_SUPERPOINT_MSG})
    add_executable(superpoint_ivps_app ${SOURCES_SUPERPOINT_IVPS})
    message(STATUS "superpoint_msg_app: ROS2 found, target enabled")
    message(STATUS "superpoint_ivps_app: ROS2 found, target enabled")
else()
    message(WARNING "superpoint_msg_app: rclcpp/sensor_msgs/std_msgs not found, target skipped")
    message(WARNING "superpoint_ivps_app: rclcpp/sensor_msgs/std_msgs not found, target skipped")
endif()
# 链接库目录（AX SDK所在路径）
link_directories(
    /soc/lib
)

target_link_libraries(example
    /soc/lib/libax_sys.so
    /soc/lib/libax_ivps.so
    ${OpenCV_LIBS}
    pthread
    dl
    m
)

target_link_libraries(superpoint_msg_app
    /soc/lib/libax_sys.so
    /soc/lib/libax_engine.so
    /soc/lib/libax_interpreter.so
    /soc/lib/libax_ivps.so
    ${OpenCV_LIBS}
    pthread
    dl
    m
)

target_link_libraries(superpoint_ivps_app
    /soc/lib/libax_sys.so
    /soc/lib/libax_engine.so
    /soc/lib/libax_interpreter.so
    /soc/lib/libax_ivps.so
    ${OpenCV_LIBS}
    pthread
    dl
    m
)

set_target_properties(example PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib"
)

set_target_properties(superpoint_msg_app PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib"
)

set_target_properties(superpoint_ivps_app PROPERTIES
    BUILD_RPATH "/soc/lib"
    INSTALL_RPATH "/soc/lib"
)

if(rclcpp_FOUND AND sensor_msgs_FOUND AND std_msgs_FOUND)
    target_include_directories(superpoint_msg_app PRIVATE
        ${rclcpp_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${std_msgs_INCLUDE_DIRS}
        ${rcl_interfaces_INCLUDE_DIRS}
    )
    target_include_directories(superpoint_ivps_app PRIVATE
        ${rclcpp_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${std_msgs_INCLUDE_DIRS}
        ${rcl_interfaces_INCLUDE_DIRS}
    )
    # 添加 ROS2 包含目录到所有相关目标
    target_link_libraries(superpoint_msg_app
        /soc/lib/libax_sys.so
        /soc/lib/libax_engine.so
        /soc/lib/libax_interpreter.so
        /soc/lib/libax_ivps.so
        ${OpenCV_LIBS}
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${std_msgs_LIBRARIES}
        pthread
        dl
        m
    )
    target_link_libraries(superpoint_ivps_app
        /soc/lib/libax_sys.so
        /soc/lib/libax_engine.so
        /soc/lib/libax_interpreter.so
        /soc/lib/libax_ivps.so
        ${OpenCV_LIBS}
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${std_msgs_LIBRARIES}
        pthread
        dl
        m
    )
    set_target_properties(superpoint_msg_app PROPERTIES
        BUILD_RPATH "/soc/lib"
        INSTALL_RPATH "/soc/lib"
    )
    set_target_properties(superpoint_ivps_app PROPERTIES
        BUILD_RPATH "/soc/lib"
        INSTALL_RPATH "/soc/lib"
    )
endif()

message(STATUS "AXERA SDK libraries linked from: /soc/lib")
