cmake_minimum_required(VERSION 3.8)
project(tennis_court_map)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)

include_directories(include)

add_executable(tennis_court_map_node
  src/tennis_court_map_node.cpp
  src/tennis_court_map_builder.cpp)
ament_target_dependencies(tennis_court_map_node rclcpp nav_msgs)

install(TARGETS tennis_court_map_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
