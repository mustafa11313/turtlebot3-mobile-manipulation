cmake_minimum_required(VERSION 3.8)
project(fetch_deliver_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(fetch_deliver_interfaces REQUIRED)

add_library(honest_link_attacher SHARED src/honest_link_attacher.cpp)
target_include_directories(honest_link_attacher PUBLIC ${GAZEBO_INCLUDE_DIRS})
ament_target_dependencies(honest_link_attacher gazebo_dev gazebo_ros rclcpp fetch_deliver_interfaces)
install(TARGETS honest_link_attacher LIBRARY DESTINATION lib)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/scenario_spawner.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY worlds models launch config DESTINATION share/${PROJECT_NAME})

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")
ament_package()
