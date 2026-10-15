cmake_minimum_required(VERSION 3.8)
project(fetch_deliver_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectObservation.msg"
  "msg/MissionStatus.msg"
  "msg/MissionEvent.msg"
  "srv/AttachObject.srv"
  "srv/DetachObject.srv"
  "action/Pick.action"
  "action/Place.action"
  "action/Align.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
