cmake_minimum_required(VERSION 3.8)
project(body_msgs)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keypoint.msg"
  "msg/BodyData.msg"
  DEPENDENCIES std_msgs geometry_msgs# Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

ament_package()