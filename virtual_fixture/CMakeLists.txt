cmake_minimum_required(VERSION 3.8)
project(virtual_fixture)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/visualize_model.py
  scripts/virtual_fixture.py
  scripts/virtual_fixture_test.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz skel_regions
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
