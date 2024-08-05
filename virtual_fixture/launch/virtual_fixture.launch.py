from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_context import LaunchContext
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory
from os.path import expanduser

import os


def generate_launch_description():
    ld = LaunchDescription()
    print(get_package_share_directory("virtual_fixture") + "/rviz/rviz_conf.rviz")
    vf_node = Node(
        package="virtual_fixture",
        executable="virtual_fixture.py",
    )
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", get_package_share_directory("virtual_fixture") + "/rviz/rviz_config.rviz"],
    )
    
    ld.add_action(rviz2)
    ld.add_action(vf_node)
    return ld