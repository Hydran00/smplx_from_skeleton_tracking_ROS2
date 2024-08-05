from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from os.path import expanduser
from launch.actions import TimerAction
import os


def generate_launch_description():
    ld = LaunchDescription()
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", get_package_share_directory("virtual_fixture") + "/rviz/rviz_config.rviz"],
    )
    
    # launch with delay
    delayed_vf_node = TimerAction(
        period = 1.0,
        actions = [Node(
        package="virtual_fixture",
        executable="virtual_fixture.py",
        )]
    )
    ld.add_action(rviz2)
    ld.add_action(delayed_vf_node)

    return ld