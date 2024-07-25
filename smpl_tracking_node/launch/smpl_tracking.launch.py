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

    parser = Node(
        package="smpl_tracking_node",
        executable="body_json_parser.py",
    )
    tracking_node = Node(
        package="smpl_tracking_node",
        executable="smpl_tracking.py",
        output="both",
        parameters=[{
            "mirror":True,
            "model_type":"smpl", # smpl or smplx
            "model_path": expanduser("~")+ "/models/smpl/smpl_male.pkl"
            # "model_path": expanduser("~")+ "/models/smplx/SMPLX_MALE.npz"
        }
        ]
    )
    
    # ld.add_action(parser)
    ld.add_action(tracking_node)
    return ld