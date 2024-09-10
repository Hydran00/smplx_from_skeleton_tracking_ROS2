from launch import LaunchDescription
from launch_ros.actions import Node
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
            "model_path": expanduser("~")+ '/SKEL_WS/SKEL/models/smpl/SMPL_MALE.pkl',
            "optimize_model": True,
        }
        ]
    )
    
    ld.add_action(parser) # uncomment if you are not using bag file
    ld.add_action(tracking_node)
    return ld