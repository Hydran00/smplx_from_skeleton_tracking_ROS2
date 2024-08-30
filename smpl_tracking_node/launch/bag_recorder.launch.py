from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
# include launch file
from launch.actions import IncludeLaunchDescription
from os.path import expanduser

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
            "optimize_model": False,
        }
        ]
    )
    # include launch file
    body_tracking = IncludeLaunchDescription(
        launch_description_source = get_package_share_directory("yolo_seg") + "/launch/body_publisher.launch.py"
    )
    
    ld.add_action(parser)
    # this node is used just for visualization
    ld.add_action(tracking_node)
    ld.add_action(body_tracking)
    return ld