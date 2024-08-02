#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import open3d as o3d
from visualize_model import Visualizer
from geometry_msgs.msg import PoseStamped
import scipy 
class VirtualFixture(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.timer = self.create_timer(0.1, self.compute_virtual_fixture)
        self.subscribtion = self.create_subscription(PoseStamped, 'target_frame', self.compute_virtual_fixture, 1)
        self.visualizer = Visualizer()
        self.vis_update_timer = self.create_timer(0.03, self.update_viz)
        # load initial scene
        # self.visualizer.add_target()
        self.get_logger().info('Virtual Fixture node has been initialized')
        
    def update_viz(self):
        self.visualizer.update()   
         
    def compute_virtual_fixture(self, msg):
        # retrieve robot ee position
        # try:
        #     t = self.tf_buffer.lookup_transform(
        #         self.base_frame, self.target_frame,
        #         rclpy.time.Time())
        # except TransformException as ex:
        #     self.get_logger().info(
        #         f'Could not transform from {self.base_frame} to {self.target_frame}: {ex}')
        #     return
        
        
        # self.get_logger().info(f'Computed virtual fixture at {t.transform.translation}')
        # self.visualizer.update_geometry("target_ref_frame", self.target_ref_frame_mesh.transform(t.transform))

        # position = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
        # orientation = (t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z)
        position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        
        # orientation = (msg.pose.orientation.w,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z)
        orientation = (1,0,0,0)
        # print RPY
        self.get_logger().info(f'RPY: {scipy.spatial.transform.Rotation.from_quat(orientation).as_euler("xyz", degrees=True)}')
        self.visualizer.update_target_pose(position, orientation)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualFixture()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()