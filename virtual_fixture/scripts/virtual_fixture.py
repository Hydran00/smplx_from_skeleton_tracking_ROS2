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

class VirtualFixture(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.get_logger().info('Virtual Fixture node has been initialized')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        self.timer = self.create_timer(0.01, self.compute_virtual_fixture)
        self.visualizer = Visualizer()
        # load initial scene
        self.visualizer.draw()        
        self.visualizer.add_target()
        
        
    def compute_virtual_fixture(self):
        # retrieve robot ee position
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame, self.target_frame,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform from {self.base_frame} to {self.target_frame}: {ex}')
            return
        self.get_logger().info(f'Computed virtual fixture at {t.transform.translation}')
        self.visualizer.update_geometry("target_ref_frame", self.target_ref_frame_mesh.transform(t.transform))

        position = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
        orientation = (t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z)
        self.visualizer.update_target_pose(position, orientation)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualFixture()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()