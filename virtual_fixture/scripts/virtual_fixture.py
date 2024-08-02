#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class VirtualFixture(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.get_logger().info('Virtual Fixture node has been initialized')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        self.timer = self.create_timer(0.1, self.compute_virtual_fixture)
        
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
        pass
    

def main(args=None):
    rclpy.init(args=args)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()