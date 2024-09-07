#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import rclpy
from rclpy.node import Node
import open3d as o3d
import os
from ctypes import *
import time 
import numpy as np
import utils
import tf2_ros
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import scipy
class VirtualFixtureCalculator(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        self.skin_load_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj"
        self.skin_save_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/skin_mesh.obj"
        self.camera_frame_name = "tool0"
        self.base_frame = "base_link"
        self.skin = o3d.io.read_triangle_mesh(self.skin_load_path)
        self.output_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/final_vf.obj"
        
    
    def computeVF(self):
        trans_matrix = self.lookup_transform()
        rib_cage = utils.compute_torax_projection(self.skin)

        # rib_cage = rib_cage.to_legacy()
        
        
        # EXTRUDE = False
        # if EXTRUDE:
        #     rib_cage_new = o3d.t.geometry.TriangleMesh.from_legacy(rib_cage)
        #     rib_cage_new = rib_cage_new.extrude_linear([0,0,-0.05])
        #     o3d.visualization.draw_geometries([rib_cage_new.to_legacy()])
        #     rib_cage = rib_cage_new.to_legacy()
        rib_cage.transform(trans_matrix)
        self.skin.transform(trans_matrix)
        # self.transform_mesh()
        o3d.io.write_triangle_mesh(self.output_path, rib_cage, write_ascii=True)
        o3d.io.write_triangle_mesh(self.skin_save_path, self.skin, write_ascii=True)
        self.get_logger().info("Virtual fixture saved to: "+self.output_path)

    def lookup_transform(self):
        try:
            # Wait for the transform between 'base_link' and 'odom' for up to 5 seconds
            while(not self.tf_buffer.can_transform(self.base_frame, self.camera_frame_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))):
                self.get_logger().info("Waiting for transform")
                rclpy.spin_once(self)
            self.get_logger().info("Transform found")
            # Lookup the transform after the waiting period
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"TransformException: {e}")
        
        trans = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame_name, rclpy.time.Time())
        # Access transform details
        self.get_logger().info(f'Translation: {trans.transform.translation}')
        self.get_logger().info(f'Rotation: {trans.transform.rotation}')
        
        trans_matrix = np.eye(4)
        trans_matrix[:3, 3] = [trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z]
        quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
        rot_matrix = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
        trans_matrix[:3, :3] = rot_matrix
        return trans_matrix
        
    

        
def main(args=None):
    rclpy.init(args=args)
    node = VirtualFixtureCalculator()
    node.computeVF()
    
    
if __name__ == '__main__':
    main()