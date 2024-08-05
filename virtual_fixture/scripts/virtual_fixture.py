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
import os
import trimesh
from ctypes import *
from visualize_model import Visualizer
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
import time 
import numpy as np
import scipy 
import utils

class VirtualFixture(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.timer = self.create_timer(0.1, self.publish_mesh)
        self.subscribtion = self.create_subscription(PoseStamped, 'target_frame', self.apply_virtual_fixture, 1)
        self.mesh_publisher = self.create_publisher(Marker, 'visualization_marker', 1) 
        self.load_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj"
        self.skin_save_path = os.path.expanduser('~')+"/ros2_ws/output_mesh.obj"
        self.areas_save_path_prefix = os.path.expanduser('~')+"/ros2_ws/areas"
        self.visualizer = Visualizer()
        self.vis_update_timer = self.create_timer(0.03, self.update_viz)
        # load initial scene
        # self.visualizer.add_target()
        self.already_scanned_areas = {x:False for x in range(1,14)}
        self.get_logger().info('Virtual Fixture node has been initialized, already scanned areas: '+str(self.already_scanned_areas)) 
        self.compute_VF()
        self.current_area_to_scan = 11
    
    def update_viz(self):
        self.visualizer.update()   
        
    def transform_mesh(self, mesh):
        # self.get_logger().info("Mesh before: "+str(mesh.get_center()))
        R = mesh.get_rotation_matrix_from_xyz((np.pi,0,0))
        mesh.rotate(R,center=mesh.get_center())
        mesh.translate((0,0.3,0),relative=False)
        # self.get_logger().info("Mesh after: "+str(mesh.get_center()))

        
        
    def compute_VF(self): 
        utils.clear_meshes(self.mesh_publisher)
        mesh =o3d.io.read_triangle_mesh(self.load_path)
        self.transform_mesh(mesh)
        o3d.io.write_triangle_mesh(self.skin_save_path, mesh)
        self.mesh = mesh
        self.get_logger().info("Mesh has been saved, sending it to Rviz")
        utils.publish_mesh(self.mesh_publisher, self.skin_save_path, 0, rgba=[0.5,0.5,0.5,0.5])
        
        self.areas_center = utils.get_protocol_areas_center(mesh)
        self.get_logger().info("Areas center: "+str(self.areas_center))
        self.spheres_dict = utils.create_spherical_areas(self.areas_center)
        for i,sphere in enumerate(self.spheres_dict.values()):
            idx = list(self.spheres_dict.keys())[i]
            areas_save_path = self.areas_save_path_prefix+str(idx)+".obj"
            o3d.io.write_triangle_mesh(areas_save_path, sphere)
            self.visualizer.add_geometry(sphere)
            self.get_logger().info(f"Sphere {idx} has been saved in {areas_save_path}, sending it to Rviz")
            utils.publish_mesh(self.mesh_publisher, areas_save_path, idx, rgba=[1.0,0.0,0.0,0.5])
        exit()
        
        
    def apply_virtual_fixture(self, msg):
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
        orientation = (msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z)
        position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        
        # orientation = (msg.pose.orientation.w,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z)
        # orientation = (1,0,0,0)
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