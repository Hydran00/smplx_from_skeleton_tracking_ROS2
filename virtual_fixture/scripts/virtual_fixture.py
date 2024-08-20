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
from virtual_fixture_msgs.msg import Areas
from std_msgs.msg import Float32MultiArray
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
        self.timer = self.create_timer(1.0, self.publish_areas)
        # self.target_framesubscribtion = self.create_subscription(PoseStamped, 'target_frame', self.apply_virtual_fixture, 1)
        self.mesh_publisher = self.create_publisher(Marker, 'visualization_marker', 1)
        self.load_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj"
        self.skin_save_path = os.path.expanduser('~')+"/ros2_ws/output_mesh.obj"
        self.areas_save_path_prefix = os.path.expanduser('~')+"/ros2_ws/areas"
        self.visualizer = Visualizer()
        self.vis_update_timer = self.create_timer(0.03, self.update_viz)
        # self.areas_pub = self.create_publisher(Areas, 'areas', 1)
        self.areas = None
        self.radius = 0.05
        
        mesh = o3d.io.read_triangle_mesh(self.load_path)

        self.output_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/projected_skel.ply"
        rib_cage = utils.compute_torax_projection(mesh)
        rib_cage = rib_cage.to_legacy()
        
        
        EXTRUDE = False
        if EXTRUDE:
            rib_cage_new = o3d.t.geometry.TriangleMesh.from_legacy(rib_cage)
            rib_cage_new = rib_cage_new.extrude_linear([0,0,-0.05])
            o3d.visualization.draw_geometries([rib_cage_new.to_legacy()])
            rib_cage = rib_cage_new.to_legacy()

        self.transform_mesh(rib_cage)
        o3d.io.write_triangle_mesh(self.output_path, rib_cage, write_ascii=True)
        self.get_logger().info("Clear rviz")
        utils.clear_meshes(self.mesh_publisher)
        time.sleep(1.0)
        self.get_logger().info("Sending rib cage to rviz")
        utils.publish_mesh(self.mesh_publisher, self.output_path, 0)
        # self.current_area_to_scan = 11
        # self.transitioning_between_areas = True
    
    def update_viz(self):
        self.visualizer.update()   
        
    def transform_mesh(self, mesh):
        R = mesh.get_rotation_matrix_from_xyz((np.pi,0,0))
        mesh.rotate(R,center=mesh.get_center())
        mesh.translate((0,0.3,0),relative=False)

    def publish_areas(self):
        if self.areas is not None:
            self.areas_pub.publish(self.areas)
    

        
def main(args=None):
    rclpy.init(args=args)
    node = VirtualFixture()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()