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
import pyassimp
# from visualize_model import Visualizer
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
import time 
import numpy as np
import scipy 
class VirtualFixture(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.timer = self.create_timer(0.1, self.publish_mesh)
        # self.subscribtion = self.create_subscription(PoseStamped, 'target_frame', self.compute_virtual_fixture, 1)
        self.mesh_publisher = self.create_publisher(Marker, 'visualization_marker', 1) 
        # self.visualizer = Visualizer()
        # self.vis_update_timer = self.create_timer(0.03, self.update_viz)
        # load initial scene
        # self.visualizer.add_target()
        self.get_logger().info('Virtual Fixture node has been initialized')
        self.publish_mesh()
    def update_viz(self):
        self.visualizer.update()   
         
    def publish_mesh(self):
        # Load or create your Open3D mesh with color
        mesh = o3d.geometry.TriangleMesh.create_sphere()
        mesh.compute_vertex_normals()
        num_vertices = len(np.asarray(mesh.vertices))
        # paint vertices red
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.random.rand(num_vertices, 3))
        # Convert Open3D mesh to Trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors)

        # Create a Trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

        # Export the Trimesh object to a .dae file
        path = os.path.expanduser('~')+"/ros_ws/output_mesh.dae"
        # Export the mesh to a .dae file preserving colors
        # trimesh_mesh.export(file_obj=path, file_type='dae', include_color=True)
        output = trimesh.exchange.dae.export_collada(trimesh_mesh)
        
        # write output
        with open(path, 'wb') as f:
            f.write(output)
        
        self.get_logger().info('Exporting file: '+str(path))
        marker = Marker()
        marker.id = 0
        # marker.mesh_resource = os.path.expanduser('~')+'/ros_ws/output_mesh.dae'
        marker.mesh_resource = "file://"+path
        marker.mesh_use_embedded_materials = True  # Need this to use textures for mesh
        marker.type = marker.MESH_RESOURCE
        marker.header.frame_id = "base_link"
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.pose.orientation.w = 1.0
        self.mesh_publisher.publish(marker)
        exit(0)
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