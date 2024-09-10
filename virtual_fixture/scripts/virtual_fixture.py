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
import pickle
from geometry_msgs.msg import TransformStamped
import scipy
class VirtualFixtureCalculator(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"
        self.skin_load_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj"
        self.skin_save_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/skin_mesh.obj"
        self.transform_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/transform.pkl"
        with open(self.transform_path, 'rb') as f:
            self.transform = pickle.load(f)
        self.get_logger().info("Transform loaded:"+str(self.transform))

        self.skin = o3d.io.read_triangle_mesh(self.skin_load_path)
        self.output_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/final_vf.obj"
        
    
    def computeVF(self):
        reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
        rib_cage = utils.compute_torax_projection(self.skin)

        o3d.visualization.draw_geometries([rib_cage, self.skin, reference_frame])
        rib_cage = self.transform_to_RH_Z_UP(rib_cage)
        self.skin = self.transform_to_RH_Z_UP(self.skin)
        o3d.visualization.draw_geometries([rib_cage, self.skin, reference_frame])
        
        # rib_cage = rib_cage.to_legacy()
        
        
        # EXTRUDE = False
        # if EXTRUDE:
        #     rib_cage_new = o3d.t.geometry.TriangleMesh.from_legacy(rib_cage)
        #     rib_cage_new = rib_cage_new.extrude_linear([0,0,-0.05])
        #     o3d.visualization.draw_geometries([rib_cage_new.to_legacy()])
        #     rib_cage = rib_cage_new.to_legacy()


        rib_cage.transform(self.transform)
        self.skin.transform(self.transform)
        o3d.visualization.draw_geometries([rib_cage, self.skin, reference_frame])

        # self.transform_mesh()
        self.skin.compute_vertex_normals()
        rib_cage.compute_vertex_normals()
        o3d.io.write_triangle_mesh(self.output_path, rib_cage, write_ascii=True)
        o3d.io.write_triangle_mesh(self.skin_save_path, self.skin, write_ascii=True)
        self.get_logger().info("Virtual fixture saved to: "+self.output_path)

    def transform_to_RH_Z_UP(self, mesh):
        # Transform the mesh to the right hand coordinate system
        # Z up
        # X forward
        
        # z = y
        # y = -x
        # x = z

        vertices = np.asarray(mesh.vertices)
        for vertex in vertices:
            x = vertex[0]
            y = vertex[1]
            z = vertex[2]
            vertex[0] = z
            vertex[1] = -x
            vertex[2] = y
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        return mesh




        
    

        
def main(args=None):
    rclpy.init(args=args)
    node = VirtualFixtureCalculator()
    node.computeVF()
    
    
if __name__ == '__main__':
    main()