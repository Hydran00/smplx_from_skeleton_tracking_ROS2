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


class VirtualFixtureCalculator(Node):
    def __init__(self):
        super().__init__('virtual_fixture')
        self.target_frame = "haptic_interface_target"
        self.base_frame = "base_link"

        self.load_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj"
        self.skin_save_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/skin_mesh.obj"
        
        mesh = o3d.io.read_triangle_mesh(self.load_path)

        self.output_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/projected_skel.obj"
        rib_cage = utils.compute_torax_projection(mesh)
        rib_cage = rib_cage.to_legacy()
        
        
        EXTRUDE = False
        if False:
            print("Extruding rib cage")
            rib_cage_new = o3d.t.geometry.TriangleMesh.from_legacy(rib_cage)
            rib_cage_new = rib_cage_new.extrude_linear([0,0,-0.05])
            o3d.visualization.draw_geometries([rib_cage_new.to_legacy()])
            rib_cage = rib_cage_new.to_legacy()

        self.transform_mesh(rib_cage)
        self.transform_mesh(mesh)
        o3d.io.write_triangle_mesh(self.output_path, rib_cage, write_ascii=True)
        o3d.io.write_triangle_mesh(self.skin_save_path, mesh, write_ascii=True)
        self.get_logger().info("Virtual fixture saved to: "+self.output_path)
        rclpy.shutdown()


    def transform_mesh(self, mesh):
        R = mesh.get_rotation_matrix_from_xyz((np.pi,0,0))
        mesh.rotate(R, center=(0,0,0))
        mesh.translate((-0.12,0.3,1.6),relative=True)

        
def main(args=None):
    rclpy.init(args=args)
    node = VirtualFixtureCalculator()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()