#!/usr/bin/env python3
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import numpy as np
import os
import time
from utils import LUNG_US_SMPL_FACES
import copy
import scipy
from materials import mat_sphere_transparent, mat_skin
import threading


def to_z_up(mesh):
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    mesh.rotate(R, center=[0, 0, 0])
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    # mesh.rotate(R, center=[0, 0, 0])


class Visualizer:

    def __init__(self):
        self.gui = gui
        self.gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window('img', width=1920, height=1080)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget)  
        self.geometries=[]
        self.sphere_list = []
        self.face_centers = []
        
        # Load meshes
        mesh_path = os.path.expanduser('~') + '/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj'
        skelethon_path = os.path.expanduser('~') + '/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj'
        cloud_path = os.path.expanduser('~') + '/ros2_ws/point_cloud.ply'
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.skelethon = o3d.io.read_triangle_mesh(skelethon_path)
        self.cloud = o3d.io.read_point_cloud(cloud_path)
        self.mat_sphere_transparent = mat_sphere_transparent
        self.mat_skin = mat_skin

        # Check if meshes are loaded correctly
        if not self.mesh.has_vertices():
            print(f"Failed to load mesh from {mesh_path}")
        if not self.skelethon.has_vertices():
            print(f"Failed to load skelethon from {skelethon_path}")

        # Compute the vertex normals
        self.mesh.compute_vertex_normals()
        self.skelethon.compute_vertex_normals()

        # Assign colors to vertices
        vertex_colors = np.ones((len(self.mesh.vertices), 3))
        for face in LUNG_US_SMPL_FACES.values():
            print(f"Vertices to color: {self.mesh.triangles[face]} with face index {face}")
            vertex_colors[self.mesh.triangles[face]] = [1, 0, 0]

        # Create spheres at face centers
        for i, face in enumerate(LUNG_US_SMPL_FACES.values()):
            # Get the coordinates of the vertices of the face
            vertices_ids = self.mesh.triangles[face]
            vertex_coords = np.asarray(self.mesh.vertices)[vertices_ids]
            # Compute the center of the face
            face_center = vertex_coords.mean(axis=0)
            self.face_centers.append(face_center)
            # Create and color spheres
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            # Normals are need for the shader to work
            sphere.compute_vertex_normals()
            sphere.translate(face_center)
            sphere.paint_uniform_color([1, 0, 0])
            self.sphere_list.append(sphere)
            # convert to Z-up
            to_z_up(sphere)
            # Add spheres to visualizer
            self.geometries.append({"name": f"sphere{i}", "geometry": sphere, "material": self.mat_sphere_transparent})

        # Assign vertex colors to the mesh
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        self.ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        self.widget.scene.camera.look_at([0,0,0], [0,0,-1], [0,1,0])
        

        # coordinate frame now is right handed Y-up but we right handed need Z-up for ROS
        # so we rotate the mesh and skeleton
        to_z_up(self.mesh)
        to_z_up(self.skelethon)
        to_z_up(self.ref_frame)
        to_z_up(self.cloud)
        to_z_up(self.ref_frame)

        # Add the mesh and skeleton to the visualizer
        self.geometries.append({"name": "ref_frame", "geometry": self.ref_frame})
        self.geometries.append({"name": "mesh", "geometry": self.mesh, "material": self.mat_skin})
        self.geometries.append({"name": "skelethon", "geometry": self.skelethon})
        self.geometries.append({"name": "cloud", "geometry": self.cloud})

       
    
    # def add_target(self):
    #     self.target_ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1) 
    #     self.widget.scene.add_geometry("target_ref_frame", self.target_ref_frame)
    
    # def update_target_pose(self, xyz, quat):
    #     self.widget.scene.remove_geometry("target_ref_frame")
    #     self.target_ref_frame.translate(xyz, relative=False)
    #     # convert quat to rotation matrix
    #     R = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
    #     self.target_ref_frame.rotate(R, center=[0, 0, 0])
    #     self.widget.scene.add_geometry("target_ref_frame", self.target_ref_frame)
        
    def draw(self):
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit'
        # TODO use correct material
        for g in self.geometries:
            self.widget.scene.add_geometry(g["name"], g["geometry"], mat)
        
        # send the gui to a separate thread
        # self.gui_thread = threading.Thread(target=self.window.run)
        # TODO fix camera orientation
        self.gui.Application.instance.run()
        
if __name__=="__main__":
    vis = Visualizer()
    vis.draw()