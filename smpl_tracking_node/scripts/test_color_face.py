#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import os
import time
from body38_to_smpl import LUNG_US_SMPL_FACES
import copy
# Define materials
mat_sphere_transparent = o3d.visualization.rendering.MaterialRecord()
mat_sphere_transparent.shader = 'defaultLitTransparency'
mat_sphere_transparent.base_color = [0.467, 0.467, 0.467, 0.6]
mat_sphere_transparent.base_roughness = 0.0
mat_sphere_transparent.base_reflectance = 0.0
mat_sphere_transparent.base_clearcoat = 1.0
mat_sphere_transparent.thickness = 1.0
mat_sphere_transparent.transmission = 1.0
mat_sphere_transparent.absorption_distance = 10
mat_sphere_transparent.absorption_color = [0.5, 0.5, 0.5]

mat_skin = o3d.visualization.rendering.MaterialRecord()
mat_skin.shader = 'defaultLitTransparency'
mat_skin.base_color = [0.467, 0.467, 0.467, 0.6]
mat_skin.base_roughness = 0.0
mat_skin.base_reflectance = 0.0
mat_skin.base_clearcoat = 1.0
mat_skin.thickness = 1.0
mat_skin.transmission = 1.0
mat_skin.absorption_distance = 10
mat_skin.absorption_color = [0.5, 0.5, 0.8]

    


def to_z_up(mesh):
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    mesh.rotate(R, center=[0, 0, 0])
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    # mesh.rotate(R, center=[0, 0, 0])

def main():
    geometries=[]
    face_centers = []
    # Load meshes
    mesh_path = os.path.expanduser('~') + '/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj'
    skelethon_path = os.path.expanduser('~') + '/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj'
    cloud_path = os.path.expanduser('~') + '/ros2_ws/point_cloud.ply'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    skelethon = o3d.io.read_triangle_mesh(skelethon_path)
    cloud = o3d.io.read_point_cloud(cloud_path)

    # Check if meshes are loaded correctly
    if not mesh.has_vertices():
        print(f"Failed to load mesh from {mesh_path}")
    if not skelethon.has_vertices():
        print(f"Failed to load skelethon from {skelethon_path}")

    # Compute the vertex normals
    mesh.compute_vertex_normals()
    skelethon.compute_vertex_normals()

    # Assign colors to vertices
    vertex_colors = np.ones((len(mesh.vertices), 3))
    for face in LUNG_US_SMPL_FACES.values():
        print(f"Vertices to color: {mesh.triangles[face]} with face index {face}")
        vertex_colors[mesh.triangles[face]] = [1, 0, 0]

    # Create spheres at face centers
    sphere_list = []
    for i, face in enumerate(LUNG_US_SMPL_FACES.values()):
        # Get the coordinates of the vertices of the face
        vertices_ids = mesh.triangles[face]
        vertex_coords = np.asarray(mesh.vertices)[vertices_ids]
        # Compute the center of the face
        face_center = vertex_coords.mean(axis=0)
        face_centers.append(face_center)
        # Create and color spheres
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        # Normals are need for the shader to work
        sphere.compute_vertex_normals()
        sphere.translate(face_center)
        sphere.paint_uniform_color([1, 0, 0])
        sphere_list.append(sphere)
        # convert to Z-up
        to_z_up(sphere)
        # Add spheres to visualizer
        geometries.append({"name": f"sphere{i}", "geometry": sphere, "material": mat_sphere_transparent})




    # Assign vertex colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    ref_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])


    # coordinate frame now is right handed Y-up but we right handed need Z-up for ROS
    # so we rotate the mesh and skeleton
    to_z_up(mesh)
    to_z_up(skelethon)
    to_z_up(ref_frame)
    to_z_up(cloud)
    to_z_up(ref_frame)

    # Add the mesh and skeleton to the visualizer
    geometries.append({"name": "ref_frame", "geometry": ref_frame})
    geometries.append({"name": "mesh", "geometry": mesh, "material": mat_skin})
    geometries.append({"name": "skelethon", "geometry": skelethon})
    geometries.append({"name": "cloud", "geometry": cloud})


    # Show the visualizer
    o3d.visualization.draw(geometries)

if __name__=="__main__":
    main()