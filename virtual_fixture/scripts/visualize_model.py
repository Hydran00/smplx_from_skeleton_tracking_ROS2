#!/usr/bin/env python3
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import numpy as np
import os
import time
from utils import LUNG_US_SMPL_FACES, to_z_up
import utils
import copy
import scipy
from materials import mat_sphere_transparent, mat_skin
import threading
from tqdm import tqdm


def visualize_normals(triangles,normals, length):
    lines = []
    colors = []
    new_vertices = []
    for i, triangle in enumerate(triangles):
        v0, v1, v2 = triangle   

        # Compute the center of the triangle (face)
        triangle_center = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        normal_start = triangle_center
        normal_end = triangle_center + normals[i] * length

        new_vertices.append(normal_start)
        new_vertices.append(normal_end)

        lines.append([2*i, 2*i + 1])
        colors.extend([[1, 0, 0]] * 2)  # Red color for normals
    new_vertices = np.array(new_vertices)
    lines = np.array(lines)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(new_vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

if __name__=="__main__":
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(os.path.expanduser('~') + '/SKEL_WS/ros2_ws/projected_skel.obj')
    # dataset = o3d.data.KnotMesh()
    # mesh = o3d.io.read_triangle_mesh(dataset.path)
    # mesh.scale(0.002, center=mesh.get_center())
    
    # sphere
    # mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    # mesh.compute_triangle_normals()
    # mesh.orient_triangles()
    # mesh.normalize_normals()

    # Create a LineSet for visualizing normals
    lines = []
    colors = []
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.triangle_normals)  # Use triangle normals

    line_set = visualize_normals(triangles, normals, 0.01)

    # Visualize the mesh and the normals
    o3d.visualization.draw_geometries([mesh, line_set])

    # try orienting the normals according to the radial direction
    center = mesh.get_center()
    for i, triangle in enumerate(triangles):
        v0, v1, v2 = triangle
        # Compute the center of the triangle (face)
        triangle_center = (vertices[v0] + vertices[v1] + vertices[v2]) / 3.0
        direction = (triangle_center - center) / np.linalg.norm(triangle_center - center)
        if np.dot(normals[i], direction) < 0:
            normals[i] = -normals[i]

    line_set = visualize_normals(triangles, normals, 0.01)
    o3d.visualization.draw_geometries([mesh, line_set])
    
