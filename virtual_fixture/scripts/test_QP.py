import open3d as o3d
import os
import numpy as np
import time
import cvxpy as cp
import time

def plot_plane_constraints(n,p, plane_list):
    size = 0.03
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.0001)
    plane_mesh.compute_vertex_normals()
    # paint the plane
    plane_mesh.paint_uniform_color([1, 1, 0])
    plane_mesh.translate(-np.array([size/2, size/2, 0.0001/2]))
    # Compute the rotation matrix to align the box normal with the plane normal
    z_axis = np.array([0, 0, 1])
    n = n / np.linalg.norm(n)  # Ensure the normal is normalized

    # Compute the rotation axis (cross product between z-axis and normal)
    rotation_axis = np.cross(z_axis, n)
    rotation_angle = np.arccos(np.dot(z_axis, n))  # Angle between z-axis and normal

    if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid rotation if the plane is already aligned
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # Convert axis-angle to rotation matrix
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        plane_mesh.rotate(R)

    # Translate the box to the correct position
    plane_mesh.translate(p)

    plane_list.append(plane_mesh)

def run_tests():
    viz = o3d.visualization.Visualizer()
    viz.create_window(window_name="Virtual Fixture Test", width=2200, height=1200)
    opt = viz.get_render_option()
    
    radius = 0.005
    current = np.array([0.0,0,0.1])
    target = np.array([0.06,0,0])
    
    plane = [[0,0,1], [0,0,0.05]]
    
    n = np.array(plane[0]).reshape(3,)
    p = np.array(plane[1]).reshape(3,)
    x = np.array([current]).reshape(3,)
    
    # QP vars
    delta_x = cp.Variable(3)
    plane_list = []
    plot_plane_constraints(n,p,plane_list)
    for plane in plane_list:
        viz.add_geometry(plane)
    constraints = []
    constraints.append(n.T @ delta_x >= -n.T @ (x - p) +radius)
    
    sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_target.translate(target)
    sphere_target.paint_uniform_color([1, 0, 0])  # Green color for the sphere
    viz.add_geometry(sphere_target)
    
    # QP
    objective = cp.Minimize(cp.norm(delta_x - (target - current)))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    target_qp = current + delta_x.value
    
    sphere_target_qp = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_target_qp.translate(target_qp)
    sphere_target_qp.paint_uniform_color([0, 1, 0])  # Green color for the sphere
    viz.add_geometry(sphere_target_qp)
    
    while(True):
        viz.poll_events()
        viz.update_renderer()
        time.sleep(0.03)
    
    
if __name__ == "__main__":
    run_tests()