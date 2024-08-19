import open3d as o3d
import os
import numpy as np
import time
import cvxpy as cp
import time
import pygame

def plot_plane_constraint(n,p, plane_list):
    size = 0.1
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

def solve_QP(x_des,x,n,p,radius):
    print("Target: ", x_des, " | Current: ", x, " | n: ", n, " | p: ", p, " | x: ", x, " | radius: ", radius)
    delta_x = cp.Variable(3)
    objective = cp.Minimize(cp.norm(delta_x - (x_des - x)))
    constraints = []
    constraints.append(n.T @ delta_x >= -n.T @ (x - p) + radius)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status != cp.OPTIMAL:
        print(f"Optimization problem not solved optimally: {problem.status}")
        return x_des - x
    return delta_x.value
    

def _run_main_loop():
    pygame.init()
    pygame.display.set_mode((50, 50))        
    viz = o3d.visualization.Visualizer()
    viz.create_window(window_name="Virtual Fixture Test", width=2200, height=1200)
    
    
    radius = 0.005
    current = np.array([0.0,0,0.1])
    target = np.array([0.0,0,0.0])
    
    plane = [[0,0,-1], [0,0,0.05]]
    
    n = -np.array(plane[0]).reshape(3,)
    p = np.array(plane[1]).reshape(3,)
    x = np.array([current]).reshape(3,)
    
    
    plane_list = []
    plot_plane_constraint(n,p,plane_list)
    for plane in plane_list:
        viz.add_geometry(plane)

    sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_target.translate(target)
    sphere_target.paint_uniform_color([1, 0, 0])  # Red color for the target
    viz.add_geometry(sphere_target)
    
    sphere_current = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere_current.translate(current)
    sphere_current.paint_uniform_color([0, 1, 0])  # Green color for the current
    viz.add_geometry(sphere_current)
    
    """
    Run the main loop to update visualization and enforce virtual fixture constraints.
    """
    

    
    clock = pygame.time.Clock()
    direction_vector = np.array([0.0, 0.0, 0.0])
    speed = 0.0005
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    direction_vector[2] -= 1.0
                if event.key == pygame.K_s:
                    direction_vector[2] += 1.0
                if event.key == pygame.K_a:
                    direction_vector[0] += 1.0
                if event.key == pygame.K_d:
                    direction_vector[0] -= 1.0
                if event.key == pygame.K_q:
                    direction_vector[1] -= 1.0
                if event.key == pygame.K_e:
                    direction_vector[1] += 1.0

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    direction_vector[2] += 1.0
                if event.key == pygame.K_s:
                    direction_vector[2] -= 1.0
                if event.key == pygame.K_a:
                    direction_vector[0] -= 1.0
                if event.key == pygame.K_d:
                    direction_vector[0] += 1.0
                if event.key == pygame.K_q:
                    direction_vector[1] += 1.0
                if event.key == pygame.K_e:
                    direction_vector[1] -= 1.0

        print(f"Direction vector: {direction_vector}")
        # Update target sphere position
        
        print("Current position: ", current," ~ Target position: ", target)
        target = target + direction_vector * speed
        
        sphere_target.translate(target, relative=False)
        viz.update_geometry(sphere_target)
        
        current = current + solve_QP(target,current,n,p,radius)
        # Enforce virtual fixture on the target position

        sphere_current.translate(current, relative=False)
        viz.update_geometry(sphere_current)
        # Poll for new events and update the renderer
        viz.poll_events()
        viz.update_renderer()

        # Update old positions for the next iteration

        time.sleep(0.03)
    
if __name__ == "__main__":
    _run_main_loop()