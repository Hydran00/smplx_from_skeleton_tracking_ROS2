#!/usr/bin/env python3

import open3d as o3d
import os
import numpy as np
import pygame
import time
import rclpy
from rclpy.node import Node
import cvxpy as cp
from scipy.spatial import KDTree
import math_utils 
from closest_on_triangle import find_closest_point_on_triangle, Location
from concurrent.futures import ThreadPoolExecutor

# Set up environment variable for SDL
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Define constants for visualization and movement
POINT_RADIUS = 0.02
MOVEMENT_SPEED = 0.00005
VISUALIZE_PLANE_CONSTRAINTS = True
class VirtualFixtureDemo(Node):
    def __init__(self):
        super().__init__('virtual_fixture_demo')
        self.get_logger().info("Starting")

        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window(window_name="Virtual Fixture Demo", width=2200, height=1200)
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True  # Display coordinate frame
        opt.mesh_show_wireframe = True    # Show mesh wireframe
        # Define screen dimensions and initialize display using pygame
        screen_width, screen_height = 50, 50
        pygame.init()
        pygame.display.set_mode((screen_width, screen_height))        
        # Load and prepare the surface (bunny mesh)
        # dataset = o3d.data.BunnyMesh()
        # self.surface = o3d.io.read_triangle_mesh(dataset.path)
        # self.surface = o3d.io.read_triangle_mesh(os.path.expanduser('~') + '/SKEL_WS/ros2_ws/bunny.ply')
        
        self.surface = o3d.io.read_triangle_mesh(os.path.expanduser('~') + '/SKEL_WS/ros2_ws/Skull.stl')
        self.surface.remove_duplicated_vertices()
        self.surface.scale(1/1000, center=(0,0,0))


        self.surface.compute_vertex_normals()  # Compute normals for the mesh
        self.viz.add_geometry(self.surface)
        
        self.surface.translate((0, 0.5, 0), relative=False)
        self.viz.add_geometry(self.surface)
        
        
        self.vertices = np.asarray(self.surface.vertices)
        self.triangles = np.asarray(self.surface.triangles)
        self.triangle_normals = np.asarray(self.surface.triangle_normals)
        self.tree = KDTree(self.vertices)  # Build KD-Tree on vertex positions
        self.iteration = 0

        self.get_logger().info("Creating mesh")
        self.get_logger().info("Computing adj list")
        self.surface.compute_adjacency_list()
        self.get_logger().info("initializing edge adj list")
        self.mesh = math_utils.Mesh(self.vertices, self.triangles, self.triangle_normals, self.surface.adjacency_list)
        self.get_logger().info("Mesh created")


        self.trianglesXfm = np.zeros((len(self.triangles), 4, 4))
        self.trianglesXfmInv = np.zeros((len(self.triangles), 4, 4))
        self.get_logger().info("Creating Xfm")
        for i, triangle in enumerate(self.triangles):
            P1, P2, P3 = [self.vertices[j] for j in triangle]
            self.trianglesXfm[i] = math_utils.compute_triangle_xfm(P1, P2, P3)
            self.trianglesXfmInv[i] = np.linalg.inv(self.trianglesXfm[i])
        self.get_logger().info("Xfm created")
        self.get_logger().info("trianglesXfm size is: " + str(self.trianglesXfm[0].shape))
        self.nearest_faces = o3d.geometry.TriangleMesh()
        self.viz.add_geometry(self.nearest_faces)


        # Initialize sphere properties (for visualization of the virtual fixture)
        self.sphere_radius = 0.01
        self.sphere_center =[0.05, 0.501, 0.05]
        self.sphere_target_center =  [0.05, 0.501, 0.05]
        self.old_sphere_center = self.sphere_center
        self.old_sphere_target_center = self.sphere_target_center

        # Create and add sphere and target sphere geometries
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_radius)
        # self.sphere.scale(1.02, center=(0, 0, 0))
        self.sphere.translate(self.sphere_center)
        self.sphere.paint_uniform_color([0, 1, 0])  # Green color for the sphere
        self.viz.add_geometry(self.sphere)

        self.sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_radius)
        self.sphere_target.translate(self.sphere_target_center)
        self.sphere_target.paint_uniform_color([1, 0, 0])  # Red color for the target sphere
        self.viz.add_geometry(self.sphere_target)

        self.planes_list = []
        
        
        
        

        # Add reference frame and ray line to the visualization
        self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.viz.add_geometry(self.reference_frame)
        self.ray_line = o3d.geometry.LineSet()
        self.ray_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0]])
        self.ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        self.ray_line.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))  # Red color for the line
        self.viz.add_geometry(self.ray_line)

        # Initialize camera parameters and optionally load saved viewpoint
        self.view_point_filename = "view_point.json"
        if os.path.exists(self.view_point_filename):
            self.get_logger().info(f"Loaded viewpoint from {self.view_point_filename}")
            self.load_view_point(self.view_point_filename)

        # Initialize movement parameters
        self.direction_vector = np.array([0.0, 0.0, 0.0])

        # QP
        # Initialize optimization variables
        self.delta_x = cp.Variable(3)
        
        # Start the main loop
        self._run_main_loop()

    def _find_nearby_triangles(self, position, max_distance):
        """
        Find triangles within a specified distance from the target position using KD-Tree.
        """
        # Convert target position to numpy array
        position = np.array(position)

        # Find nearby vertices using KD-Tree
        indices = self.tree.query_ball_point(position, max_distance)
        

        # Create a boolean array that checks if any of the vertices in each triangle match the given indices
        matches = np.isin(self.triangles, indices)

        # A triangle is part of `local_area` if any its vertices are in the provided indices
        triangle_in_local_area = np.any(matches, axis=1)
        
        # Collect triangles that use these vertices
        trianges_idx = np.where(triangle_in_local_area)[0]
        
        nearby_triangles = []
        # Now select the triangles inside the local area
        # for triangle in trianges_idx:
        #     V1 = self.vertices[self.triangles[triangle][0]]
        #     V2 = self.vertices[self.triangles[triangle][1]]
        #     V3 = self.vertices[self.triangles[triangle][2]]
        #     closest_point = find_closest_point_on_triangle(position, self.trianglesXfm[triangle], self.trianglesXfmInv[triangle], V1, V2, V3)[0]
        #     distance = np.linalg.norm(position - closest_point)
        #     if distance <= max_distance:
        #         nearby_triangles.append(triangle)
        
        # return nearby_triangles
        return trianges_idx


    def compute_constraints(self, nearby_triangles):
        
        # List to store all constraint planes
        constraint_planes = []

        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            # Submit tasks for each triangle Ti
            futures = [executor.submit(self.compute_constraint_for_triangle, Ti) for Ti in nearby_triangles]
            
            # Collect results as they complete
            for future in futures:
                result = future.result()
                constraint_planes.extend(result)
        
        return constraint_planes
    
    
    def compute_constraint_for_triangle(self,Ti):
        constraint_planes_local = []

        # Get vertices of the triangle
        V1 = self.vertices[self.triangles[Ti][0]]
        V2 = self.vertices[self.triangles[Ti][1]]
        V3 = self.vertices[self.triangles[Ti][2]]

        # Find the closest point on the triangle to the sphere center
        CPi, triangle_pos = find_closest_point_on_triangle(self.sphere_center, self.trianglesXfm[Ti], self.trianglesXfmInv[Ti], V1, V2, V3)

        # Normalize the triangle normal
        Ni = self.triangle_normals[Ti] / np.linalg.norm(self.triangle_normals[Ti])

        # Check if CPi is in the triangle and the normal points towards the sphere center
        if triangle_pos == Location.IN and Ni.T @ (self.sphere_center - CPi) >= 0:
            constraint_planes_local.append([Ni, CPi])

        # Handle points on the vertex
        elif triangle_pos in (Location.V1, Location.V2, Location.V3):
            if np.linalg.norm(self.sphere_center - CPi) < 0.001:
                # Average normals for stability
                active_normal = Ni.copy()
                neighbors = self.mesh.adjacency_dict[(Ti, triangle_pos+3)]
                for neighbor in neighbors:  # This should be the correct neighbors for the vertex
                    if neighbor is None:
                        continue

                    # Find closest points of the neighboring triangles
                    V1_neighbor = self.vertices[self.triangles[neighbor][0]]
                    V2_neighbor = self.vertices[self.triangles[neighbor][1]]
                    V3_neighbor = self.vertices[self.triangles[neighbor][2]]
                    CP_neighbor, triangle_pos_neighbor = find_closest_point_on_triangle(
                        self.sphere_center, 
                        self.trianglesXfm[neighbor], 
                        self.trianglesXfmInv[neighbor], 
                        V1_neighbor, 
                        V2_neighbor, 
                        V3_neighbor
                    )

                    if np.linalg.norm(CPi - CP_neighbor) < 0.001 and self.mesh.is_locally_concave(Ti, neighbor, triangle_pos_neighbor):
                        # Accumulate normals and average later
                        active_normal += self.triangle_normals[neighbor] / np.linalg.norm(self.triangle_normals[neighbor])

                # Normalize the averaged normal
                active_normal /= np.linalg.norm(active_normal)
                constraint_planes_local.append([active_normal, CPi])
        else:
            neighbors = self.mesh.adjacency_dict[(Ti, triangle_pos)]
            for neighbor in neighbors:
                if neighbor is None:
                    continue

                neighbor_face = self.triangles[neighbor]
                V1_neighbor = self.vertices[neighbor_face[0]]
                V2_neighbor = self.vertices[neighbor_face[1]]
                V3_neighbor = self.vertices[neighbor_face[2]]

                CPia, triangle_pos_a = find_closest_point_on_triangle(self.sphere_center, self.trianglesXfm[neighbor], self.trianglesXfmInv[neighbor], V1_neighbor, V2_neighbor, V3_neighbor)

                if np.linalg.norm(CPi - CPia) < 0.001:
                    if self.mesh.is_locally_convex(Ti, neighbor, triangle_pos_a):
                        N = (self.sphere_center - CPi) / np.linalg.norm(self.sphere_center - CPi)
                        constraint_planes_local.append([N, CPi])

                    elif Ni.T @ (self.sphere_center - CPi) >= 0 and self.mesh.is_locally_concave(Ti, neighbor, triangle_pos):
                        constraint_planes_local.append([Ni, CPi])

        return constraint_planes_local


    def _enforce_virtual_fixture(self, target_position, sphere_radius):
        """
        Enforce the virtual fixture by adjusting the sphere center based on nearby triangles.
        """
        # Define a small buffer distance to prevent penetration
        lookup_area = 0.001

        # Find nearby triangles within a specified distance
        max_distance = sphere_radius + lookup_area
        nearby_triangles = self._find_nearby_triangles(self.sphere_center, max_distance)
        
        if len(nearby_triangles) == 0:
            # No nearby triangles found; return the target position
            return target_position - self.sphere_center

        # Compute the constraint planes based on the nearby triangles
        constraint_planes = self.compute_constraints(nearby_triangles)
                    
        constraints = []
        self.delta_x = cp.Variable(3)
        for plane in constraint_planes:
            n = plane[0]
            x = self.sphere_center
            p = plane[1]
            # n^T ∆x ≥ -n^T (x - p)
            constraints.append(n.T @ self.delta_x >= -n.T @ (x - p) + sphere_radius)
            
            if VISUALIZE_PLANE_CONSTRAINTS:
                # plot the plane
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

                self.planes_list.append(plane_mesh)
                self.viz.add_geometry(plane_mesh, reset_bounding_box=False)
            
            
            
            # constraints.append(n.T @ self.delta_x >= -n.T @ p)
        # Solve the optimization problem that accounts also for the radius
        
        direction = (target_position - self.sphere_center)/np.linalg.norm(target_position - self.sphere_center)
        step_size = min(np.linalg.norm(target_position - self.sphere_center), self.sphere_radius)
        delta_x_des = direction * step_size
        objective = cp.Minimize(cp.norm(self.delta_x - delta_x_des))
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)

        if problem.status != cp.OPTIMAL:
            self.get_logger().warn(f"Optimization problem not solved optimally: {problem.status}")
            return np.zeros(3)
        
        # Return the adjusted sphere center     
        new_center = self.delta_x.value
        
        return new_center

    def _move_sphere(self, current, direction_vector, speed):
        """
        Update the sphere's position based on direction vector and speed.
        """
        return current + direction_vector * speed

    def _run_main_loop(self):
        """
        Run the main loop to update visualization and enforce virtual fixture constraints.
        """
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.direction_vector[2] -= 1.0
                    if event.key == pygame.K_s:
                        self.direction_vector[2] += 1.0
                    if event.key == pygame.K_a:
                        self.direction_vector[0] += 1.0
                    if event.key == pygame.K_d:
                        self.direction_vector[0] -= 1.0
                    if event.key == pygame.K_q:
                        self.direction_vector[1] -= 1.0
                    if event.key == pygame.K_e:
                        self.direction_vector[1] += 1.0
                    if event.key == pygame.K_v:
                        self.save_view_point(self.view_point_filename)

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        self.direction_vector[2] += 1.0
                    if event.key == pygame.K_s:
                        self.direction_vector[2] -= 1.0
                    if event.key == pygame.K_a:
                        self.direction_vector[0] -= 1.0
                    if event.key == pygame.K_d:
                        self.direction_vector[0] += 1.0
                    if event.key == pygame.K_q:
                        self.direction_vector[1] += 1.0
                    if event.key == pygame.K_e:
                        self.direction_vector[1] -= 1.0

            # self.get_logger().info(f"Direction vector: {self.direction_vector}")
            # Update target sphere position
            self.sphere_target_center = self._move_sphere(self.old_sphere_target_center, self.direction_vector, MOVEMENT_SPEED)
            self.sphere_target.translate(self.sphere_target_center, relative=False)
            self.viz.update_geometry(self.sphere_target)
            
            # Enforce virtual fixture on the target position
            constrained_position = self._enforce_virtual_fixture(self.sphere_target_center, self.sphere_radius)
            if np.linalg.norm(constrained_position) > 1e-6:
                damping_factor = 0.9  # Adjust this value to control the damping effect
                constrained_position = damping_factor * constrained_position
                self.sphere_center = self._move_sphere(self.old_sphere_center, constrained_position/np.linalg.norm(constrained_position), np.linalg.norm(constrained_position))
            self.sphere.translate(self.sphere_center, relative=False)
            self.viz.update_geometry(self.sphere)
            # Poll for new events and update the renderer
            self.viz.poll_events()
            self.viz.update_renderer()
            if VISUALIZE_PLANE_CONSTRAINTS:
                for plane in self.planes_list:
                    self.viz.remove_geometry(plane, reset_bounding_box=False)
                self.planes_list = []
            # Update old positions for the next iteration
            self.old_sphere_target_center = self.sphere_target_center
            self.old_sphere_center = self.sphere_center
            self.iteration += 1
            
    def save_view_point(self, filename):
        """
        Save the current viewpoint of the visualizer to a JSON file.
        """
        self.get_logger().info(f"Saving view point to {filename}")
        param = self.viz.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)

    def load_view_point(self, filename):
        """
        Load and apply a saved viewpoint from a JSON file.
        """
        self.get_logger().info(f"Loading view point from {filename}")
        ctr = self.viz.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(filename)
        ctr.convert_from_pinhole_camera_parameters(param)


def main(args=None):
    rclpy.init(args=args)
    demo = VirtualFixtureDemo()
    rclpy.spin(demo)
    demo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
