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


# Set up environment variable for SDL
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Define constants for visualization and movement
POINT_RADIUS = 0.005
MOVEMENT_SPEED = 0.003

class VirtualFixtureDemo(Node):
    def __init__(self):
        super().__init__('virtual_fixture_demo')
        self.get_logger().info("Starting")

        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True  # Display coordinate frame
        opt.mesh_show_wireframe = True    # Show mesh wireframe
        
        # Define screen dimensions and initialize display using pygame
        screen_width, screen_height = 50, 50
        pygame.init()
        pygame.display.set_mode((screen_width, screen_height))        
        
        # Load and prepare the surface (bunny mesh)
        dataset = o3d.data.BunnyMesh()
        self.surface = o3d.io.read_triangle_mesh(dataset.path)
        # self.surface = o3d.io.read_triangle_mesh(dataset.path)
        load_path = os.path.expanduser('~') + "/SKEL_WS/ros2_ws/projected_skel.ply"
        # self.surface = o3d.io.read_triangle_mesh(load_path)
        
        self.surface.compute_vertex_normals()  # Compute normals for the mesh
        self.viz.add_geometry(self.surface)
        self.surface.translate((0, 0.5, 0), relative=False)
        self.viz.add_geometry(self.surface)

        # Initialize raycasting scene for virtual fixture enforcement
        self.get_logger().info("Init raycasting scene")
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_new = o3d.t.geometry.TriangleMesh.from_legacy(self.surface)
        self.scene.add_triangles(mesh_new)
        
        self.vertices = np.asarray(self.surface.vertices)
        self.triangles = np.asarray(self.surface.triangles)
        self.triangle_normals = np.asarray(self.surface.triangle_normals)
        self.tree = KDTree(self.vertices)  # Build KD-Tree on vertex positions

        # Initialize sphere properties (for visualization of the virtual fixture)
        sphere_radius = POINT_RADIUS
        self.sphere_center = [0.05, 0.501, 0.25226753]
        self.sphere_target_center = [0.05, 0.501, 0.25226753]
        self.old_sphere_center = self.sphere_center
        self.old_sphere_target_center = self.sphere_target_center

        # Create and add sphere and target sphere geometries
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        self.sphere.translate(self.sphere_center)
        self.sphere.paint_uniform_color([0, 1, 0])  # Green color for the sphere
        self.viz.add_geometry(self.sphere)

        self.sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        self.sphere_target.translate(self.sphere_target_center)
        self.sphere_target.paint_uniform_color([1, 0, 0])  # Red color for the target sphere
        self.viz.add_geometry(self.sphere_target)

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

        # Start the main loop
        self._run_main_loop()

    def _closest_point_on_triangle(self, triangle_vertices, point):
        """
        Compute the closest point on a triangle to a given point.
        """
        A = triangle_vertices[0]
        B = triangle_vertices[1]
        C = triangle_vertices[2]

        AB = B - A
        AC = C - A
        BC = C - B
        AP = point - A
        BP = point - B
        CP = point - C

        d1 = np.dot(AB, AP)
        d2 = np.dot(AC, AP)
        d3 = np.dot(AB, BP)
        d4 = np.dot(AC, BP)
        d5 = np.dot(BC, CP)
        d6 = np.dot(-AB, CP)

        # Check and return the closest point based on triangle edge and face tests
        if d1 <= 0 and d2 <= 0:
            return A
        if d3 >= 0 and d6 <= 0:
            return B
        if d4 >= 0 and d5 >= 0:
            return C

        vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            v = d1 / (d1 - d3)
            return A + v * AB

        vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            w = d2 / (d2 - d6)
            return A + w * AC

        va = d3 * d6 - d5 * d4
        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
            u = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return B + u * (C - B)

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return A + AB * v + AC * w

    def _find_nearby_triangles(self, target_position, max_distance):
        """
        Find triangles within a specified distance from the target position using KD-Tree.
        """
        # Convert target position to numpy array
        target_position = np.array(target_position)

        # Find nearby vertices using KD-Tree
        indices = self.tree.query_ball_point(target_position, max_distance)
        
        # Collect triangles that use these vertices
        nearby_triangles = []
        # for idx in indices:
            # Check which triangles contain the vertex
            # for triangle_idx, triangle in enumerate(self.triangles):
            #     if idx in triangle:
            #         # Compute the closest point on the triangle and its distance
            #         A, B, C = self.vertices[triangle]
            #         closest_point = self._closest_point_on_triangle([A, B, C], target_position)
            #         distance = np.linalg.norm(closest_point - target_position)
            #         if distance <= max_distance:
            #             nearby_triangles.append((triangle, distance))

            # use select_by_index instead
            # triangles = self.triangles[np.where(self.triangles == idx)[0]]
            # for triangle in triangles:
            #     A, B, C = [self.vertices[i] for i in triangle]
            #     closest_point = self._closest_point_on_triangle([A, B, C], target_position)
            #     distance = np.linalg.norm(closest_point - target_position)
            #     if distance <= max_distance:
            #         nearby_triangles.append((triangle, distance))

        local_area = self.surface.select_by_index(indices)
        for triangle in np.asarray(local_area.triangles):
            A, B, C = [self.vertices[i] for i in triangle]
            closest_point = self._closest_point_on_triangle([A, B, C], target_position)
            distance = np.linalg.norm(closest_point - target_position)
            if distance <= max_distance:
                nearby_triangles.append((triangle, distance))

        return nearby_triangles

    def _enforce_virtual_fixture(self, target_position, sphere_radius):
        """
        Enforce the virtual fixture by adjusting the sphere center based on nearby triangles.
        """
        # Define a small buffer distance to prevent penetration
        buffer_distance = 0.001

        # Find nearby triangles within a specified distance
        max_distance = sphere_radius + buffer_distance
        nearby_triangles = self._find_nearby_triangles(target_position, max_distance)
        
        if not nearby_triangles:
            # No nearby triangles found; return the target position
            return target_position

        # Initialize optimization variables
        delta_x = cp.Variable(3)
        objective = cp.Minimize(cp.norm(delta_x - (target_position - self.sphere_center)))
        
        # Construct constraints based on distances to nearby triangles
        constraints = []
        for triangle, distance in nearby_triangles:
            A, B, C = [self.vertices[i] for i in triangle]
            closest_point = self._closest_point_on_triangle([A, B, C], np.array(target_position))
            triangle_normal = self.triangle_normals[np.where((self.triangles == triangle).all(axis=1))[0][0]]
            
            # Constraint to ensure sphere does not penetrate the mesh
            constraints.append(triangle_normal.T @ delta_x >= -triangle_normal.T @ (self.sphere_center - closest_point) + buffer_distance)
        self.get_logger().info(f"Constraints length: {len(constraints)}")
        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return self.sphere_center + delta_x.value


    def _move_sphere(self, current, direction_vector, speed):
        """
        Update the sphere's position based on direction vector and speed.
        """
        return current + direction_vector * speed

    def _run_main_loop(self):
        """
        Main loop to handle visualization updates and user input.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self._handle_key_events(event)
            
            # Update target sphere position and move it
            self.sphere_target_center = self._move_sphere(self.old_sphere_target_center, self.direction_vector, MOVEMENT_SPEED)
            self.sphere_target.translate(self.sphere_target_center, relative=False)
            self.viz.update_geometry(self.sphere_target)
            
            # Enforce the virtual fixture and update sphere position
            self.vf = self._enforce_virtual_fixture(self.sphere_target_center, POINT_RADIUS)
            self.sphere_center = self.old_sphere_center + (np.array(self.vf) - np.array(self.old_sphere_center))*0.1
            self.sphere.translate(self.sphere_center, relative=False)
            self.viz.update_geometry(self.sphere)
            
            # Poll for new events and update the renderer
            self.viz.poll_events()
            self.viz.update_renderer()
            # time.sleep(0.03)
            
            # Update old positions
            self.old_sphere_center = self.sphere_center
            self.old_sphere_target_center = self.sphere_target_center

    def _handle_key_events(self, event):
        """
        Handle user input for controlling the direction vector and saving/loading view points.
        """
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

    def save_view_point(self, filename):
        """
        Save the current viewpoint to a file.
        """
        param = self.viz.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)
        self.get_logger().info(f"View point saved to {filename}")

    def load_view_point(self, filename):
        """
        Load a saved viewpoint from a file.
        """
        ctr = self.viz.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(filename)
        ctr.convert_from_pinhole_camera_parameters(param, True)

if __name__ == "__main__":
    rclpy.init()
    demo = VirtualFixtureDemo()
    rclpy.shutdown()

    # TODO vedere capire perche il codice vecchio che usava solo un triangolo funzionava
    # TODO capire se il problema sono le normali 