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

# Set up environment variable for SDL
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Define constants for visualization and movement
POINT_RADIUS = 0.002
MOVEMENT_SPEED = 0.00005
VISUALIZE_LOCAL_AREA = True
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
        self.surface = o3d.io.read_triangle_mesh(os.path.expanduser('~') + '/SKEL_WS/ros2_ws/Skull.stl')
        # scale down by 1000
        self.surface.scale(1/1000, center=(0, 0, 0))

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
        self.mesh = math_utils.Mesh(self.vertices, self.triangles, self.triangle_normals)
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
        sphere_radius = POINT_RADIUS
        self.sphere_center =[0.05, 0.501, 0.05]
        self.sphere_target_center =  [0.05, 0.501, 0.05]
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

        self.sphere_cp = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius/3)
        self.sphere_cp.translate(self.sphere_target_center)
        self.sphere_cp.paint_uniform_color([0, 0, 1])  # Red color for the cp sphere
        self.viz.add_geometry(self.sphere_cp)

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

    def _find_nearby_triangles(self, position, max_distance):
        """
        Find triangles within a specified distance from the target position using KD-Tree.
        """
        # Convert target position to numpy array
        position = np.array(position)

        # Find nearby vertices using KD-Tree
        indices = self.tree.query_ball_point(position, max_distance)
        
        # Collect triangles that use these vertices
        nearby_triangles = []
        
        if VISUALIZE_LOCAL_AREA:
            self.local_area = self.surface.select_by_index(indices)
            if self.iteration == 0:
                self.local_area_viz = o3d.geometry.TriangleMesh()
                self.local_area_viz.vertices = self.local_area.vertices
                self.local_area_viz.triangles = self.local_area.triangles
                self.local_area_viz.compute_vertex_normals()
                # paint red
                self.local_area_viz.paint_uniform_color([1, 1, 0])
                self.viz.add_geometry(self.local_area_viz)
            else:
                self.local_area_viz.vertices = self.local_area.vertices
                self.local_area_viz.triangles = self.local_area.triangles
                self.local_area_viz.compute_vertex_normals()
                self.local_area_viz.paint_uniform_color([1, 1, 0])
                self.viz.update_geometry(self.local_area_viz)

        # Create a boolean array that checks if any of the vertices in each triangle match the given indices
        matches = np.isin(self.triangles, indices)

        # A triangle is part of `local_area` if all its vertices are in the provided indices
        triangle_in_local_area = np.all(matches, axis=1)

        trianges_idx = np.where(triangle_in_local_area)[0]
        # Now select the triangles that match
        for triangle in trianges_idx:
            V1 = self.vertices[self.triangles[triangle][0]]
            V2 = self.vertices[self.triangles[triangle][1]]
            V3 = self.vertices[self.triangles[triangle][2]]
            closest_point = find_closest_point_on_triangle(position, self.trianglesXfm[triangle], self.trianglesXfmInv[triangle], V1, V2, V3)[0]
            distance = np.linalg.norm(position - closest_point)
            if distance <= max_distance:
                nearby_triangles.append(triangle)
        self.iteration += 1
        
        # return nearby_triangles
        # return np.where(triangle_in_local_area)[0]
        return nearby_triangles

    def _enforce_virtual_fixture(self, target_position, sphere_radius):
        """
        Enforce the virtual fixture by adjusting the sphere center based on nearby triangles.
        """
        # Define a small buffer distance to prevent penetration
        buffer_distance = 0.008

        # Find nearby triangles within a specified distance
        max_distance = sphere_radius + buffer_distance
        nearby_triangles = self._find_nearby_triangles(self.old_sphere_center, max_distance)
        # self.get_logger().info(f"{self.iteration}) Nearby triangles: {nearby_triangles}")
        if len(nearby_triangles) == 0:
            # No nearby triangles found; return the target position
            return target_position

        # Initialize optimization variables
        delta_x = cp.Variable(3)
        objective = cp.Minimize(cp.norm(delta_x - (target_position - self.sphere_center)))
        
        # Construct constraints based on distances to nearby triangles
        contraints_planes = []
        constraints = []


        ''' From the paper:

        Algorithm 1
        for triangle Ti ∈ T do
          if CPi in-triangle & N T  i (x - CPi) ≥ 0 then
            add {Ni, CPi} to L ;
          else if CPi on-edge then
            Find adjacent triangle(s) Ti,a ;
          if CPi == CPi,a & locally convex then
            add {x - CPi, CPi} to L ;
          else if N T  i (x - CPi) ≥ 0 & locally concave
              then  add {Ni, CPi} to L ; end
                
        '''
        # Ti is a triangle in the region 
        T = np.array(nearby_triangles)
        # CPi is the closest point on the triangle to the sphere center
        # self.get_logger().info(f"{self.iteration}) Nearby triangles shape: {T.shape}")
        
        for i, Ti in enumerate(T):
            V1 = self.vertices[self.triangles[Ti][0]]
            V2 = self.vertices[self.triangles[Ti][1]]
            V3 = self.vertices[self.triangles[Ti][2]]
            CPi, triangle_pos = find_closest_point_on_triangle(self.sphere_center, self.trianglesXfm[Ti], self.trianglesXfmInv[Ti], V1, V2, V3)
            Ni = self.triangle_normals[Ti]
            self.sphere_cp.translate(CPi, relative=False)
            self.viz.update_geometry(self.sphere_cp)
            # Check if CPi is in the triangle and the normal points towards the sphere center
            if  Ni.T @ (self.sphere_center - CPi) >= 0 and triangle_pos == Location.IN:
                contraints_planes.append([Ni, CPi])
                self.get_logger().info(f"{self.iteration}) CPi is in the triangle")
            # Check if CPi is on the edge
            elif triangle_pos != Location.IN:
                neighbors = self.mesh.get_neighbors_of_face(Ti, triangle_pos)
                nearest_vertices = np.zeros((len(neighbors), 3))
                nearest_faces = np.zeros((len(neighbors), 3))
                for n, neighbor in enumerate(neighbors):
                    if neighbor is None:
                        self.get_logger().info(f"{self.iteration}) Neighbor is none")
                        continue
                    self.get_logger().info(f"{self.iteration}) Neighbor: {neighbor}")
                    V1 = self.vertices[self.triangles[neighbor][0]]
                    V2 = self.vertices[self.triangles[neighbor][1]]
                    V3 = self.vertices[self.triangles[neighbor][2]]
                    nearest_vertices[n] = [V1, V2, V3]
                    nearest_faces[n] = self.triangles[neighbor]
                    CPa, triangle_pos_a = find_closest_point_on_triangle(self.sphere_center, self.trianglesXfm[neighbor], self.trianglesXfmInv[neighbor], V1, V2, V3)
                    if abs(np.linalg.norm(CPi - CPa))<1e-7 and self.mesh.is_locally_convex(Ti, neighbor, triangle_pos_a):
                        constraints_planes.append([self.sphere_center - CPi, CPi])
                    elif Ni.T @ (self.sphere_center - CPi) >= 0 and self.mesh.is_locally_concave(Ti, neighbor, triangle_pos):
                        contraints_planes.append([Ni, CPi])

                    self.nearest_faces.vertices = o3d.utility.Vector3dVector(nearest_vertices)
                    self.nearest_faces.triangles = o3d.utility.Vector3iVector(nearest_faces)
                    self.nearest_faces.compute_vertex_normals()
                    self.nearest_faces.paint_uniform_color([0, 1, 0])
                    self.viz.update_geometry(self.nearest_faces)



            # Adjusted constraint to ensure sphere stays on the surface
            # constraints.append(triangle_normal.T @ (self.sphere_center + delta_x - closest_point) >= buffer_distance+sphere_radius)

        for plane in contraints_planes:
            constraints.append(plane[0].T @ delta_x >= -plane[0].T @ (self.sphere_center - plane[1]))
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        # self.get_logger().info(f"{self.iteration}) Optimization status: {problem.status}")

        # Return the adjusted sphere center
        return self.sphere_center + delta_x.value

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
            constrained_position = self._enforce_virtual_fixture(self.sphere_target_center, POINT_RADIUS)
            self.sphere_center = constrained_position
            self.sphere.translate(self.sphere_center, relative=False)
            self.viz.update_geometry(self.sphere)

            # Poll for new events and update the renderer
            self.viz.poll_events()
            self.viz.update_renderer()

            # Update old positions for the next iteration
            self.old_sphere_center = np.array(self.sphere_center)
            self.old_sphere_target_center = np.array(self.sphere_target_center)


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
