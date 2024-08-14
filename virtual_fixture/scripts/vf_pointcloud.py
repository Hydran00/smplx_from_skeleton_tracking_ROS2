#!/usr/bin/env python3

import open3d as o3d
import os
import numpy as np
import pygame
import time
import rclpy
from rclpy.node import Node

# Set up environment variable for SDL
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Define constants
POINT_RADIUS = 0.01
MOVEMENT_SPEED = 0.003

class VirtualFixtureDemo(Node):
    def __init__(self):
        super().__init__('virtual_fixture_demo')
        self.get_logger().info("Starting")

        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True
        opt.mesh_show_wireframe = True
        
        # Define screen dimensions and initialize display
        screen_width, screen_height = 50, 50
        pygame.init()
        pygame.display.set_mode((screen_width, screen_height))        
        
        # Load and prepare the surface
        load_path = os.path.expanduser('~') + "/SKEL_WS/ros2_ws/projected_skel.ply"
        self.surface = o3d.io.read_triangle_mesh(load_path)
        self.viz.add_geometry(self.surface)
        self.surface.translate((0, 0.5, 0), relative=False)
        self.viz.add_geometry(self.surface)

        # Init raycasting
        self.get_logger().info("Init raycasting scene")
        self.scene = o3d.t.geometry.RaycastingScene()
        mesh_new = o3d.t.geometry.TriangleMesh.from_legacy(self.surface)
        self.scene.add_triangles(mesh_new)
        
        # Initialize sphere properties
        sphere_radius = POINT_RADIUS
        self.sphere_center = [0.05,0.501,0.25226753]#self.surface.get_center() + (0.05, 0.1, 0.16)
        self.sphere_target_center = [0.05,0.501,0.25226753]
        self.old_sphere_center = self.sphere_center
        self.old_sphere_target_center = self.sphere_target_center
        # self.sphere_center = self._enforce_virtual_fixture(self.sphere_center, self.surface, sphere_radius)

        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        self.sphere.translate(self.sphere_center)
        self.sphere.paint_uniform_color([0, 1, 0])
        self.viz.add_geometry(self.sphere)

        self.sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        self.sphere_target.translate(self.sphere_target_center)
        self.sphere_target.paint_uniform_color([1, 0, 0])
        self.viz.add_geometry(self.sphere_target)

        self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.viz.add_geometry(self.reference_frame)
        self.ray_line = o3d.geometry.LineSet()
        self.ray_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0]])
        self.ray_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        self.ray_line.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        self.viz.add_geometry(self.ray_line)

        # Initialize camera parameters
        self.view_point_filename = "view_point.json"
        if os.path.exists(self.view_point_filename):
            self.get_logger().info(f"Loaded viewpoint from {self.view_point_filename}")
            self.load_view_point("view_point.json")

        # Initialize movement parameters
        self.direction_vector = np.array([0.0, 0.0, 0.0])
        

        # Main loop
        self._run_main_loop()

    def _enforce_virtual_fixture(self, sphere_center, surface, sphere_radius):
        # distance, closest_point = self._point_to_mesh_distance(sphere_center, surface)
        
        # if distance < sphere_radius:
        #     direction = sphere_center - closest_point
        #     direction /= np.linalg.norm(direction)
        #     sphere_center = closest_point + direction * sphere_radius

        # if not self._is_sphere_outside(sphere_center, surface, sphere_radius):
        #     direction = sphere_center - closest_point
        #     direction /= np.linalg.norm(direction)
        #     sphere_center = closest_point + direction * sphere_radius


        
        sphere_center = self.enforce(self.sphere_target_center, self.old_sphere_center, sphere_radius)


        return sphere_center


    def enforce(self,sphere_center,old_sphere_center, sphere_radius):
        self.get_logger().info("Sphere center: " + str(sphere_center))
        ray_direction = sphere_center - old_sphere_center
        ray_direction_normalized = ray_direction / np.linalg.norm(ray_direction)
        ray_start = old_sphere_center

        rays = o3d.core.Tensor([ [*ray_start, *ray_direction_normalized]], dtype=o3d.core.Dtype.Float32)
        intersections = self.scene.cast_rays(rays)

        self.ray_line.points = o3d.utility.Vector3dVector([ray_start, ray_start + ray_direction_normalized*0.1])
        self.viz.update_geometry(self.ray_line)
        # check if the ray intersects with the surface
        # no intersection
        self.get_logger().info(f"t_hit: {intersections['t_hit'].cpu().numpy()}")
        if intersections['t_hit'][0].cpu().numpy() == np.inf:
            self.get_logger().info("No intersection")
            return sphere_center

        # intersection occurs after the target point

        if np.linalg.norm(ray_direction) + sphere_radius < intersections['t_hit'][0].cpu().numpy():
            self.get_logger().info("A")
            distance, closest_point = self._point_to_mesh_distance(sphere_center, self.surface)
            if distance < sphere_radius:
                direction = sphere_center - closest_point
                direction /= np.linalg.norm(direction)
                sphere_center = closest_point + direction * sphere_radius
            
            return sphere_center
        
        # intersection occurs before the target point -> enforce VF
        else:
            self.get_logger().info("B")
            # return intersections['t_hit'][0].cpu().numpy()*ray_direction_normalized + ray_start - sphere_radius*ray_direction_normalized

            # compute closest point on the surface
            distance, closest_point = self._point_to_mesh_distance(sphere_center, self.surface)
            if distance < sphere_radius:
                direction = sphere_center - closest_point
                direction /= np.linalg.norm(direction)
                sphere_center = closest_point + direction * sphere_radius
            return sphere_center



    def _is_sphere_outside(self, sphere_center, surface, sphere_radius):
        distance, _ = self._point_to_mesh_distance(sphere_center, surface)
        return distance >= sphere_radius

    def _move_sphere(self, sphere_center, direction_vector, speed):
        return sphere_center + direction_vector * speed

    def _point_to_mesh_distance(self, point, mesh):
        query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
        ans = self.scene.compute_closest_points(query_point)
        closest_point = ans['points'][0].cpu().numpy()
        distance = np.linalg.norm(query_point - closest_point)
        return distance, closest_point

    def _run_main_loop(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self._handle_key_events(event)
            
            self.sphere_center = self._move_sphere(self.sphere_center, self.direction_vector, MOVEMENT_SPEED)
            
            self.sphere_center = self._enforce_virtual_fixture(self.sphere_center, self.surface, POINT_RADIUS)
            self.sphere.translate(self.sphere_center, relative=False)
            self.viz.update_geometry(self.sphere)
            
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.03)
            self.old_sphere_center = self.sphere_center

    def _handle_key_events(self, event):
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
        param = self.viz.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(filename, param)
        self.get_logger().info(f"View point saved to {filename}")

    def load_view_point(self, filename):
        ctr = self.viz.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(filename)
        ctr.convert_from_pinhole_camera_parameters(param, True)

if __name__ == "__main__":
    rclpy.init()
    demo = VirtualFixtureDemo()
    rclpy.shutdown()
