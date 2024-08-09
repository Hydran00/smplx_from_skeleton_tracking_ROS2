#!/usr/bin/env python3

import open3d as o3d
import os

from materials import mat_sphere_transparent, mat_skin

from ament_index_python.packages import get_package_share_directory
from utils import *
import time
import pygame

# Initialize pygame for keyboard input
pygame.init()

POINT_RADIUS = 0.01

# Create a function to enforce the virtual fixture
def enforce_virtual_fixture(sphere_center, surface, sphere_radius):
    distance, closest_point = point_to_mesh_distance(sphere_center, surface)
    
    # If the sphere is intersecting the surface, adjust its position
    if distance < sphere_radius:
        # Compute the direction to move the sphere
        direction = sphere_center - closest_point
        direction /= np.linalg.norm(direction)
        
        # Move the sphere center to be on top of the surface
        sphere_center = closest_point + direction * sphere_radius
    
    return sphere_center

def move_sphere(sphere_center, direction_vector, speed):
    return sphere_center + direction_vector * speed

def point_to_mesh_distance(point, mesh):
    # mesh = mesh.from_legacy()
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh)
    # Add triangle meshes and remember ids
    query_point = o3d.core.Tensor([point], dtype=o3d.core.Dtype.Float32)
    ans = scene.compute_closest_points(query_point)
    closest_point = ans['points'][0].cpu().numpy()
    distance = np.linalg.norm(query_point - closest_point)
    print("Distance from closest is ", distance)
    
    # triangle_colors = np.ones((len(mesh.triangle.indices), 3))
    # # triangle_colors[::3] = [1, 0, 0]
    # triangle_colors[ans['primitive_ids'][0].cpu().numpy(),:] = [1, 0, 0]
    # mesh.triangle.colors = o3d.core.Tensor(triangle_colors, dtype=o3d.core.float32)
    
    return distance, closest_point

class TestVF:
    def __init__(self):

        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True
        opt.mesh_show_wireframe = True
        # Define screen dimensions and initialize display
        screen_width, screen_height = 50, 50
        screen = pygame.display.set_mode((screen_width, screen_height))        
        
        load_path = os.path.expanduser('~')+"/ros2_ws/projected_skel.ply"
        surface =o3d.io.read_triangle_mesh(load_path)
        self.viz.add_geometry(surface)
        # Create a point cloud from the sphere center
        self.sphere_center = surface.get_center() + (0.05,0.1,-0.16)
        sphere_radius = 0.01
        self.sphere_center = enforce_virtual_fixture(self.sphere_center, surface, sphere_radius)

        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        self.sphere.translate(self.sphere_center)
        self.viz.add_geometry(self.sphere)
        
        # MOVEMENT PARAMS
        self.speed = 0.003  # Movement speed per iteration

        self.direction_vector = np.array([0.0, 0.0, 0.0])  # Movement direction vector
        # Main loop (example with a static sphere, can be adapted for dynamic updates)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Movement controls
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.direction_vector[2] += 1.0  # Move forward
                    if event.key == pygame.K_s:
                        self.direction_vector[2] -= 1.0  # Move backward
                    if event.key == pygame.K_a:
                        self.direction_vector[0] -= 1.0  # Move left
                    if event.key == pygame.K_d:
                        self.direction_vector[0] += 1.0  # Move right
                    if event.key == pygame.K_q:
                        self.direction_vector[1] -= 1.0  
                    if event.key == pygame.K_e:
                        self.direction_vector[1] += 1.0  


                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        self.direction_vector[2] -= 1.0  # Stop moving forward
                    if event.key == pygame.K_s:
                        self.direction_vector[2] += 1.0  # Stop moving backward
                    if event.key == pygame.K_a:
                        self.direction_vector[0] += 1.0  # Stop moving left
                    if event.key == pygame.K_d:
                        self.direction_vector[0] -= 1.0  # Stop moving right
                    if event.key == pygame.K_q:
                        self.direction_vector[1] += 1.0
                    if event.key == pygame.K_e:
                        self.direction_vector[1] -= 1.0
                
                
            self.sphere_center = move_sphere(self.sphere_center, self.direction_vector, self.speed)

            # Enforce the virtual fixture
            self.sphere_center = enforce_virtual_fixture(self.sphere_center, surface, sphere_radius)
            # Visualize the updated scene
            self.sphere.translate(self.sphere_center,relative=False)
            self.viz.update_geometry(self.sphere)

            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.03)
            # For demonstration purposes, exit the loop after one iteration
            # break




    #     self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
    #     self.viz.add_geometry(self.reference_frame)
        
    #     # start = mesh.get_center() + (0.05,0.2,-0.16)
    #     start = np.array([0.0,0.0,0.005])
    #     end = np.array([0.2,0.2,-0.005])
        
    #     print("Starting point is ", start)
    #     self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    #     self.sphere.compute_vertex_normals()
    #     self.sphere.paint_uniform_color([1, 0, 0])
    #     self.sphere.translate(start, relative=False)
    #     self.viz.add_geometry(self.sphere)
    #     traj = self.linear_interpolation_movement(start, end)

    #     # draw trajectory
    #     start = traj[0]
    #     end = traj[-1]
    #     self.line = o3d.geometry.LineSet()
    #     self.line.points = o3d.utility.Vector3dVector([start, end])
    #     self.line.lines = o3d.utility.Vector2iVector([[0, 1]])
    #     self.line.colors = o3d.utility.Vector3dVector([[0,0,0]])
    #     self.viz.add_geometry(self.line)
        
    #     self.vf_position = traj[0]
    #     self.vf_position_last = traj[0]
        
    #     # self.block()
    #     # self.current_desired = traj[0]

    #     self.simulate_motion(traj)
        
    # def block(self):
    #     while True:
    #         self.viz.poll_events()
    #         self.viz.update_renderer()
    #         time.sleep(0.03)
        
    # def linear_interpolation_movement(self,start, distance):
    #     """
    #     Linear interpolation of a movement
    #     """
    #     traj = []
    #     steps = 2000
    #     for i in range(steps):
    #         traj.append(start + i*distance/steps)
    #     return traj
    
    # def simulate_motion(self,traj):
    #     """
    #     Simulate the motion of the sphere
    #     """
    #     for point in traj:
    #         transl = self.computeVF(point, POINT_RADIUS)
    #         self.sphere.translate(transl, relative=False)
            
    #         self.viz.update_geometry(self.sphere)
    #         self.viz.poll_events()
    #         self.viz.update_renderer()
    #         time.sleep(0.01)
            
    #         self.vf_position_last = self.vf_position
    
    # def computeVF(self,point, point_radius=0.01):
    #     """
    #     Consider each point of the cloud as a sphere, find the closest and check the point is inside the sphere
    #     If it is, then project the point to the sphere
    #     """
        
    #     #find the closest point
    #     distances = np.linalg.norm(self.points-self.vf_position, axis=1)
    #     closest_idx = np.argmin(distances)
    #     closest_point = self.points[closest_idx,:]
    #     # update closest sphere
    #     self.sphere_closest.translate(closest_point, relative=False)
    #     # self.sphere_desired.translate(self.current_desired, relative=False)
    #     # check if the point is inside the sphere
    #     distance = np.linalg.norm(closest_point - point)
        
    #     set_point = None
    #     if distance < point_radius:
    #         # project the point to the sphere
    #         direction = self.vf_position - closest_point
            
    #         # check if the direction is similar to the normal of that point
    #         if np.dot(direction, [0,0,1]) < 0:
    #             direction = -direction
            
    #         direction = direction/np.linalg.norm(direction)
    #         projected_point = closest_point + direction*point_radius
    #         set_point = projected_point
            
    #         # apply parallel component
    #         parallel_component = np.dot(point - projected_point, direction)*direction
    #         set_point = projected_point + parallel_component
            
    #     else:
    #         set_point = point
        
        
    #     # self.current_desired += (set_point - self.current_desired)*0.1
    #     return set_point
        
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    test = TestVF()
