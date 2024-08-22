#!/usr/bin/env python3

import open3d as o3d
import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import cvxpy as cp
from scipy.spatial import KDTree
import math_utils 
from closest_on_triangle import find_closest_point_on_triangle, Location

# Set up environment variable for SDL
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Define constants for visualization and movement
SPHERE_RADIUS = 0.01
BUFFER_AREA = SPHERE_RADIUS*2
VISUALIZE_PLANE_CONSTRAINTS = True

class VirtualFixtureDemo(Node):
    def __init__(self):
        super().__init__('virtual_fixture_demo')
        self.get_logger().info("Starting")

        self.surface = o3d.io.read_triangle_mesh(os.path.expanduser('~') + '/SKEL_WS/ros2_ws/projected_skel.ply')

        self.surface.compute_triangle_normals()
        self.surface.orient_triangles()
        self.surface.normalize_normals()
        
        self.mesh = math_utils.Mesh(np.asarray(self.surface.vertices), np.asarray(self.surface.triangles), np.asarray(self.surface.triangle_normals))
        self.tree = KDTree(self.mesh.vertices)  # Build KD-Tree on vertex positions
        self.iteration = 0

        self.mesh.triangle_xfm = self.mesh.triangle_xfm
        self.mesh.triangle_xfm_inv = self.mesh.triangle_xfm_inv


        self.sphere_radius = SPHERE_RADIUS
        self.sphere_center = self.sphere_target_center = [0.0,0.0,0.0]
        self.old_sphere_center = self.sphere_center
        self.old_sphere_target_center = self.sphere_target_center
        # QP
        # Initialize optimization variables
        self.delta_x = cp.Variable(3)
        self.pose_subscriber = self.create_subscription(PoseStamped, 'target_frame', self.pose_callback, 1)
        self.marker_publisher = self.create_publisher(Marker, 'visualization_marker', 1)
        self.pose_publisher = self.create_publisher(PoseStamped, 'target_frame_vf', 1)
        
        self.ros_input = np.array([0.0, 0.0, 0.0])
        # Start the main loop
        self._run_main_loop()

    def pose_callback(self, msg):
        self.ros_input = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])


    def _find_nearby_triangles(self, position, max_distance):
        """
        Find triangles within a specified distance from the target position using KD-Tree.
        """
        # Convert target position to numpy array
        position = np.array(position)

        # Find nearby vertices using KD-Tree
        indices = self.tree.query_ball_point(position, max_distance)
        

        # Create a boolean array that checks if any of the vertices in each triangle match the given indices
        matches = np.isin(self.mesh.faces, indices)

        # A triangle is part of `local_area` if any its vertices are in the provided indices
        triangle_in_local_area = np.any(matches, axis=1)
        
        # Collect triangles that use these vertices
        trianges_idx = np.where(triangle_in_local_area)[0]
        
        nearby_triangles = []
        return trianges_idx

    def get_closest_on_triangle(self, point, face):
        V1, V2, V3 = self.mesh.vertices[self.mesh.faces[face]]
        point, location = find_closest_point_on_triangle(self.sphere_center, self.mesh.triangle_xfm[face], self.mesh.triangle_xfm_inv[face], V1, V2, V3)
        return [point, location]

    def _enforce_virtual_fixture(self, target_position, sphere_radius):
        """
        Enforce the virtual fixture by adjusting the sphere center based on nearby triangles.
        """
        # Define a small buffer distance to prevent penetration
        lookup_area = BUFFER_AREA

        # Find nearby triangles within a specified distance
        max_distance = sphere_radius + lookup_area
        nearby_triangles = self._find_nearby_triangles(self.sphere_center, max_distance)
        
        if len(nearby_triangles) == 0:
            # No nearby triangles found; return the target position
            return target_position - self.sphere_center

        # Construct constraints based on distances to nearby triangles
        constraint_planes = []

        eps = 0.001
        T = np.array(nearby_triangles)
        
        # precompute CPi for all triangles
        CP = {Ti: self.get_closest_on_triangle(self.sphere_center, Ti) for Ti in T}
        
        # TODO check i integrtiy
        l = 0 # list index
        
        for i,Ti in enumerate(T):
            # Find the closest point on the triangle to the sphere center
            # CPi, triangle_pos = self.get_closest_on_triangle(self.sphere_center, Ti)
            CPi, cpi_loc = CP[Ti]
            # Normalize the triangle normal
            Ni = self.mesh.normals[Ti]
            Ni = Ni / np.linalg.norm(Ni)

            # Check if CPi is in the triangle and the normal points towards the sphere center
            if cpi_loc == Location.IN:
                if Ni.T @ (self.sphere_center - CPi) >= 0:
                    constraint_planes.append([Ni, CPi])
                    continue
            
            # Handle points on the edges or vertices
            else:
                # on vertex
                if cpi_loc in [Location.V1, Location.V2, Location.V3]:
                    n = np.copy(Ni)
                    if np.linalg.norm(self.sphere_center - CPi) < eps:
                        # find all same vertex, average and remove
                        for face in T[i+1:]:
                            CPia, _ = CP[face]
                            if(np.linalg.norm(CPia - CPi) < eps):
                                n += self.mesh.normals[face] / np.linalg.norm(self.mesh.normals[face])
                                # retrieve the index of the face to remove
                                idx = np.where(T == face)
                                T = np.delete(T, idx)
                                
                                    
                        # normalize normal
                        n /= np.linalg.norm(n)
                        constraint_planes.append([n, CPi])
                        l += 1
                        continue
                    # proceed as normal
                    else:
                        if cpi_loc == Location.V1:
                            neighborIdx1 = self.mesh.adjacency_dict[(Ti, Location.V1V2)]
                            neighborIdx2 = self.mesh.adjacency_dict[(Ti, Location.V1V3)]
                        elif cpi_loc == Location.V2:
                            neighborIdx1 = self.mesh.adjacency_dict[(Ti, Location.V1V2)]
                            neighborIdx2 = self.mesh.adjacency_dict[(Ti, Location.V2V3)]
                        elif cpi_loc == Location.V3:
                            neighborIdx1 = self.mesh.adjacency_dict[(Ti, Location.V1V3)]
                            neighborIdx2 = self.mesh.adjacency_dict[(Ti, Location.V2V3)]
                        else:
                            self.get_logger().info("Error: location not on vertex")
                            exit()
 
                        keep = False
                        if len(neighborIdx1) > 0:
                            neighborIdx1 = neighborIdx1[0]
                            is_in_cp_list = True
                            CPia,_ = CP.get(neighborIdx1, [None,None])
                            if CPia is None:
                                is_in_cp_list = False
                                CPia,_ = self.get_closest_on_triangle(self.sphere_center, neighborIdx1)
                            if (np.linalg.norm(CPia - CPi) < eps):
                                # remove neighbor1
                                for face in T[i+1:]:
                                    if face == neighborIdx1:
                                        idx = np.where(T == face)
                                        T = np.delete(T, idx)
                                        if is_in_cp_list:
                                            CP[face] = [CP[face][0],Location.VOID]
                                        keep = True
                                        break
                        if len(neighborIdx2) > 0:
                            neighborIdx2 = neighborIdx2[0]
                            is_in_cp_list = True
                            CPia,_ = CP.get(neighborIdx2, [None,None])
                            if CPia is None:
                                is_in_cp_list = False
                                CPia,_ = self.get_closest_on_triangle(self.sphere_center, neighborIdx2)
                            if (np.linalg.norm(CPia - CPi) < eps):
                                # remove neighbor2
                                for face in T[i+1:]:
                                    if face == neighborIdx2:
                                        idx = np.where(T == face)
                                        T = np.delete(T, idx)
                                        if is_in_cp_list:
                                            CP[face] = [CP[face][0],Location.VOID]
                                        keep = True
                                        break
                        if keep:
                            n = self.sphere_center - CPi
                            n /= np.linalg.norm(n)
                            constraint_planes.append([n, CPi])
                            l += 1
                            continue
                
                # on edge
                else:
                    if cpi_loc != Location.VOID:
                        # Get the neighboring triangle face index
                        neighborIdx = self.mesh.adjacency_dict[(Ti, cpi_loc)]
                        if len(neighborIdx) > 0:
                            neighborIdx = neighborIdx[0]
                            is_in_cp_list = True                            
                            CPia,cpia_loc = CP.get(neighborIdx, [None,None])
                            is_in_cp_list = True
                            if CPia is None:
                                is_in_cp_list = False
                                CPia,cpia_loc = self.get_closest_on_triangle(self.sphere_center, neighborIdx)
                            # if neightbor on the same edge, then closest point must have been identical. We proceed only when locally concave.
                            if(self.mesh.is_locally_concave(Ti, neighborIdx, cpi_loc)):
                                # Add constraint using the vector between sphere center and closest point
                                if np.linalg.norm(CPi - CPia) < eps:
                                    if np.linalg.norm(self.sphere_center - CPi) < eps:
                                        n = np.copy(Ni)
                                        n += self.mesh.normals[neighborIdx] / np.linalg.norm(self.mesh.normals[neighborIdx])
                                        n /= np.linalg.norm(n)
                                        constraint_planes.append([n, CPi])
                                        l += 1
                                        continue
                                    else:
                                        if is_in_cp_list:
                                            CP[neighborIdx] = [CP[neighborIdx][0],Location.VOID]
                                        
                                        for face in T[i+1:]:
                                            if face == neighborIdx:
                                                idx = np.where(T == face)
                                                T = np.delete(T, idx)
                                                break
                                        n = self.sphere_center - CPi
                                        n /= np.linalg.norm(n)
                                        constraint_planes.append([n, CPi])
                                        l += 1
                                        continue
                        
                            # if convex and on the positive side of normal
                            elif cpia_loc != Location.VOID and Ni.T @ (self.sphere_center - CPi) >= 0:
                                constraint_planes.append([Ni, CPi])
                                l += 1
                                continue
                        else:
                            continue
            
            # otherwise, delete current mesh
            CP[Ti] = [CP[Ti][0],Location.VOID]
            # print('Trying to remove', i, 'from', T)
            idx = np.where(T == Ti)
            T = np.delete(T, idx)
            
                        
        constraints = []
        self.delta_x = cp.Variable(3)
        for plane in constraint_planes:
            n = plane[0]
            x = self.sphere_center
            p = plane[1]
            # n^T ∆x ≥ -n^T (x - p)
            constraints.append(n.T @ self.delta_x >= -n.T @ (x - p) + sphere_radius)
            
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
        while rclpy.ok():
            self.sphere_target_center = self.ros_input
            
            # Enforce virtual fixture on the target position
            constrained_position = self._enforce_virtual_fixture(self.sphere_target_center, self.sphere_radius)
            if np.linalg.norm(constrained_position) > 1e-6:
                damping_factor = 0.9
                constrained_position = damping_factor * constrained_position
                self.sphere_center = self._move_sphere(self.old_sphere_center, constrained_position/np.linalg.norm(constrained_position), np.linalg.norm(constrained_position))
            msg = PoseStamped()
            msg.pose.position.x = self.sphere_center[0]
            msg.pose.position.y = self.sphere_center[1]
            msg.pose.position.z = self.sphere_center[2]
            msg.header.frame_id = "world"
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pose_publisher.publish(msg)
            # Update old positions for the next iteration
            self.old_sphere_target_center = self.sphere_target_center
            self.old_sphere_center = self.sphere_center
            self.iteration += 1
            rclpy.spin_once(self, timeout_sec=0.01)
            

def main(args=None):
    rclpy.init(args=args)
    demo = VirtualFixtureDemo()
    rclpy.spin(demo)
    demo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
