#!/usr/bin/env python3

import open3d as o3d
import os

from materials import mat_sphere_transparent, mat_skin

from ament_index_python.packages import get_package_share_directory
from utils import *
import time
POINT_RADIUS = 0.01
class TestVF:
    def __init__(self):

        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True
        opt.mesh_show_wireframe = True
        
        
        load_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skin.obj"
        mesh =o3d.io.read_triangle_mesh(load_path)

        # pcd = compute_torax_projection(mesh)
        # self.pcd = o3d.io.read_point_cloud(os.path.expanduser('~')+"/ros2_ws/pcd.ply")
        
        points_num = 200
        self.pcd = get_flat_surface_point_cloud(points_num,0.02)
        # self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # orient_normals_towards_camera_location
        # self.pcd.orient_normals_towards_camera_location()
        self.points = np.asarray(self.pcd.points)
        self.viz.add_geometry(self.pcd)
        self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
        self.viz.add_geometry(self.reference_frame)
        
        print("Loaded pcd with ", len(self.pcd.points), " points")
        
        self.point_sphere_list = []
        for i in range(len(self.pcd.points)):
            # create a sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=POINT_RADIUS)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([1, 1, 1])
            sphere.translate(self.pcd.points[i], relative=False)
            self.viz.add_geometry(sphere)
        
        # check if the pcd contains nans
        # start = mesh.get_center() + (0.05,0.2,-0.16)
        start = np.array([0.0,0.0,0.005])
        end = np.array([0.2,0.2,-0.005])
        
        print("Starting point is ", start)
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        self.sphere.compute_vertex_normals()
        self.sphere.paint_uniform_color([1, 0, 0])
        self.sphere.translate(start, relative=False)
        self.viz.add_geometry(self.sphere)
        traj = self.linear_interpolation_movement(start, end)
        # draw trajectory
        start = traj[0]
        end = traj[-1]
        self.line = o3d.geometry.LineSet()
        self.line.points = o3d.utility.Vector3dVector([start, end])
        self.line.lines = o3d.utility.Vector2iVector([[0, 1]])
        self.line.colors = o3d.utility.Vector3dVector([[0,0,0]])
        self.viz.add_geometry(self.line)
        
        
        self.sphere_closest = o3d.geometry.TriangleMesh.create_sphere(radius=POINT_RADIUS+0.00001)
        self.sphere_closest.compute_vertex_normals()
        self.sphere_closest.paint_uniform_color([0, 0, 1])
        self.sphere_closest.translate(start, relative=False)
        self.viz.add_geometry(self.sphere_closest)
        
        
        self.sphere_desired = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        self.sphere_desired.compute_vertex_normals()
        self.sphere_desired.paint_uniform_color([0, 1, 0])
        self.sphere_desired.translate(start, relative=False)
        self.viz.add_geometry(self.sphere_desired)
        
        self.vf_position = traj[0]
        self.vf_position_last = traj[0]
        
        # self.block()
        # self.current_desired = traj[0]

        self.simulate_motion(traj)
        
    def block(self):
        while True:
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.03)
        
    def linear_interpolation_movement(self,start, distance):
        """
        Linear interpolation of a movement
        """
        traj = []
        steps = 2000
        for i in range(steps):
            traj.append(start + i*distance/steps)
        return traj
    
    def simulate_motion(self,traj):
        """
        Simulate the motion of the sphere
        """
        for point in traj:
            transl = self.computeVF(point, POINT_RADIUS)
            self.sphere.translate(transl, relative=False)
            
            self.viz.update_geometry(self.sphere)
            self.viz.update_geometry(self.sphere_closest)
            self.viz.update_geometry(self.sphere_desired)
            self.viz.poll_events()
            self.viz.update_renderer()
            time.sleep(0.01)
            
            self.vf_position_last = self.vf_position
    
    def computeVF(self,point, point_radius=0.01):
        """
        Consider each point of the cloud as a sphere, find the closest and check the point is inside the sphere
        If it is, then project the point to the sphere
        """
        
        #find the closest point
        distances = np.linalg.norm(self.points-self.vf_position, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = self.points[closest_idx,:]
        # update closest sphere
        self.sphere_closest.translate(closest_point, relative=False)
        # self.sphere_desired.translate(self.current_desired, relative=False)
        # check if the point is inside the sphere
        distance = np.linalg.norm(closest_point - point)
        
        set_point = None
        if distance < point_radius:
            # project the point to the sphere
            direction = self.vf_position - closest_point
            
            # check if the direction is similar to the normal of that point
            if np.dot(direction, [0,0,1]) < 0:
                direction = -direction
            
            direction = direction/np.linalg.norm(direction)
            projected_point = closest_point + direction*point_radius
            set_point = projected_point
            
            # apply parallel component
            parallel_component = np.dot(point - projected_point, direction)*direction
            set_point = projected_point + parallel_component
            
        else:
            set_point = point
        
        
        # self.current_desired += (set_point - self.current_desired)*0.1
        return set_point
        
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    test = TestVF()
