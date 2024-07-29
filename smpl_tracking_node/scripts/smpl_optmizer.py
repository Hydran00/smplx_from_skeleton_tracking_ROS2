import torch
import numpy as np
import open3d as o3d

import torch.nn.functional as F
from losses import LandmarkLoss, MaxMixturePrior
import os
from chamferdist import ChamferDistance
import time 
import scipy
class SMPLModelOptimizer:
    def __init__(self, smpl_model, learning_rate=0.01, num_betas=10):
        self.smpl_model = smpl_model  # Assumes smpl_model is an instance of a pre-defined SMPL model class
        self.num_betas = num_betas

        # Ensure betas is a leaf tensor
        self.betas = torch.zeros((1, num_betas), dtype=torch.float32, requires_grad=True, device='cuda:0')
        self.body_pose = torch.zeros((1, 66), dtype=torch.float32, device='cuda:0', requires_grad=True)

        self.chamfer_distance = ChamferDistance().to('cuda:0')
        self.prior = MaxMixturePrior(prior_folder=os.path.expanduser('~')+'/models/prior', num_gaussians=8).to('cuda:0')
        self.landmark_loss = LandmarkLoss().to('cuda:0')
        self.mesh = o3d.geometry.TriangleMesh()
        self.sphere_list = []
        self.target_sphere_list = []
        
        # self.viz = o3d.visualization.Visualizer()

    def optimize_model(self, logger, target_point_cloud, global_orient, global_position, body_pose, landmarks,viz):
        self.viz = viz

        self.target_point_cloud_tensor = self.point_cloud_to_tensor(target_point_cloud).to('cuda:0')
        self.global_orient = global_orient.to('cuda:0').detach().requires_grad_()
        self.global_position = global_position.to('cuda:0').detach().requires_grad_()
        self.body_pose = body_pose.to('cuda:0').detach().requires_grad_()
        self.target_landmarks = landmarks.to('cuda:0')
        
        logger.info(f"Body pose shape: {self.body_pose.shape}")
        # Initialize visualizer
        self.target_point_cloud = target_point_cloud
        
        old_neck_position = self.body_pose[0, 11*3:11*3+3]
    
        # OPTIMIZING 
        self.optimize(logger, params=[self.global_position, self.global_orient], loss_type='transl', num_iterations=200)
        
        self.optimize(logger, params=[self.body_pose], lr=0.001, loss_type='pose', num_iterations=200)
        # reset head position
        # translate the body to the surface
        with torch.no_grad():
            offset = -self.compute_distance_from_pelvis_joint_to_surface(0)
            offset = torch.tensor([offset], dtype=torch.float32, device='cuda:0')
            self.global_position[0, :] =  self.global_position[0, :] + offset
        
        self.optimize(logger, params=[self.betas], lr=0.0001, loss_type='shape', num_iterations=200)

        with torch.no_grad():
            self.body_pose[0, 11*3:11*3+3] = old_neck_position
            
   
        return self.betas, self.body_pose, self.global_position
    
    def point_cloud_to_tensor(self, point_cloud):
        if isinstance(point_cloud, torch.Tensor):
            return point_cloud
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            points = np.asarray(point_cloud.points)
            return torch.tensor(points, dtype=torch.float32)
        else:
            raise TypeError("Unsupported point cloud format")



    def compute_loss(self, type, generated_point_cloud, landmarks=None):
        generated_point_cloud_tensor = self.point_cloud_to_tensor(generated_point_cloud).to('cuda:0')
        
        if type == "transl":
            data_loss = self.chamfer_distance(self.target_point_cloud_tensor.unsqueeze(0), generated_point_cloud_tensor.unsqueeze(0), reverse=True)
            return data_loss
        
        if type == "pose":
            self.logger.info(f"Landmarks shape: {landmarks.shape}") # 45 x 3
            self.logger.info(f"Target Landmarks shape: {self.target_landmarks.shape}") # 1 x 72
            lmk = landmarks[:24].reshape(1, 72)
            landmark_loss = self.landmark_loss(lmk, self.target_landmarks)
            prior_loss = self.prior.forward(self.body_pose, self.betas)
            return 0.1 * landmark_loss #+ 0.01 * prior_loss
        
        elif type == "shape":
            prior_loss = self.prior.forward(self.body_pose, self.betas)
            beta_loss = (self.betas**2).mean()
            data_loss = self.chamfer_distance(generated_point_cloud_tensor.unsqueeze(0), self.target_point_cloud_tensor.unsqueeze(0), reverse=True)
            return 0.1 * prior_loss + 0.01 * beta_loss + 1 * data_loss


    def optimize(self, logger, params=[], lr=0.01, loss_type='all', num_iterations=1000, landmarks=None, viz=None):
        optimizer = torch.optim.Adam(params, lr)
        for i in range(num_iterations):
            optimizer.zero_grad()
            output = self.smpl_model(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                transl=self.global_position,
                return_verts=True
            )
            vertices = output.vertices[0]
            joints = output.joints[0]
            landmarks = joints
            self.logger = logger
            
            logger.info(f"joint shape: {joints.shape}")
            logger.info(f"vertices shape: {vertices.shape}")
            
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().detach().numpy())
            self.mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)

            loss = self.compute_loss(loss_type, vertices, landmarks)
            loss.backward()
            optimizer.step()
            logger.info(f"Iteration {i}: Loss = {loss.item()}")
            if i % 50 == 0:
                logger.info(f"Iteration {i}: Loss = {loss.item()}")

            # if i == 0:
            #     self.viz.add_geometry(self.mesh)
            # else:
            #     self.viz.update_geometry(self.mesh)

            for j in range(24):
                if i == 0:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    
                    sphere.compute_vertex_normals()
                    sphere_target.compute_vertex_normals()
                    
                    random_color = np.random.rand(3)
                    
                    sphere.paint_uniform_color(random_color)
                    sphere_target.paint_uniform_color(random_color)
                    
                    position = landmarks[j].cpu().detach().numpy()
                    position_target = self.target_landmarks[0, j*3:j*3+3].cpu().detach().numpy()
                    
                    sphere.translate(position, relative=False)
                    sphere_target.translate(position_target, relative=False)
                    
                    self.viz.add_geometry(sphere)
                    self.sphere_list.append(sphere)
                    self.viz.add_geometry(sphere_target)
                    self.target_sphere_list.append(sphere_target)
                else:
                    self.sphere_list[j].translate(landmarks[j].cpu().detach().numpy(), relative=False)
                    self.viz.update_geometry(self.sphere_list[j])
                    self.target_sphere_list[j].translate(self.target_landmarks[0, j*3:j*3+3].cpu().detach().numpy(), relative=False)
                    self.viz.update_geometry(self.target_sphere_list[j])
            # time.sleep(0.1)
            self.viz.poll_events()
            self.viz.update_renderer()
            start_time = time.time()
            while time.time() - start_time < 3:
                self.viz.poll_events()
                self.viz.update_renderer()
        self.viz.remove_geometry(self.mesh)
        for sphere in self.sphere_list:
            self.viz.remove_geometry(sphere)
        for sphere in self.target_sphere_list:
            self.viz.remove_geometry(sphere)
        self.sphere_list = []
        self.target_sphere_list = []
        
        return self.betas, self.body_pose
    
    def compute_distance_from_pelvis_joint_to_surface(self, it):
        # SPONSORED BY https://github.com/matteodv99tn
        # define scene
        humanoid = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(humanoid)
        
        # use open3d ray casting to compute distance from pelvis joint to surface
        start = self.global_position[0].cpu().detach().numpy()
        
        # direction is the third column of the global rotation matrix
        rotm = scipy.spatial.transform.Rotation.from_rotvec(self.global_orient.cpu().detach().numpy())
        direction = rotm.as_matrix()[:, 2][0]
        
        self.logger.info(f"start: {start}")
        self.logger.info(f"direction: {direction}")
        
        ray = o3d.core.Tensor([ [*start, *direction]], dtype=o3d.core.Dtype.Float32)
        
        ans = scene.cast_rays(ray)
        
        
        # Visualize
        length = 2.0
        end = start + length * direction
        points = [start, end]
        lines = [[0, 1]]
        colors = [[1, 0, 0]]  # Red color for the line

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Step 4: Visualize the LineSet
        if it==0:
            self.viz.add_geometry(line_set)
        else:
            self.viz.update_geometry(line_set)
        
        self.logger.info(f"Distance from pelvis joint to surface: {ans}")
        self.logger.info(f"Vertex ID: {ans['primitive_ids']}")
        

        offset = ans['t_hit'][0].cpu().numpy()
        return offset * direction