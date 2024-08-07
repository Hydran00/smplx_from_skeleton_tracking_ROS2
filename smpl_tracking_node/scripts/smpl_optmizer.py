import torch
import numpy as np
import open3d as o3d

import torch.nn.functional as F
from utils import LandmarkLoss, MaxMixturePrior, compute_distance_from_pelvis_joint_to_surface
import os
from chamferdist import ChamferDistance
import time 
from utils import block
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

    def optimize_model(self, logger, target_point_cloud, global_orient, global_position, body_pose, landmarks, viz):
        self.viz = viz

        self.target_point_cloud_tensor = self.point_cloud_to_tensor(target_point_cloud).to('cuda:0')
        self.global_orient = global_orient.to('cuda:0').detach().requires_grad_()
        self.global_position = global_position.to('cuda:0').detach().requires_grad_()
        self.body_pose = body_pose.to('cuda:0').detach().requires_grad_()
        self.target_landmarks = landmarks.to('cuda:0')
        
        # Initialize visualizer
        self.target_point_cloud = target_point_cloud
        
        # backup old head and neck position since we will reset them after optimizing (zed body tracking of head and neck is not accurate)
        old_neck_position = self.body_pose[0, 11*3:11*3+3]
        old_head_position = self.body_pose[0, 14*3:14*3+3]
    
        # OPTIMIZING
        output = self.get_smpl()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(output.vertices[0].cpu().detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)
        
             
        self.optimize(logger, params=[self.global_position, self.global_orient], loss_type='transl', num_iterations=200)
        
        self.optimize(logger, params=[self.body_pose], lr=0.001, loss_type='pose', num_iterations=300)
        # reset head position
        # translate the body to the surface
        
        with torch.no_grad():
            offset = - compute_distance_from_pelvis_joint_to_surface(mesh, self.global_position, self.global_orient)
            offset = torch.tensor([offset], dtype=torch.float32, device='cuda:0')
            self.global_position[0, :] =  self.global_position[0, :] + offset
        
        # optimize arms and legs again
        self.arms_leg_idx = [3,4,9,10,17,18,19,20,21,22]
        self.arms_leg_params = torch.cat([self.body_pose[0, idx*3:idx*3+3] for idx in self.arms_leg_idx], dim=0).reshape(1, 30).to('cuda:0').detach().requires_grad_()
        self.optimize(logger, params=[self.body_pose], lr=0.001, loss_type='pose', num_iterations=300)
        
        with torch.no_grad():
            for i, idx in enumerate(self.arms_leg_idx):
                self.body_pose[0, idx*3:idx*3+3] = self.arms_leg_params[0, i*3:i*3+3]
        
            
        self.optimize(logger, params=[self.betas], lr=0.0001, loss_type='shape', num_iterations=200)
        with torch.no_grad():
            self.body_pose[0, 11*3:11*3+3] = old_neck_position
            self.body_pose[0, 14*3:14*3+3] = old_head_position

        # remove gradient tracking
        self.global_position = self.global_position.detach()
        self.global_orient = self.global_orient.detach()
        self.body_pose = self.body_pose.detach()
        self.betas = self.betas.detach()

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
            lmk = landmarks[:24].reshape(1, 72)
            landmark_loss = self.landmark_loss(lmk, self.target_landmarks)
            data_loss = self.chamfer_distance(self.target_point_cloud_tensor.unsqueeze(0), generated_point_cloud_tensor.unsqueeze(0), reverse=True)
            prior_loss = self.prior.forward(self.body_pose, self.betas)
            return 0.1 * data_loss + landmark_loss + 0.01 * prior_loss
        
        elif type == "shape":
            prior_loss = self.prior.forward(self.body_pose, self.betas)
            beta_loss = (self.betas**2).mean()
            data_loss = self.chamfer_distance(generated_point_cloud_tensor.unsqueeze(0), self.target_point_cloud_tensor.unsqueeze(0), reverse=True)
            return 0.1 * prior_loss + 0.01 * beta_loss + 1 * data_loss

    def get_smpl(self):
        return self.smpl_model(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                transl=self.global_position,
                return_verts=True
            )

    def optimize(self, logger, params=[], lr=0.01, loss_type='all', num_iterations=1000, landmarks=None, viz=None):
        optimizer = torch.optim.Adam(params, lr)
        for i in range(num_iterations):
            optimizer.zero_grad()
            output = self.get_smpl()
            vertices = output.vertices[0]
            joints = output.joints[0]
            landmarks = joints
            self.logger = logger
            
            
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().detach().numpy())
            self.mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)

            loss = self.compute_loss(loss_type, vertices, landmarks)
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                logger.info(f"Iteration {i}: Loss = {loss.item()}")

            if i == 0:
                self.viz.add_geometry(self.mesh)
            else:
                self.viz.update_geometry(self.mesh)

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
            # while time.time() - start_time < 0.5:
            #     self.viz.poll_events()
            #     self.viz.update_renderer()
        self.viz.remove_geometry(self.mesh)
        for sphere in self.sphere_list:
            self.viz.remove_geometry(sphere)
        for sphere in self.target_sphere_list:
            self.viz.remove_geometry(sphere)
        self.sphere_list = []
        self.target_sphere_list = []
        
        return self.betas, self.body_pose
    
