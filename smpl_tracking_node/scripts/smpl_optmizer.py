import torch
import numpy as np
import open3d as o3d

import torch.nn.functional as F
from utils import LandmarkLoss, MaxMixturePrior
import os
from chamferdist import ChamferDistance
import time 
from utils import block
import scipy
class SMPLModelOptimizer:
    def __init__(self, smpl_model, learning_rate=0.01, num_betas=10):
        self.smpl_model = smpl_model  # Assumes smpl_model is an instance of a pre-defined SMPL model class
        self.num_betas = num_betas

        # Ensure betas is a leaf tensor
        self.betas = torch.zeros((1, num_betas), dtype=torch.float32, requires_grad=True, device='cuda:0')
        self.body_pose = torch.zeros((1, 66), dtype=torch.float32, device='cuda:0', requires_grad=True)

        self.chamfer_distance = ChamferDistance().to('cuda:0')
        self.prior = MaxMixturePrior(prior_folder=os.path.expanduser('~')+'/SKEL_WS/SKEL/models/prior', num_gaussians=8).to('cuda:0')
        self.landmark_loss = LandmarkLoss().to('cuda:0')
        self.mesh = o3d.geometry.TriangleMesh()
        self.sphere_list = []
        self.target_sphere_list = []
        
        # self.viz = o3d.visualization.Visualizer()
    def get_mask_upper_body(self):
        mask = torch.ones((1,72), device='cuda:0')
        # mask neck
        # mask[0, 12*3:12*3+3] = 0
        # # mask head
        # mask[0, 13*3:13*3+3] = 0
        # mask[0, 15*3:15*3+3] = 0
        # mask legs [4,5,7,8,10,11]
        mask[0, 1*3:1*3+3] = 0
        mask[0, 2*3:2*3+3] = 0
        mask[0, 4*3:4*3+3] = 0
        mask[0, 5*3:5*3+3] = 0
        mask[0, 7*3:7*3+3] = 0
        mask[0, 8*3:8*3+3] = 0
        mask[0, 10*3:10*3+3] = 0
        mask[0, 11*3:11*3+3] = 0
        return mask

    def get_mask_torax(self):
        # mask neck, head, legs, arms
        idx = [4,5,8,7,10,11,12,13,16,17,18,19,20,21,22,23]
        mask = torch.ones((1,72), device='cuda:0')
        for i in idx:
            mask[0, i*3:i*3+3] = 0
        return mask

    def optimize_model(self, logger, target_point_cloud, global_orient, global_position, body_pose, landmarks, viz, optimize_legs=False):
        """
        target_point_cloud is the point cloud of the target object to be optimized to
        global_orient: Global orientation of the smpl model
        global_position: Global position of the smpl model
        body_pose: Pose of the smpl model
        landmarks: Landmarks detected by skeleton tracker
        viz: Open3D visualizer object
        optimize_legs: If True, optimize legs. If False, keep legs fixed
        """
        self.viz = viz

        self.target_point_cloud_tensor = self.point_cloud_to_tensor(target_point_cloud).to('cuda:0')
        self.global_orient = global_orient.to('cuda:0').detach().requires_grad_()
        self.global_position = global_position.to('cuda:0').detach().requires_grad_()
        self.body_pose = body_pose.to('cuda:0').detach().requires_grad_()
        self.target_landmarks = landmarks.to('cuda:0')
        
        # Initialize visualizer
        self.target_point_cloud = target_point_cloud

        # Get mask for upper body
        if optimize_legs:
            self.mask = torch.ones((1,72), device='cuda:0')
        else:
            self.mask = self.get_mask_upper_body()

        # OPTIMIZING
        output = self.get_smpl()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(output.vertices[0].cpu().detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)
        # viz.add_geometry(mesh)
        # viz.poll_events()
        # viz.update_renderer()
        # block(viz)        
             
        self.optimize(logger, params=[self.global_position], loss_type='transl', num_iterations=100)

        # block(viz)
        
        # with torch.no_grad():
        #     offset = - compute_distance_from_pelvis_joint_to_surface(self.mesh, self.global_position, self.global_orient)
        #     if np.any(np.isinf(offset)):
        #         logger.info("Offset is infinite. Skipping optimization")
        #         offset = torch.tensor([np.array([0,0,0])], dtype=torch.float32, device='cuda:0')
        #     else:
        #         offset = torch.tensor([np.array(offset)], dtype=torch.float32, device='cuda:0')
        #     self.global_position[0, :] =  self.global_position[0, :] + offset
        # self.optimize(logger, params=[self.body_pose], lr=0.001, loss_type='pose', num_iterations=200)
        
        # translate the body to the surface

        # # optimize arms and legs again
        self.optimize(logger, params=[self.body_pose,self.betas], lr=0.001, loss_type='pose', num_iterations=1000)

        # self.optimize(logger, params=[self.betas], lr=0.01, loss_type='shape', num_iterations=200)
        
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
            # apply mask
            lmk_masked = lmk * self.mask
            target_lmk_masked = self.target_landmarks * self.mask
            landmark_loss = self.landmark_loss(lmk_masked, target_lmk_masked)
            # data_loss = self.chamfer_distance(self.target_point_cloud_tensor.unsqueeze(0), generated_point_cloud_tensor.unsqueeze(0), reverse=True)
            body_pose_masked = self.body_pose * self.mask[:, 3:]
            prior_loss = self.prior.forward(body_pose_masked, self.betas)
            # return 0.1 * data_loss + landmark_loss + 0.01 * prior_loss
            self.logger.info(f"Landmark loss: {landmark_loss}")

            beta_loss = (self.betas**2).mean()

            return 10 * landmark_loss + 0.01 * prior_loss + 10 * beta_loss
        
        elif type == "shape":
            # apply mask
            body_pose_masked = self.body_pose * self.mask[:, 3:]
            prior_loss = self.prior.forward(body_pose_masked, self.betas)
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
        logger.info(f"Optimizing {loss_type}...")
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
            
            ####################
            # RAYCAST LANDMARKS TO GET SURFACE POINTS
            humanoid = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(humanoid)

            # use open3d ray casting to compute distance from pelvis joint to surface
            # direction is the third column of the global rotation matrix
            rotm = scipy.spatial.transform.Rotation.from_rotvec(self.global_orient.cpu().detach().numpy())
            direction = rotm.as_matrix()[:, 2][0]
            
            ray_list = []
            for h in range(24):
                start = landmarks[h].cpu().detach().numpy()
                ray_list.append([*start, *direction])

            # start = self.global_position.cpu().detach().numpy()
            # ray_list.append([*start, *direction])

            rays = o3d.core.Tensor(ray_list, dtype=o3d.core.Dtype.Float32)

            ans = scene.cast_rays(rays)

            offsets = ans['t_hit']
            with torch.no_grad():
                for h in range(24):
                    offset = offsets[h].cpu().numpy()
                    landmarks[h][0] = ray_list[h][0] + offset * direction[0]
                    landmarks[h][1] = ray_list[h][1] + offset * direction[1]
                    landmarks[h][2] = ray_list[h][2] + offset * direction[2]
                # offset = offsets[24].cpu().numpy()
                # self.global_position[0, 0] = ray_list[24][0] + offset * direction[0]
                # self.global_position[0, 1] = ray_list[24][1] + offset * direction[1]
                # self.global_position[0, 2] = ray_list[24][2] + offset * direction[2]
            ####################
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
    
