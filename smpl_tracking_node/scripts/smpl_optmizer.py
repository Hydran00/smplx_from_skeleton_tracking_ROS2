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
from ament_index_python.packages import get_package_share_directory

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
        self.upper_body_idx_path = get_package_share_directory("virtual_fixture")+ '/skel_regions/upper_body_frontal.txt'
        upper_body_faces = []
        self.upper_body_vertices = set()
        with open(self.upper_body_idx_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                vars = line.split()
                self.upper_body_vertices.add(int(vars[1]))
                self.upper_body_vertices.add(int(vars[2]))
                self.upper_body_vertices.add(int(vars[3]))
        self.upper_body_vertices = list(self.upper_body_vertices)

        # self.viz = o3d.visualization.Visualizer()
    def get_mask_upper_body(self):
        mask = torch.ones((1,69), device='cuda:0')
        with torch.no_grad():
            mask_idx = [0,1,3,4,6,7,9,10]
            for i in mask_idx:
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
        # Initialize visualizer
        self.viz = viz
        # Get mask for upper body
        if optimize_legs:
            self.mask = torch.ones((1,69), device='cuda:0')
        else:
            self.mask = self.get_mask_upper_body()

        self.target_point_cloud_tensor = self.point_cloud_to_tensor(target_point_cloud).to('cuda:0')
        self.global_orient = global_orient.to('cuda:0').detach().requires_grad_()
        self.global_position = global_position.to('cuda:0').detach().requires_grad_()
        self.body_pose = body_pose.to('cuda:0').detach()
        self.body_pose  = (self.body_pose * self.mask).requires_grad_()

        self.target_landmarks = landmarks.to('cuda:0')
        

        self.optimize(logger, params=[self.global_position], lr=0.005, loss_type='transl', num_iterations=100)
        self.optimize(logger, params=[self.body_pose], lr=0.001, loss_type='pose', num_iterations=300)
        self.optimize(logger, params=[self.betas], lr=0.002, loss_type='shape', num_iterations=200)
        
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

    def filter_upper_body(self, points):
        return points[self.upper_body_vertices]

    def compute_loss(self, type, generated_point_cloud, landmarks=None):
        generated_point_cloud_tensor = self.point_cloud_to_tensor(generated_point_cloud).to('cuda:0')
        generated_point_cloud_tensor_upper_body = self.filter_upper_body(generated_point_cloud_tensor)
        
        if type == "transl":
            data_loss = self.chamfer_distance(self.target_point_cloud_tensor.unsqueeze(0), generated_point_cloud_tensor_upper_body.unsqueeze(0), reverse=True)
            return data_loss
        
        if type == "pose":
            lmk = landmarks[1:24].reshape(1, 69)
            # apply mask
            lmk_masked = lmk * self.mask
            target_lmk_masked = self.target_landmarks[0][3:72] * self.mask
            landmark_loss = self.landmark_loss(lmk_masked, target_lmk_masked)
            data_loss = self.chamfer_distance(self.target_point_cloud_tensor.unsqueeze(0), generated_point_cloud_tensor.unsqueeze(0), reverse=True)
            body_pose_masked= self.body_pose * self.mask
            prior_loss = self.prior.forward(body_pose_masked, self.betas)
            return 10 * landmark_loss + 0.5 * data_loss + 0.5 * prior_loss
        
        elif type == "shape":
            # apply mask
            body_pose_masked = self.body_pose * self.mask
            prior_loss = self.prior.forward(body_pose_masked, self.betas)
            beta_loss = (self.betas**2).mean()
            data_loss = self.chamfer_distance(generated_point_cloud_tensor_upper_body.unsqueeze(0), self.target_point_cloud_tensor.unsqueeze(0), reverse=True)
            return 0.5 * prior_loss + 1 * data_loss + 0.5 * beta_loss

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
                    
                    sphere.paint_uniform_color([1,0,0])
                    sphere_target.paint_uniform_color([0,1,0])
                    
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
    
