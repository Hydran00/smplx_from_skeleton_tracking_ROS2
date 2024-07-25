import torch
from torch import nn
import numpy as np
import open3d as o3d
from tqdm import tqdm
import sys, os, pickle
import time
import torch.nn.functional as F
from chamferdist import ChamferDistance

class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()

    def forward(self,scan_landmarks,template_landmarks):
        return torch.sum((scan_landmarks - template_landmarks)**2)

# FROM SMPLify paper
class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=torch.float32, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)

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
        
        self.viz = o3d.visualization.Visualizer()

    def optimize_model(self, logger, target_point_cloud, global_orient, global_position, body_pose, landmarks):

        self.target_point_cloud_tensor = self.point_cloud_to_tensor(target_point_cloud).to('cuda:0')
        self.global_orient = global_orient.to('cuda:0').detach().requires_grad_()
        self.global_position = global_position.to('cuda:0').detach().requires_grad_()
        self.body_pose = body_pose.to('cuda:0').detach().requires_grad_()
        self.target_landmarks = landmarks.to('cuda:0')
        

        # Initialize visualizer
        self.target_point_cloud = target_point_cloud
        
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.mesh_show_wireframe = True
        
        self.viz.add_geometry(self.target_point_cloud)

        # OPTIMIZING 
        self.optimize(logger, params=[self.global_position, self.global_orient], loss_type='transl', num_iterations=500)
        self.optimize(logger, params=[self.body_pose], lr=0.001, loss_type='pose', num_iterations=500)
        self.optimize(logger, params=[self.betas], lr=0.001, loss_type='shape', num_iterations=50)

        
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
            # landmark_loss = self.landmark_loss(landmarks, self.target_landmarks)
            return data_loss
        
        if type == "pose":
            landmark_loss = self.landmark_loss(landmarks, self.target_landmarks)
            prior_loss = self.prior.forward(self.body_pose, self.betas)
            return 0.1 * landmark_loss #+ 0.01 * prior_loss
        
        elif type == "shape":
            prior_loss = self.prior.forward(self.body_pose, self.betas)
            beta_loss = (self.betas**2).mean()
            data_loss = self.chamfer_distance(generated_point_cloud_tensor.unsqueeze(0), self.target_point_cloud_tensor.unsqueeze(0), reverse=True)
            return 1 * prior_loss + 1 * beta_loss + 1 * data_loss


    def optimize(self, logger, params=[], lr=0.01, loss_type='all', num_iterations=1000, landmarks=None):
        """
        Optimizes the parameters of the model

        :param params: (list) list of parameters to optimize
        """

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
            landmarks = output.joints.to('cuda:0')  # Move landmarks to GPU
            landmarks = landmarks[:, :24, :].reshape(1, 72)
            
            self.mesh.vertices = o3d.utility.Vector3dVector(output.vertices[0].cpu().detach().numpy())
            self.mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)
            
            
            loss = self.compute_loss(loss_type, output.vertices[0].cpu(), landmarks)
            loss.backward()
            optimizer.step()
            logger.info(f"Iteration {i}: Loss = {loss.item()}")
            if i % 50 == 0:
                logger.info(f"Iteration {i}: Loss = {loss.item()}")

            if i == 0:
                self.viz.add_geometry(self.mesh)
            else:
                self.viz.update_geometry(self.mesh)

            # create a sphere for each landmark
            for j in range(24):
                if i == 0:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)

                    sphere.compute_vertex_normals()
                    sphere_target.compute_vertex_normals()

                    random_color = np.random.rand(3)

                    sphere.paint_uniform_color(random_color)
                    sphere_target.paint_uniform_color(random_color)

                    position = landmarks[0, j*3:j*3+3].cpu().detach().numpy()
                    position_target = self.target_landmarks[0, j*3:j*3+3].cpu().detach().numpy()

                    sphere.translate(position, relative=False)
                    sphere_target.translate(position_target, relative=False)

                    self.viz.add_geometry(sphere)
                    self.sphere_list.append(sphere)
                    self.viz.add_geometry(sphere_target)
                    self.target_sphere_list.append(sphere_target)
                else:
                    self.sphere_list[j].translate(landmarks[0, j*3:j*3+3].cpu().detach().numpy(), relative=False)
                    self.viz.update_geometry(self.sphere_list[j])
                    self.viz.update_geometry(self.target_sphere_list[j])
            
            self.viz.poll_events()
            self.viz.update_renderer()
        
        self.viz.clear_geometries()
        self.sphere_list = []
        self.target_sphere_list = []
        
        return self.betas, self.body_pose



# Example usage
# if __name__ == "__main__":
#     smpl_model = YourSMPLModelClass()  # Initialize your SMPL model here
#     target_point_cloud = o3d.geometry.PointCloud()  # Load your target point cloud here
#     target_point_cloud.points = o3d.utility.Vector3dVector(np.load("target_point_cloud.npy"))  # Load your point cloud points

#     optimizer = SMPLModelOptimizer(smpl_model)
#     global_orient = torch.zeros((1, 3), dtype=torch.float32).to('cuda:0')
#     global_position = torch.zeros((1, 3), dtype=torch.float32).to('cuda:0')
#     body_pose = torch.zeros((1, 69), dtype=torch.float32).to('cuda:0')

#     optimized_betas = optimizer.optimize(target_point_cloud, global_orient, global_position, body_pose)

#     print("Optimized Betas:", optimized_betas)
