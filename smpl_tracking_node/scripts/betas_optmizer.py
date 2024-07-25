import torch
from torch import nn
import numpy as np
import open3d as o3d
from tqdm import tqdm
import sys, os, pickle

class ChamferDistance(nn.Module):
    def forward(self, pc1, pc2):
        # Ensure both tensors are on the same device
        device = pc1.device
        pc2 = pc2.to(device)
        
        # Compute pairwise squared distances
        diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)
        dist = torch.sum(diff ** 2, dim=-1)
        
        # Compute Chamfer Distance
        dist1 = torch.min(dist, dim=2)[0]
        dist2 = torch.min(dist, dim=1)[0]
        
        return torch.mean(dist1) + torch.mean(dist2)
# PRIOR LOSS - FROM SMPLify paper
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
        self.learning_rate = learning_rate
        self.num_betas = num_betas

        # Ensure betas is a leaf tensor
        self.betas = torch.zeros((1, num_betas), dtype=torch.float32, requires_grad=True, device='cuda:0')
        
        # Define optimizer
        self.optimizer = torch.optim.Adam([self.betas], lr=self.learning_rate)
        self.chamfer_distance = ChamferDistance().to('cuda:0')
        self.prior = MaxMixturePrior(prior_folder=os.path.expanduser('~')+'/models/prior', num_gaussians=8).to('cuda:0')
        
    def point_cloud_to_tensor(self, point_cloud):
        # Convert Open3D PointCloud to PyTorch tensor
        points = np.asarray(point_cloud.points)
        return torch.tensor(points, dtype=torch.float32, device='cuda:0')


    def compute_loss(self, generated_point_cloud, target_point_cloud):
        data_loss = self.chamfer_distance(generated_point_cloud.unsqueeze(0), self.target_point_cloud_tensor.unsqueeze(0))
        prior_loss = self.prior.forward(self.body_pose, self.betas)
        beta_loss = (self.betas**2).mean()
        tot_loss = data_loss + prior_loss
        return tot_loss + 0.1 * beta_loss + 10 * prior_loss

    def optimize(self, logger, target_point_cloud, global_orient, global_position, body_pose, num_iterations=1000):
        self.target_point_cloud_tensor = self.point_cloud_to_tensor(target_point_cloud)
        self.global_orient = global_orient.to('cuda:0')
        self.global_position = global_position.to('cuda:0')
        self.body_pose = body_pose.to('cuda:0')

        # init visualizer
        self.target_point_cloud = target_point_cloud
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.mesh_show_wireframe = True
        self.mesh = o3d.geometry.TriangleMesh()
        self.viz.add_geometry(self.target_point_cloud)

        for i in tqdm(range(num_iterations)):
            self.optimizer.zero_grad()
            
            output = self.smpl_model(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                transl=self.global_position,
                return_verts=True
            )
            self.mesh.vertices = o3d.utility.Vector3dVector(output.vertices[0].cpu().detach().numpy())
            self.mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)
            if(i==0):
                self.viz.add_geometry(self.mesh)
            else:
                self.viz.update_geometry(self.mesh)
            self.viz.poll_events()
            self.viz.update_renderer()
            
            generated_point_cloud = output.vertices[0].cpu()  # Move to CPU for Chamfer distance calculation
            
            # Move tensors to GPU before calculating loss
            generated_point_cloud = generated_point_cloud.to('cuda:0')
            self.target_point_cloud_tensor = self.target_point_cloud_tensor.to('cuda:0')

            loss = self.compute_loss(generated_point_cloud, self.target_point_cloud_tensor)
            
            loss.backward()
            # print gradients
            logger.info(str(self.betas.grad))
            
            self.optimizer.step()
            if i % 50 == 0:
                logger.info(f"Iteration {i}: Loss = {loss.item()}")
            # tqdm.write(f"Iteration {i}: Loss = {loss.item()}")   
        # Close the visualizer
        self.viz.destroy_window()
        
        
        return self.betas

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
