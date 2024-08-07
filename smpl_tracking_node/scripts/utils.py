import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from chamferdist import ChamferDistance
import sys, os, pickle
import scipy
import time
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


def compute_distance_from_pelvis_joint_to_surface(human_mesh, global_position, global_orient):
    # SUGGESTED BY https://github.com/matteodv99tn
    # define scene
    humanoid = o3d.t.geometry.TriangleMesh.from_legacy(human_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(humanoid)
    
    # use open3d ray casting to compute distance from pelvis joint to surface
    start = global_position[0].cpu().detach().numpy()
    
    # direction is the third column of the global rotation matrix
    rotm = scipy.spatial.transform.Rotation.from_rotvec(global_orient.cpu().detach().numpy())
    direction = rotm.as_matrix()[:, 2][0]

    ray = o3d.core.Tensor([ [*start, *direction]], dtype=o3d.core.Dtype.Float32)
    
    # logger.info(f"start: {start}")
    # logger.info(f"direction: {direction}")
    
    ans = scene.cast_rays(ray)
    
    # Visualize
    # length = 2.0
    # end = start + length * direction
    # points = [start, end]
    # lines = [[0, 1]]
    # colors = [[1, 0, 0]]  # Red color for the line

    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(points)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector(colors)

    # # Step 4: Visualize the LineSet
    # if it==0:
    #     viz.add_geometry(line_set)
    # else:
    #     viz.update_geometry(line_set)
    
    
    # self.logger.info(f"Distance from pelvis joint to surface: {ans}")
    # self.logger.info(f"Vertex ID: {ans['primitive_ids']}")
    
    offset = ans['t_hit'][0].cpu().numpy()
    # block(viz)
    return offset * direction


def block(viz):
    while True:
        viz.poll_events()
        viz.update_renderer()
        time.sleep(0.03)
