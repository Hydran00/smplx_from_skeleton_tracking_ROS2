import torch
from torch import nn
import numpy as np
import open3d as o3d

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

    def point_cloud_to_tensor(self, point_cloud):
        # Convert Open3D PointCloud to PyTorch tensor
        points = np.asarray(point_cloud.points)
        return torch.tensor(points, dtype=torch.float32, device='cuda:0')

    def optimize(self, target_point_cloud, global_orient, global_position, body_pose, num_iterations=1000):
        self.target_point_cloud = self.point_cloud_to_tensor(target_point_cloud)
        self.global_orient = global_orient.to('cuda:0')
        self.global_position = global_position.to('cuda:0')
        self.body_pose = body_pose.to('cuda:0')

        for i in range(num_iterations):
            self.optimizer.zero_grad()
            
            output = self.smpl_model(
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                transl=self.global_position,
                return_verts=True
            )

            generated_point_cloud = output.vertices[0].cpu()  # Move to CPU for Chamfer distance calculation
            
            # Move tensors to GPU before calculating loss
            generated_point_cloud = generated_point_cloud.to('cuda:0')
            self.target_point_cloud = self.target_point_cloud.to('cuda:0')

            loss = self.chamfer_distance(generated_point_cloud.unsqueeze(0), self.target_point_cloud.unsqueeze(0))
            
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss.item()}")
                
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
