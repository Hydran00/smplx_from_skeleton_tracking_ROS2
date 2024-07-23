import numpy as np
from scipy.spatial import cKDTree

class SMPLModelOptimizer:
    def __init__(self, smpl_model, target_point_cloud, learning_rate=0.01, num_betas=10):
        self.smpl_model = smpl_model  # Assumes smpl_model is an instance of a pre-defined SMPL model class
        self.learning_rate = learning_rate
        self.betas = np.zeros(num_betas)  # Initialize betas to zeros
        self.num_betas = num_betas

    def chamfer_distance(self, pc1, pc2):
        kdtree1 = cKDTree(pc1)
        kdtree2 = cKDTree(pc2)
        distances1, _ = kdtree1.query(pc2)
        distances2, _ = kdtree2.query(pc1)
        return np.mean(distances1) + np.mean(distances2)

    def compute_loss(self, generated_point_cloud):
        return self.chamfer_distance(generated_point_cloud, self.target_point_cloud)

    def compute_gradients(self):
        # Assuming smpl_model has a method that computes vertices and their gradients w.r.t. betas
        verts, grad_betas = self.smpl_model(self.betas, return_grad=True)
        loss = self.compute_loss(verts)
        gradients = np.zeros(self.num_betas)

        for i in range(self.num_betas):
            # Compute gradient of the loss w.r.t. each beta
            gradients[i] = np.mean(2 * (verts - self.target_point_cloud) @ grad_betas[:, i])

        return loss, gradients

    def optimize(self, target_point_cloud, num_iterations=1000):
        self.target_point_cloud = target_point_cloud
        for i in range(num_iterations):
            loss, gradients = self.compute_gradients()
            self.betas -= self.learning_rate * gradients
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss}")
                
        return self.betas

# Example usage
if __name__ == "__main__":
    optimizer = SMPLModelOptimizer(smpl_model, target_point_cloud)
    optimized_betas = optimizer.optimize()

    print("Optimized Betas:", optimized_betas)
# smpl_model = YourSMPLModelClass()  # Initialize your SMPL model here
# target_point_cloud = np.load("target_point_cloud.npy")  # Load your target point cloud here

# optimizer = SMPLModelOptimizer(smpl_model, target_point_cloud)
# optimized_betas = optimizer.optimize()

# print("Optimized Betas:", optimized_betas)