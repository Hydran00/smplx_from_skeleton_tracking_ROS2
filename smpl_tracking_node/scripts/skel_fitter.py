#!/usr/bin/env python3.8
import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "SKEL_WS", "SKEL"))
sys.path.append(os.path.join(os.path.expanduser("~"), "SKEL_WS", "SKEL", "skel_venv", "lib", "python3.8", "site-packages","torch"))
sys.path.append(os.path.join(os.path.expanduser("~"), "SKEL_WS", "SKEL", "skel_venv", "lib", "python3.8", "site-packages","numpy"))
sys.path.append(os.path.join(os.path.expanduser("~"), "SKEL_WS", "SKEL", "skel_venv", "lib", "python3.8", "site-packages","numpy.libs"))
sys.path.append(os.path.join(os.path.expanduser("~"), "SKEL_WS", "SKEL", "skel_venv", "lib", "python3.8", "site-packages","numpy-1.24.4.dist-info"))
import trimesh
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import numpy as np
from body_msgs.msg import BodyData

import scipy
import torch
import open3d as o3d
import smplx

from skel.skel_model import SKEL
# Importing required mappings
from body38_to_smpl import ZED_BODY_38_TO_SMPL_BODY_24 as map, ZED_BODY_38_TO_SMPL_BODY_24_MIRROR as map_mir

# Constants
NUM_BETAS = 10
NUM_EXPRESSIONS = 10
NUM_GLOBAL_ORIENT_JOINTS = 1 # pelvis
NUM_BODY_JOINTS = 21 # 24 - 3 (pelvis and hands)
NUM_FACE_JOINTS = 3
NUM_HAND_JOINTS = 15

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SMPLXTracking(Node):
    def __init__(self):
        super().__init__('pose_depth_estimator')

        self.bridge = CvBridge()
        
        # Subscriber for body tracking data
        self.create_subscription(BodyData, 'body_tracking_data', self.callback, 10)
        
        # PARAMS
        self.declare_parameter('model_path', '../models/smplx/SMPLX_MALE.npz')
        self.declare_parameter('model_type', 'smplx')
        self.declare_parameter('mirror', True)
        
        # Get params from ros parameter
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
        self.mirror = self.get_parameter('mirror').get_parameter_value().bool_value
        
        # log parameters
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Model type: {self.model_type}")
        self.get_logger().info(f"Mirror: {self.mirror}")        
        
        if(os.path.exists(self.model_path) == False):
            self.get_logger().error("Model path does not exist")
            exit(1)
        
        
        # Load SMPLX/SMPL model
        if(self.model_type == 'smplx'):
            self.model = smplx.create(self.model_path, model_type=self.model_type,
                                        num_betas=NUM_BETAS, num_expressions=NUM_EXPRESSIONS, use_pca=False).to(DEVICE)
        elif(self.model_type == 'smpl'):
            self.model = smplx.create(self.model_path, model_type=self.model_type, num_betas=NUM_BETAS).to(DEVICE)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}, supported are 'smpl' and 'smplx'")
        
        self.skel_model = SKEL(gender='male').to(DEVICE)
        
        # Initialize pose and shape parameters
        self.betas = torch.zeros((1, NUM_BETAS)).to(DEVICE)
        self.global_orient = torch.zeros((1, 3)).to(DEVICE)
        if self.model_type == 'smplx':
            self.body_pose = torch.zeros((1, 3 * NUM_BODY_JOINTS)).to(DEVICE)
            self.jaw_pose = torch.zeros((1, 3)).to(DEVICE)
            self.left_eye_pose = torch.zeros((1, 3)).to(DEVICE)
            self.right_eye_pose = torch.zeros((1, 3)).to(DEVICE)
            self.left_hand_pose = torch.zeros((1, 3 * NUM_HAND_JOINTS)).to(DEVICE)
            self.right_hand_pose = torch.zeros((1, 3 * NUM_HAND_JOINTS)).to(DEVICE)
            self.expression = torch.zeros((1, NUM_EXPRESSIONS)).to(DEVICE)
        else:
            # smpl has 23 body params (21 + 2 hands)
            self.body_pose = torch.zeros((1, 3 * NUM_BODY_JOINTS + 3 * 2)).to(DEVICE)
        self.current_body_pose = None 
        
        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        opt.mesh_show_wireframe = True
        self.first = True
        self.mesh = o3d.geometry.TriangleMesh()
        
        # Timer for periodic drawing
        self.timer = self.create_timer(0.01, self.draw)
        self.get_logger().info("Tracking node started")
    
    @staticmethod
    def quaternion_to_rotvec(self, quat):
        """Convert quaternion to rotation vector using scipy."""
        q = np.array([quat.x, quat.y, quat.z, quat.w])
        if np.linalg.norm(q) == 0:
            return np.zeros(3)
        q = q / np.linalg.norm(q)
        rvec = scipy.spatial.transform.Rotation.from_quat(q).as_rotvec()
        rvec = np.array(rvec)
        for i in range(3):
            rvec[i] = self.wrap_angle(rvec[i])
        return rvec 
    
    @staticmethod
    # wrap angles to [-pi, pi]
    def wrap_angle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def callback(self, msg):
        """Callback for body tracking data."""
        global_orientation = self.quaternion_to_rotvec(msg.global_root_orientation)
        self.global_orient[0] = torch.tensor(global_orientation).to(DEVICE)
        self.current_body_pose = msg.keypoints[0].local_orientation_per_joint


    def draw(self):
        """Draw the SMPLX model with the current pose."""
        if self.current_body_pose is None:
            return
        # add body pose (valid for both smpl and smplx)
        for i in range(NUM_BODY_JOINTS):
            if(self.mirror):
                rvec = self.quaternion_to_rotvec(self.current_body_pose[map_mir[i+1]])
            else:
                rvec = self.quaternion_to_rotvec(self.current_body_pose[map[i+1]])
            self.body_pose[0][3*i] = rvec[0]
            self.body_pose[0][3*i+1] = rvec[1]
            self.body_pose[0][3*i+2] = rvec[2]
        
        # add hand pose
        if self.model_type == 'smpl':
            
            if(self.mirror):
                rvec = self.quaternion_to_rotvec(self.current_body_pose[map_mir[21]])
            else:
                rvec = self.quaternion_to_rotvec(self.current_body_pose[map[21]])
                
            self.body_pose[0][3*21] = rvec[0]
            self.body_pose[0][3*21+1] = rvec[1]
            self.body_pose[0][3*21+2] = rvec[2]
        
        else:
            # TODO: add hand pose for smplx
            pass
            
        # Now convert to SKEL model
        skel = SKEL(gender='female').to(DEVICE)

        # Set parameters to default values (T pose)
        pose = torch.zeros(1, skel.num_q_params).to(DEVICE) # (1, 46)
        betas = torch.zeros(1, skel.num_betas).to(DEVICE) # (1, 10)
        trans = torch.zeros(1, 3).to(DEVICE)
        
        skel_output = skel(pose, betas, trans)
        # output_copy = 
        # visualize
        vertices = skel_output.skel_verts.detach().cpu().numpy()[0]
        faces = skel.skel_f.cpu()
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.show()

        # vertices = output.vertices[0].detach().cpu().numpy()
        # faces = self.model.faces
        # self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # if self.first:
        #     self.viz.add_geometry(self.mesh)
        #     self.first = False
        
        # self.viz.update_geometry(self.mesh)
        # self.viz.poll_events()
        # self.viz.update_renderer()
        
def main(args=None):
    rclpy.init(args=args)
    node = SMPLXTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
