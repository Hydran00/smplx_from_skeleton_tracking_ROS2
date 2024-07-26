#!/usr/bin/python3
import rclpy
from rclpy.node import Node
import numpy as np
from body_msgs.msg import BodyData
from sensor_msgs.msg import PointCloud2
import scipy
import torch
import torch.utils.dlpack
import open3d as o3d
import smplx
import os
from dataclasses import dataclass
from smpl_params_sender import SMPLParamsSender
from smpl_optmizer import SMPLModelOptimizer
import time
@dataclass
class SMPLParams:
    betas: torch.Tensor
    gender: str
    global_position: torch.Tensor
    global_orient: torch.Tensor
    body_pose_axis_angle: torch.Tensor

# Importing required mappings
from body38_to_smpl import ZED_BODY_38_TO_SMPL_BODY_24 as MAP, ZED_BODY_38_TO_SMPL_BODY_24_MIRROR as MAP_MIR

from open3d_converter import fromPointCloud2
# Constants
NUM_BETAS = 10
NUM_EXPRESSIONS = 10
NUM_BODY_JOINTS = 21 # 24 - 3 (transl/pelvis and hands)
NUM_FACE_JOINTS = 3
NUM_HAND_JOINTS = 15

OPTIMIZE_BETAS = True

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SMPLXTracking(Node):
    def __init__(self):
        super().__init__('SMPLX_tracking')
        self.get_logger().info("Initializing SMPLX tracking node")
        
        # Subscriber for body tracking data
        self.create_subscription(BodyData, 'body_tracking_data', self.callback_bd, 1)
        self.create_subscription(PointCloud2, 'point_cloud', self.callback_pc, 1)

        
        # PARAMS
        self.declare_parameter('model_path', '../models/smplx/SMPLX_MALE.npz')
        self.declare_parameter('model_type', 'smplx')
        self.declare_parameter('mirror', False)
        
        # Get params from ros parameter
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
        self.mirror = self.get_parameter('mirror').get_parameter_value().bool_value
        
        if self.mirror:
            self.map = MAP_MIR
        else:
            self.map = MAP
        
        # log parameters
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Model type: {self.model_type}")
        self.get_logger().info(f"Mirror: {self.mirror}")        
        
        
        if(os.path.exists(self.model_path) == False):
            self.get_logger().error("Model path does not exist")
            exit(1)
        
        
        # self.get_logger().info("Loading model: {}".format(self.model_type))
        # Load SMPLX/SMPL model
        if(self.model_type == 'smplx'):
            self.model = smplx.create(self.model_path, model_type=self.model_type,
                                        num_betas=NUM_BETAS, num_expressions=NUM_EXPRESSIONS, use_pca=False).to(DEVICE)
        elif(self.model_type == 'smpl'):
            self.model = smplx.create(self.model_path, model_type=self.model_type, num_betas=NUM_BETAS).to(DEVICE)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}, supported are 'smpl' and 'smplx'")
        
        # Initialize pose and shape parameters
        self.betas = torch.zeros((1, NUM_BETAS)).to(DEVICE)
        self.global_orient = torch.zeros((1, 3)).to(DEVICE)
        self.global_position = torch.zeros((1, 3)).to(DEVICE)
        self.gender = 'male'
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
        self.landmarks = torch.zeros((1, 3 * (NUM_BODY_JOINTS+3))).to(DEVICE)    

        # Open3D visualization setup
        self.viz = o3d.visualization.Visualizer()
        self.viz.create_window()
        opt = self.viz.get_render_option()
        opt.show_coordinate_frame = True
        # opt.background_color = np.asarray([0.5, 0.5, 0.5])
        opt.mesh_show_wireframe = True
        self.first_mesh = True
        self.first_skel_mesh = True
        self.first_point_cloud = True
        self.mesh = o3d.geometry.TriangleMesh()
        self.point_cloud = o3d.geometry.PointCloud()
        
        # Timer for periodic drawing
        self.timer = self.create_timer(0.03, self.draw)
        self.get_logger().info("Tracking node started")
        
        if OPTIMIZE_BETAS:
            self.param_sender = SMPLParamsSender(DEVICE)
            self.betas_optimizer = SMPLModelOptimizer(self.model, learning_rate=0.1, num_betas=NUM_BETAS)
        self.betas_optimized = False
        
        # self.reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        # self.viz.add_geometry(self.reference_frame)
        self.spheres = []
        self.spheres_smpl = []
    
    @staticmethod
    def quaternion_to_rotvec(quat):
        def wrap_angle(angle):
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle
        """Convert quaternion to rotation vector using scipy."""
        q = np.array([quat.x, quat.y, quat.z, quat.w])
        if np.linalg.norm(q) == 0:
            return np.zeros(3)
        q = q / np.linalg.norm(q)
        rvec = scipy.spatial.transform.Rotation.from_quat(q).as_rotvec()
        rvec = np.array(rvec)
        for i in range(3):
            rvec[i] = wrap_angle(rvec[i])
        return rvec 
    

    def callback_bd(self, msg):
        """Callback for body tracking data."""
        # self.get_logger().info("Received body tracking data")
        for i in range(NUM_BODY_JOINTS+3):
                self.landmarks[0][i*3] = msg.keypoints[self.map[i]].position.x
                self.landmarks[0][i*3 + 1] = msg.keypoints[self.map[i]].position.y
                self.landmarks[0][i*3 + 2] = msg.keypoints[self.map[i]].position.z
                
        # SMPL translation seems to be defined with respect to the SPINE_2
        self.global_position[0] = torch.tensor([msg.keypoints[2].position.x, msg.keypoints[2].position.y , msg.keypoints[2].position.z]).to(DEVICE)
        
        self.global_orient[0] = torch.tensor(self.quaternion_to_rotvec(msg.global_root_orientation)).to(DEVICE)
        self.current_body_pose = msg.local_orientation_per_joint
        
        self.draw()
        

    def callback_pc(self, msg):
        """Callback for point cloud data."""
        # self.get_logger().info("Received point cloud data")
            
        fromPointCloud2(self, self.point_cloud, msg)

        if self.first_point_cloud:
            self.viz.add_geometry(self.point_cloud)
            self.first_point_cloud = False
        
        if(self.point_cloud.is_empty()):
            self.get_logger().error("Point cloud is empty")
            return
        self.viz.update_geometry(self.point_cloud)
        self.viz.poll_events()
        self.viz.update_renderer()
        pass
        
        

    def draw(self):
        # self.get_logger().info("Drawing")
        """Draw the SMPLX model with the current pose."""
        if self.current_body_pose is None:
            return

        # add body pose (valid for both smpl and smplx)
        for i in range(NUM_BODY_JOINTS):
            rvec = self.quaternion_to_rotvec(self.current_body_pose[self.map[i+1]])
            self.body_pose[0][3*i] = rvec[0]
            self.body_pose[0][3*i+1] = rvec[1]
            self.body_pose[0][3*i+2] = rvec[2]
        
        # add hand pose
        if self.model_type == 'smpl':
            rvec_r = self.quaternion_to_rotvec(self.current_body_pose[self.map[21]])
            rvec_l = self.quaternion_to_rotvec(self.current_body_pose[self.map[22]])
                
            self.body_pose[0][3*21] = rvec_r[0]
            self.body_pose[0][3*21+1] = rvec_r[1]
            self.body_pose[0][3*21+2] = rvec_r[2]
            self.body_pose[0][3*22] = rvec_r[0]
            self.body_pose[0][3*22+1] = rvec_r[1]
            self.body_pose[0][3*22+2] = rvec_r[2]
        
        else:
            # TODO: add hand pose for smplx
            pass
        
        if not self.first_mesh and not self.first_point_cloud and not self.betas_optimized and OPTIMIZE_BETAS:
            self.get_logger().info("Optimizing params") 

            self.betas, self.body_pose, self.global_position = self.betas_optimizer.optimize_model(
                                                        self.get_logger(),
                                                       self.point_cloud, 
                                                       self.global_orient, 
                                                       self.global_position, 
                                                       self.body_pose,
                                                       self.landmarks)
            self.betas_optimized = True

        # forward pass
        if self.model_type == 'smpl':
            output = self.model(
                transl=self.global_position,
                betas=self.betas,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                return_verts=True
            )
        else:
            output = self.model(
                betas=self.betas,
                transl=self.global_position,
                global_orient=self.global_orient,
                body_pose=self.body_pose,
                jaw_pose=self.jaw_pose,
                leye_pose=self.left_eye_pose,
                reye_pose=self.right_eye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                expression=self.expression,
                return_verts=True,
                return_full_pose=False
            )


        # self.get_logger().info("SMPLX model forward pass completed")
        
        vertices = output.vertices[0].detach().cpu().numpy() 
        faces = self.model.faces

        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)        
        
        landmarks_smpl = output.joints.detach().cpu().numpy()
        landmarks_smpl = landmarks_smpl[:, :24, :].reshape(1, 72)
        if self.first_mesh:
            self.viz.add_geometry(self.mesh)
            
            # add 21 spheres foreach joint location
            # for i in range(NUM_BODY_JOINTS):
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            #     sphere_smpl = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            #     sphere.compute_vertex_normals()
            #     sphere_smpl.compute_vertex_normals()
            #     sphere.paint_uniform_color([0, 1, 0])
            #     sphere_smpl.paint_uniform_color([1, 0, 0])
            #     sphere.translate(self.landmarks[0][i*3:i*3+3].detach().cpu().numpy(), relative=False)
            #     sphere_smpl.translate(landmarks_smpl[0][i*3:i*3+3], relative=False)
            #     self.spheres.append(sphere)
            #     self.spheres_smpl.append(sphere_smpl)
            #     self.viz.add_geometry(sphere)
            #     self.viz.add_geometry(sphere_smpl)
            self.first_mesh = False
            self.get_logger().info("First mesh received")
        else:
            self.viz.update_geometry(self.mesh)
            # for i in range(NUM_BODY_JOINTS):
            #     self.spheres[i].translate(self.landmarks[0][i*3:i*3+3].detach().cpu().numpy(), relative=False)
            #     self.spheres_smpl[i].translate(landmarks_smpl[0][i*3:i*3+3], relative=False)
            #     self.viz.update_geometry(self.spheres[i])
            #     self.viz.update_geometry(self.spheres_smpl[i])

        
        # send params to docker

        if self.betas_optimized:
            # self.params.body_pose_axis_angle = torch.cat([self.body_pose, torch.tensor([[0, 0, 0]]).to(self.device), torch.tensor([[0, 0, 0]]).to(self.device)], dim=1)
            if self.model_type == 'smpl':
                self.get_logger().info("Sending optimized betas, waiting for skel mesh")
                params = SMPLParams(self.betas, self.gender, self.global_position, self.global_orient, self.body_pose)
            else:
                # TODO: add hand pose for smplx
                params = SMPLParams(self.betas, self.gender, self.global_position, self.global_orient, torch.cat([self.body_pose, torch.tensor([[0, 0, 0]]).to(DEVICE), torch.tensor([[0, 0, 0]]).to(DEVICE)], dim=1))

            self.param_sender.send(params)
            if(self.first_skel_mesh):
                self.skel_mesh = o3d.geometry.TriangleMesh()
                self.viz.poll_events()
                self.viz.update_renderer()
                self.param_sender.receive_skel(self.skel_mesh)
                self.viz.add_geometry(self.skel_mesh)
                self.first_skel_mesh = False
                self.get_logger().info("First skel mesh received")
                # self.viz.update_geometry(self.skel_mesh)
                # break only if space is pressed
                
                # remove the smpl mesh
                self.viz.remove_geometry(self.mesh) 
                while True:
                    # detect key press
                    # self.param_sender.receive_skel(self.skel_mesh)
                    self.viz.poll_events()
                    self.viz.update_renderer()
                    time.sleep(0.03)
                    self.get_logger().info("Waiting for space key press")
            else:
                self.param_sender.receive_skel(self.skel_mesh)

            self.viz.poll_events()
            self.viz.update_renderer()
        
        self.viz.poll_events()
        self.viz.update_renderer()
            
        
def main(args=None):
    rclpy.init(args=args)
    node = SMPLXTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()