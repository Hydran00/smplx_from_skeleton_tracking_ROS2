#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from body_msgs.msg import BodyData
from sensor_msgs.msg import PointCloud2
import tf2_ros
from tf2_ros import Buffer, TransformListener
import pickle
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
from utils import  block
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

# Device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SMPLTracking(Node):
    def __init__(self):
        super().__init__('SMPL_tracking')
        self.get_logger().info("Initializing SMPL tracking node")
        
        # Subscriber for body tracking data
        self.create_subscription(BodyData, 'body_tracking_data', self.callback_bd, 1)
        self.create_subscription(PointCloud2, 'point_cloud', self.callback_pc, 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.transform_path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/transform.pkl"
        self.camera_frame_name = "zed2_left_camera_frame"
        self.base_frame = "base_link"
        
        # PARAMS
        self.declare_parameter('model_path', '../smpl/SMPL_MALE.pkl')
        self.declare_parameter('model_type', 'smpl')
        self.declare_parameter('mirror', False)
        self.declare_parameter('optimize_model', True)
        
        # Get params from ros parameter
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value
        self.mirror = self.get_parameter('mirror').get_parameter_value().bool_value
        # if False, use this script just for visualization 
        self.optimize_model = self.get_parameter('optimize_model').get_parameter_value().bool_value

        if self.optimize_model == True:
            self.max_points = 80000
        else:
            self.max_points = 10000
        
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
        
        self.timer = self.create_timer(0.03, self.update_model)
        if(self.model_type == 'smpl'):
            self.model = smplx.create(self.model_path, model_type=self.model_type, num_betas=NUM_BETAS).to(DEVICE)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}, only supported 'smpl'")
        
        # Initialize pose and shape parameters
        self.betas = torch.zeros((1, NUM_BETAS)).to(DEVICE)
        self.global_orient = torch.zeros((1, 3)).to(DEVICE)
        self.global_position = torch.zeros((1, 3)).to(DEVICE)
        self.gender = 'male'
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
        self.offset_computed = False
        self.mesh = o3d.geometry.TriangleMesh()
        self.point_cloud = o3d.geometry.PointCloud()
        
        self.get_logger().info("Tracking node started")
        
        if self.optimize_model:
            self.param_sender = SMPLParamsSender(DEVICE)
            self.params_optimizer = SMPLModelOptimizer(self.model, learning_rate=0.1, num_betas=NUM_BETAS)
        self.betas_optimized = False
        
        # self.viz.add_geometry(self.reference_frame)
        self.spheres = []
        self.spheres_smpl = []
        self.it=0

        output = self.model(
        transl=self.global_position,
        betas=self.betas,
        global_orient=self.global_orient,
        body_pose=self.body_pose,
        return_verts=True
        )
        vertices = output.vertices[0].detach().cpu().numpy() 
        faces = self.model.faces
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh("Tpose.ply", self.mesh)
    
    def dump_transform(self):
        try:
            # Wait for the transform between 'base_link' and 'odom' for up to 5 seconds
            while(not self.tf_buffer.can_transform(self.base_frame, self.camera_frame_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))):
                self.get_logger().info("Waiting for transform")
                rclpy.spin_once(self)
            self.get_logger().info("Transform found")
            # Lookup the transform after the waiting period
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"TransformException: {e}")
        
        trans = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame_name, rclpy.time.Time())
        # Access transform details
        self.get_logger().info(f'Translation: {trans.transform.translation}')
        self.get_logger().info(f'Rotation: {trans.transform.rotation}')
        
        trans_matrix = np.eye(4)
        trans_matrix[:3, 3] = [trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z]
        quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
        rot_matrix = scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()
        trans_matrix[:3, :3] = rot_matrix
        with open(self.transform_path, 'wb') as f:
            # dump information to that file
            self.get_logger().info("Dumping transform: \n" + str(trans_matrix))
            pickle.dump(trans_matrix, f)


        
    
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
        with torch.no_grad():
            self.global_position[0] = torch.tensor([msg.keypoints[2].position.x, msg.keypoints[2].position.y , msg.keypoints[2].position.z]).to(DEVICE)
            self.global_orient[0] = torch.tensor(self.quaternion_to_rotvec(msg.global_root_orientation)).to(DEVICE)
            
        self.current_body_pose = msg.local_orientation_per_joint
        # self.update_model()
        

    def callback_pc(self, msg):
        """Callback for point cloud data."""
        # self.get_logger().info("Received point cloud data")
        fromPointCloud2(self, self.point_cloud, msg, self.max_points)
        # self.get_logger().info("Converted")            
        
        if(self.point_cloud.is_empty()):
            # self.get_logger().error("Point cloud is empty")
            return
       
        if self.first_point_cloud:

            self.viz.add_geometry(self.point_cloud)
            self.first_point_cloud = False
    

        self.viz.update_geometry(self.point_cloud)
        self.viz.poll_events()
        self.viz.update_renderer()
        
    def get_skel_model(self):
        # self.params.body_pose_axis_angle = torch.cat([self.body_pose, torch.tensor([[0, 0, 0]]).to(self.device), torch.tensor([[0, 0, 0]]).to(self.device)], dim=1)
        self.get_logger().info("Sending optimized betas, waiting for skel mesh")
        params = SMPLParams(self.betas, self.gender, self.global_position, self.global_orient, self.body_pose)
        self.param_sender.send(params)
        if(self.first_skel_mesh):
            self.skel_mesh = o3d.geometry.TriangleMesh()
            self.viz.poll_events()
            self.viz.update_renderer()
            self.param_sender.receive_skel(self.skel_mesh)
            self.viz.add_geometry(self.skel_mesh)
            self.first_skel_mesh = False
            self.get_logger().info("First skel mesh received")
            block(self.viz)


    def update_model(self):
        """Draw the SMPL model with the current pose."""
        if self.current_body_pose is None or self.first_point_cloud:
            self.get_logger().error("Body pose or point cloud are not available")
            return

        # add body pose
        for i in range(NUM_BODY_JOINTS):
            rvec = self.quaternion_to_rotvec(self.current_body_pose[self.map[i+1]])
            self.body_pose[0][3*i] = rvec[0]
            self.body_pose[0][3*i+1] = rvec[1]
            self.body_pose[0][3*i+2] = rvec[2]
        
        # add hand pose
        rvec_r = self.quaternion_to_rotvec(self.current_body_pose[self.map[21]])
        rvec_l = self.quaternion_to_rotvec(self.current_body_pose[self.map[22]])
            
        self.body_pose[0][3*21] = rvec_r[0]
        self.body_pose[0][3*21+1] = rvec_r[1]
        self.body_pose[0][3*21+2] = rvec_r[2]
        self.body_pose[0][3*22] = rvec_r[0]
        self.body_pose[0][3*22+1] = rvec_r[1]
        self.body_pose[0][3*22+2] = rvec_r[2]
        
        
        # optimize only the first iteration
        if self.optimize_model and not self.betas_optimized and self.first_mesh:
            # remove outliers from the point cloud

            # filter the point cloud with a plane on the Z axis
            filtered_points = np.asarray(self.point_cloud.points)[np.asarray(self.point_cloud.points)[:, 2] < self.global_position[0, 2].cpu().detach().numpy()+0.2]
            colors = np.asarray(self.point_cloud.colors)[np.asarray(self.point_cloud.points)[:, 2] < self.global_position[0, 2].cpu().detach().numpy()+0.2]
            self.point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.viz.update_geometry(self.point_cloud)
            self.viz.remove_geometry(self.mesh)
            # dump pointcloud
            o3d.io.write_point_cloud("point_cloud.ply", self.point_cloud)
            
            # perform forward step to create the mesh used in adjust_landmarks
            output = self.model(
            transl=self.global_position,
            betas=self.betas,
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            return_verts=True
            )
            vertices = output.vertices[0].detach().cpu().numpy() 
            faces = self.model.faces
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.mesh.triangles = o3d.utility.Vector3iVector(faces)
            
            self.betas, self.body_pose, self.global_position = self.params_optimizer.optimize_model(
                                                        self.get_logger(),
                                                       self.point_cloud, 
                                                       self.global_orient, 
                                                       self.global_position, 
                                                       self.body_pose,
                                                       self.landmarks,
                                                       self.viz)
            self.betas_optimized = True
            
        # offset is already added in the optimization, but not in the normal case. 
        # Skipping the first iteration since the mesh is needed for the computation of the offset
        # if not self.first_mesh and not self.optimize_model:
        #     # offset is due to the fact that the ZED Body Tracking SDK project the joints on the surface of the cloud and not in the anatomical position
        #     with torch.no_grad():
        #         if not self.offset_computed:
        #             offset = -compute_distance_from_pelvis_joint_to_surface(self.mesh, self.global_position, self.global_orient)
        #             # check if inf is returned
        #             self.offset = torch.tensor(offset, dtype=torch.float32, device='cuda:0')
        #             self.offset_computed = True
        #             if True in torch.isinf(self.offset):
        #                 self.get_logger().error("Offset is infinite")
        #                 return
        #         else:
        #             self.global_position[0, :] =  self.global_position[0, :] + self.offset
                    # block()
                    
            # self.viz.add_geometry(self.mesh)


        # forward pass
        output = self.model(
            transl=self.global_position,
            betas=self.betas,
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            return_verts=True
        )
        vertices = output.vertices[0].detach().cpu().numpy() 
        faces = self.model.faces
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(faces)        
        landmarks_smpl = output.joints.detach().cpu().numpy()
        landmarks_smpl = landmarks_smpl[:, :24, :].reshape(1, 72)
        
        if self.first_mesh:
            self.viz.add_geometry(self.mesh)
            self.first_mesh = False
            self.get_logger().info("First mesh received")
            
            # add 21 spheres foreach joint location
            # for i in range(NUM_BODY_JOINTS):
            #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            #     sphere_smpl = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
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

        else:
            self.viz.update_geometry(self.mesh)
            # for i in range(NUM_BODY_JOINTS):
            #     self.spheres[i].translate(self.landmarks[0][i*3:i*3+3].detach().cpu().numpy(), relative=False)
            #     self.spheres_smpl[i].translate(landmarks_smpl[0][i*3:i*3+3], relative=False)
            #     self.viz.update_geometry(self.spheres[i])
            #     self.viz.update_geometry(self.spheres_smpl[i])

        # GET SKEL MODEL
        if self.betas_optimized:
            self.viz.remove_geometry(self.mesh)
            path = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/beta_opt.obj"
            o3d.io.write_triangle_mesh(path, self.mesh)
            self.dump_transform()
            self.get_skel_model()
            self.viz.remove_geometry(self.mesh) 
            while rclpy.ok():
                # delete timer
                self.timer.cancel()
                # vf_calculator = VirtualFixtureCalculator()
                # self.get_logger().info("Virtual fixture calculated -> exiting")
                self.param_sender.receive_skel(self.skel_mesh)
                self.viz.poll_events()
                self.viz.update_renderer()
                time.sleep(0.03)

        self.viz.poll_events()
        self.viz.update_renderer()
        self.it=0
        # while True:

        #     self.viz.poll_events()
        #     self.viz.update_renderer()
        #     time.sleep(0.03)
        
        
            
        
def main(args=None):
    rclpy.init(args=args)
    node = SMPLTracking()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()