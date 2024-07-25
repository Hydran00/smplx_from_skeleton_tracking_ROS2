#! /usr/bin/python3
import json
import numpy as np
import mmap
from rclpy.node import Node
import os
import trimesh
from io import BytesIO
import open3d as o3d
import time
'''
This scripts is used to send SMPL parameters collected by smpl_tracking_node to docker for the SKEL fitting procedure 
'''
class SMPLParamsSender(Node):
    def __init__(self, device):
        super().__init__('ParamSender')
        self.size = 1024 * 1024 * 10
        self.device = device
        self.file_path = os.path.expanduser('~') + '/mmap/mmap_smpl_params.txt'
        self.skel_path = os.path.expanduser('~') + '/mmap/mmap_skel.txt'
        self.flag_path = os.path.expanduser('~') + '/mmap/mmap_skel_flag.txt'
        with open(self.file_path, 'wb') as f:
            f.write(b'\x00' * (self.size))
        with open(self.skel_path, 'wb') as f:
            f.truncate(0) # need '0' when using r+
        with open(self.flag_path, 'wb') as f:
            f.truncate(0)
    def send(self, data):
        # self.get_logger().info("Sending SMPL parameters on mmap")
        with open(self.file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), self.size)
            data = {
                "betas": data.betas.detach().cpu().numpy().tolist(), # torch tensor
                "gender": data.gender, # string
                "global_position": data.global_position.detach().cpu().numpy().tolist(), # torch tensor
                "global_orient": data.global_orient.detach().cpu().numpy().tolist(), # torch tensor
                "body_pose_axis_angle" : data.body_pose_axis_angle.detach().cpu().numpy().tolist() # torch tensor
            }
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')
            json_length = len(json_bytes)
            
            if json_length > self.size - 4:
                raise ValueError("Data size exceeds shared memory size")

            mm.seek(0)
            mm.write(json_length.to_bytes(4, byteorder='little'))
            mm.write(json_bytes)
            mm.close()
    
    def receive_skel(self, mesh):

        # simple way to synchronize docker and host
        while True:
            if not os.path.exists(self.flag_path):
                continue
            # check flag
            with open(self.flag_path, 'r') as flag_file:
                if flag_file.read().strip() != '1':
                    continue
            os.remove(self.flag_path)
            
            # read the file
            with open(self.skel_path, 'r+b') as f:
                # Memory-map the file, size 0 means whole file
                mm = mmap.mmap(f.fileno(), 0)

                mm.seek(0)

                obj_data = mm.read()
                # Read the data from the mmap file
                obj_data_io = BytesIO(obj_data)
                # Close the memory-mapped file
                mm.close()

            # Reconstruct the mesh from the serialized data
            output = trimesh.load(obj_data_io, file_type='obj')

            mesh.vertices = o3d.utility.Vector3dVector(output.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(output.faces)

            # rotate the mesh 180 on y  axis on local coordinate
            # mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi, 0)))



            return mesh
