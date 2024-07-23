#! /usr/bin/python3
import json
import numpy as np
import mmap
from rclpy.node import Node
import os
'''
This scripts is used to send SMPL parameters collected by smpl_tracking_node to docker for the SKEL fitting procedure 
'''
class SMPLParamsSender(Node):
    def __init__(self, device):
        super().__init__('ParamSender')
        self.size = 1024 * 1024 * 10
        self.device = device
        self.file_path = os.path.expanduser('~') + '/mmap_file.txt'
        with open(self.file_path, 'wb') as f:
            f.write(b'\x00' * (self.size))
    
    def send(self, data):
        # self.get_logger().info("Sending SMPL parameters on mmap")
        with open(self.file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), self.size)
            data = {
                "betas": data.betas.detach().cpu().numpy().tolist(), # torch tensor
                "gender": data.gender, # string
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
