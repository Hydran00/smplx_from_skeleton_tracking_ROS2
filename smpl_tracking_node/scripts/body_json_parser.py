#!/usr/bin/env python3
import socket
import struct
import threading
from collections import deque
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Quaternion
from body_msgs.msg import Keypoint, BodyData  # Assume these are defined as per the custom message definitions

class ParseBodyJson(Node):
    def __init__(self, port=20000, multicast_ip_address="230.0.0.1", use_multicast=True, buffer_size=65536):
        super().__init__('zed_streaming_client')
        self.get_logger().info("Starting ZED Body Tracking Data Parser, if the node is not receiving data, please disable firewall")
        self.port = port
        self.multicast_ip_address = multicast_ip_address
        self.use_multicast = use_multicast
        self.buffer_size = buffer_size
        self.show_zed_fusion_metrics = False
        self.new_data_available = False
        self.received_data_buffer = deque(maxlen=buffer_size)
        self.publisher_ = self.create_publisher(BodyData, 'body_tracking_data', 1)
        self.timer = self.create_timer(0.01, self.update)
        self.initialize_udp_listener()
        print('Node started')

    def initialize_udp_listener(self):
        self.client_data = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.client_data.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if self.use_multicast:
            self.client_data.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                        struct.pack("4sl", socket.inet_aton(self.multicast_ip_address), socket.INADDR_ANY))

        self.client_data.bind(('', self.port))
        self.get_logger().info("UDP - Start Receiving..")
        threading.Thread(target=self.receive_udp_packet, daemon=True).start()

    def receive_udp_packet(self):
        while True:
            try:
                received_bytes, _ = self.client_data.recvfrom(65536)
                self.parse_packet(received_bytes)
            except Exception as e:
                self.get_logger().error(f"Error receiving packet: {e}")

    def parse_packet(self, received_bytes):
        self.received_data_buffer.append(received_bytes)
        self.new_data_available = True

    def is_new_data_available(self):
        return self.new_data_available

    def get_last_bodies_data(self):
        if self.received_data_buffer:
            data_json = self.received_data_buffer[-1].decode('utf-8')
            data = json.loads(data_json)
            return data.get('bodies', None)
        return None

    def update(self):
        if self.is_new_data_available():
            print("New data available")
            bodies_data = self.get_last_bodies_data()
            if bodies_data:
                self.publish_body_data(bodies_data)
            self.new_data_available = False

    def publish_body_data(self, bodies_data):
        for body in bodies_data.get('body_list', []):
            body_msg = BodyData()

            # Fill BodyData message
            body_msg.global_position = self.vector3_from_dict(body['position'])
            body_msg.global_root_orientation = self.quaternion_from_dict(body['global_root_orientation'])

            for kp in body['keypoint']:
                keypoint_msg = Keypoint()
                keypoint_msg.position = self.vector3_from_dict(kp)
                keypoint_msg.local_position_per_joint = [self.vector3_from_dict(lp) for lp in body['local_position_per_joint']]
                keypoint_msg.local_orientation_per_joint = [self.quaternion_from_dict(lo) for lo in body['local_orientation_per_joint']]
                body_msg.keypoints.append(keypoint_msg)
            print("pub")
            self.publisher_.publish(body_msg)

    @staticmethod
    def vector3_from_dict(data):
        vector = Vector3()
        vector.x = data['x']
        vector.y = data['y']
        vector.z = data['z']
        return vector

    @staticmethod
    def quaternion_from_dict(data):
        quaternion = Quaternion()
        quaternion.x = data['x']
        quaternion.y = data['y']
        quaternion.z = data['z']
        quaternion.w = data['w']
        return quaternion

    def on_destroy(self):
        if self.client_data:
            self.get_logger().info("Stop receiving ..")
            self.client_data.close()

def main(args=None):
    rclpy.init(args=args)
    client = ParseBodyJson()

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.on_destroy()
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
