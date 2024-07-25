from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
import ctypes
import struct
from ctypes import *


MAX_POINTS = 20000
# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# filtered_data_idx = 0
# old_max = 0

def fromPointCloud2(node, buffer, ros_cloud):
    # global filtered_data_idx, old_max
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))
    
    # if len(cloud_data)> old_max:
    #     # extract randomly 10k points
    # filtered_data_idx = np.random.choice(len(cloud_data), min(10000, len(cloud_data)), replace=False)
    filtered_data_idx = np.random.choice(len(cloud_data), min(len(cloud_data),MAX_POINTS), replace=False)
    #     old_max = len(cloud_data)
    
    
    cloud_data = [cloud_data[i] for i in filtered_data_idx]

    # Check empty
    if len(cloud_data)==0:
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==np.float32: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data]
        # combine
        buffer.points = o3d.utility.Vector3dVector(np.array(xyz))
        buffer.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        pass
