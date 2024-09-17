import numpy as np
import pickle
import open3d as o3d
import os
import copy 

# Load the transform
with open(os.path.expanduser("~")+"/SKEL_WS/ros2_ws/transform.pkl", 'rb') as f:
    transform = pickle.load(f)

# Load the visual and point cloud objects
vf = o3d.io.read_triangle_mesh(os.path.expanduser("~")+"/SKEL_WS/ros2_ws/final_vf2.obj")
pc = o3d.io.read_point_cloud(os.path.expanduser("~")+"/SKEL_WS/ros2_ws/point_cloud.ply")
skel = o3d.io.read_triangle_mesh(os.path.expanduser("~")+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj")
# Create reference frames
reference_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
reference_frame2 = copy.deepcopy(reference_frame1)
reference_frame2.translate([0,0,0.2])
reference_frame2.scale(0.8, center=[0,0,0])

# Function to convert from LH Y-up to RH Z-up
def to_RH_Z_UP(pc):
    if isinstance(pc, o3d.geometry.TriangleMesh):
        vertices = np.asarray(pc.vertices)
    elif isinstance(pc, o3d.geometry.PointCloud):
        vertices = np.asarray(pc.points)
    
    # Swapping axes: LH (X, Y, Z) -> RH (X, Z, -Y)
    for vertex in vertices:
        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        vertex[0] = z     # X stays the same
        vertex[1] = -x     # Y becomes Z
        vertex[2] = y    # Z becomes -Y

    # Update the vertices or points based on the object type
    if isinstance(pc, o3d.geometry.TriangleMesh):
        pc.vertices = o3d.utility.Vector3dVector(vertices)
    elif isinstance(pc, o3d.geometry.PointCloud):
        pc.points = o3d.utility.Vector3dVector(vertices)
    
    return pc



# First convert the point cloud and reference frame to RH Z-up
pc = to_RH_Z_UP(pc)
reference_frame2 = to_RH_Z_UP(reference_frame2)

# rotate 180 deg on Z axis

# Then apply the transformation matrix
pc.transform(transform)
# reference_frame2.transform(transform)

vf.paint_uniform_color([1.0, 0,0])
# skel.transform(transform)
# Visualize the result

# rotate 180 deg on Z axis
def mirror_x(pc):
    if isinstance(pc, o3d.geometry.TriangleMesh):
        vertices = np.asarray(pc.vertices)
    elif isinstance(pc, o3d.geometry.PointCloud):
        vertices = np.asarray(pc.points)
    
    # Swapping axes: LH (X, Y, Z) -> RH (X, Z, -Y)
    for vertex in vertices:
        vertex[0] = -vertex[0]
    # Update the vertices or points based on the object type
    if isinstance(pc, o3d.geometry.TriangleMesh):
        pc.vertices = o3d.utility.Vector3dVector(vertices)

    elif isinstance(pc, o3d.geometry.PointCloud):
        pc.points = o3d.utility.Vector3dVector(vertices)
    return pc

vf = mirror_x(vf)
pc = mirror_x(pc)

vf_new = o3d.t.geometry.TriangleMesh.from_legacy(vf)
pc_new = o3d.t.geometry.PointCloud.from_legacy(pc)

o3d.visualization.draw([vf_new, pc_new])
