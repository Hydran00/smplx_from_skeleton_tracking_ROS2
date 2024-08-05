# LUNG_US_SMPL_VERTICES = {
#     "right_basal_midclavicular" : 928, #661,#929      # 13
#     "right_upper_midclavicular": 594, #595,#596       # 14
#     "left_basal_midclavicular" : 4415, #4414,#4417  # 11
#     "left_upper_midclavicular": 4082, #4084,#4085   # 12
# }
LUNG_US_SMPL_FACES = {
    # hand labelled face indices
    11 : 13345, #4414,#4417  # left_basal_midclavicular
    12: 13685, #4084,#4085   # left_upper_midclavicular
    13 : 6457, #661,#929      # right_basal_midclavicular
    14: 884, #595,#596       # right_upper_midclavicular
}
import numpy as np
def to_z_up(mesh):
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    mesh.rotate(R, center=[0, 0, 0])
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    # mesh.rotate(R, center=[0, 0, 0])  

import open3d as o3d
def get_protocol_areas_center(mesh):
    """
    Get the centers of the areas of interest in the mesh for the lung US protocol.
    """
    face_centers=[]
    # Create spheres at face centers
    for i, face in enumerate(LUNG_US_SMPL_FACES.values()):
        # Get the coordinates of the vertices of the face
        vertices_ids = mesh.triangles[face]
        vertex_coords = np.asarray(mesh.vertices)[vertices_ids]
        # Compute the center of the face
        face_center = vertex_coords.mean(axis=0)
        face_centers.append(face_center)
        # Create and color spheres
    return face_centers

def create_spherical_areas(centers, radius=0.05, color=[1, 0, 0]):
    """
    Given the XYZ areas center, create spheres at those locations.
    """
    centers = np.asarray(centers)
    # centers[:,2] += 0.1
    sphere_dict = {}
    for i, face in enumerate(LUNG_US_SMPL_FACES.values()):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        # Normals are need for the shader to work
        sphere.compute_vertex_normals()
        sphere.translate(centers[i], relative=False)
        sphere.paint_uniform_color(color)
        sphere_dict[list(LUNG_US_SMPL_FACES.keys())[i]] = sphere
    return sphere_dict

from visualization_msgs.msg import Marker

def clear_meshes(publisher):
    """
    Clear all meshes from the Rviz visualization.
    """
    marker = Marker()
    marker.action = marker.DELETEALL
    publisher.publish(marker)
def publish_mesh(publisher,path, id, rgba):
    """
    Publish a mesh to the Rviz visualization.
    """
    marker = Marker()
    # clear marker with id 0
    marker.id = id
    marker.action = marker.DELETE
    marker.type = marker.MESH_RESOURCE
    marker.header.frame_id = "base_link"
    marker.action = marker.ADD
    marker.mesh_resource = "file://"+path
    marker.mesh_use_embedded_materials = True  # Need this to use textures for mesh
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.pose.orientation.w = 1.0
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]
    marker.color.a = rgba[3]
    print("Publishing mesh with a : ",marker.color.a)
    publisher.publish(marker)