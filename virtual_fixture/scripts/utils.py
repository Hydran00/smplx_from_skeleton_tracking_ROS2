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


# LEFT_PARAVERTEBRAL_LINE_SKEL_FACES = [52835, 55372,56561,54119,52445, 48009,46624,43131,22791,24779,22158,25637]


import numpy as np
def to_z_up(mesh):
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    mesh.rotate(R, center=[0, 0, 0])
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    # mesh.rotate(R, center=[0, 0, 0])  


def color_faces(mesh, faces, color):
    """
    Color the faces of a mesh with a specific color.
    """
    # Initialize triangle colors
    triangle_colors = np.ones((len(mesh.triangle.indices), 3))

    # Color specific faces red
    for face in faces:
        if face < len(triangle_colors):
            triangle_colors[face] = color

    # Set the colors to the mesh
    mesh.triangle.colors = o3d.core.Tensor(triangle_colors, dtype=o3d.core.float32)


import open3d as o3d
import os
import time
from materials import mat_sphere_transparent, mat_skin

def compute_torax_projection(mesh):
    """
    Computes the projection of the SKEL torax to the SMPL mesh
    """
    humanoid = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    skel_path =os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj"
    skel_model = o3d.io.read_triangle_mesh(skel_path)
    faces_list_file_path = os.path.expanduser('~')+"/Desktop/selected_torax2.txt"

    with open(faces_list_file_path, 'r') as file:
        lines = file.readlines()
        # lines are in the format a b c d e and I want e
        skel_faces = [int(line.split()[4]) for line in lines]

    skel_model_new  = o3d.t.geometry.TriangleMesh.from_legacy(skel_model)
    color_faces(skel_model_new, skel_faces, [1.0, 0.0, 0.0])
    o3d.visualization.draw([skel_model_new])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(humanoid)
    

    skel_face_vertices_idx = np.asarray(skel_model.triangles)[skel_faces]
    skel_vertex_positions = np.asarray(skel_model.vertices)
    smpl_faces_intersection = []
    smpl_points_intersection = []
    
    # geometries = []
    # # get 100 random faces

    # for i, skel_face in enumerate(skel_faces[:100]):
    #     # add a little sphere at the face center
    #     skel_face_vertices = skel_face_vertices_idx[i]
    #     skel_face_center = skel_vertex_positions[skel_face_vertices].mean(axis=0)
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    #     sphere.translate(skel_face_center, relative=False)
    #     geometries.append(sphere)
    #     print("Adding sphere number ",i)
    # geometries.append(skel_model)
    # o3d.visualization.draw(geometries)
    pcd = o3d.t.geometry.PointCloud()

    for i, skel_face in enumerate(skel_faces):

        # get xyz of the face center
        skel_face_vertices = skel_face_vertices_idx[i]
        skel_face_center = skel_vertex_positions[skel_face_vertices].mean(axis=0)

        # get the local z axis of the human mesh
        rotm = mesh.get_rotation_matrix_from_xyz((0, 0, 0))
        


        
        # direction is the third column of the global rotation matrix
        direction = -rotm[:, 2]

        # compute direction as the radial vector from 


        print("Computing :",skel_face_center,direction, " for face ",i,"/",len(skel_faces))
        ray = o3d.core.Tensor([ [*skel_face_center, *direction]], dtype=o3d.core.Dtype.Float32)

        # logger.info(f"start: {start}")
        # logger.info(f"direction: {direction}")
    
        ans = scene.cast_rays(ray)
        smpl_faces_intersection.append(ans['primitive_ids'][0].cpu().numpy())
        z_distance = ans['t_hit'][0].cpu().numpy()
        smpl_points_intersection.append(skel_face_center + z_distance * direction)
    # paint triangles hit by the ray red
    
    pcd.point.positions = o3d.core.Tensor(smpl_points_intersection, dtype=o3d.core.Dtype.Float32)

    print("Intersecting faces: ",smpl_faces_intersection)


    pcd.point.colors = o3d.core.Tensor(np.zeros((len(smpl_points_intersection), 3)), dtype=o3d.core.Dtype.Float32)
    pcd.point.colors[:, 0] = 1.0
    # pcd.point.positions[:, 2] -= 0.1
    print("Saving mesh")

    geometries = [{
        "name": "humanoid",
        "geometry": humanoid,
        "material": mat_skin
    },
    {
        "name": "skel",
        "geometry": skel_model,
        # "material": mat_sphere_transparent
    },
    {
        "name": "pcd",
        "geometry": pcd,
        # "material": mat_sphere_transparent
    }]

    color_faces(humanoid, smpl_faces_intersection, [1.0, 0.0, 0.0])
    o3d.visualization.draw([humanoid])
    
    humanoid.compute_vertex_normals()
    # skel_model.translate([0,0.0,-0.1],relative=True)

    o3d.visualization.draw(geometries)
    # reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=mesh.get_center())
    # o3d.visualization.draw_geometries([mesh, pcd.to_legacy(),skel_model,reference_frame], mesh_show_back_face=True)



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