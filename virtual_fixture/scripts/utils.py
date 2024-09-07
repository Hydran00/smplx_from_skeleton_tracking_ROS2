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
from tqdm import tqdm
from ament_index_python.packages import get_package_share_directory
from rib_boundary import retrieve_vf_from_rib
import pickle
def compute_torax_projection(mesh):
    """
    Computes the projection of the SKEL torax to the SMPL mesh
    """

    humanoid = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    skel_path =os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj"
    skel_model = o3d.io.read_triangle_mesh(skel_path)
    faces_list_file_path = get_package_share_directory("virtual_fixture")+ '/skel_regions/full_torax.txt'
    ribs_path_prefix = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/src/smplx_from_skeleton_tracking_ROS2/virtual_fixture/skel_regions/rib"
    params_path = os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_params.pkl"
    available_ribs = [(2,'r'),(3,'r'),(4,'r'),(5,'r'),(6,'r'),(7,'r'),(2,'l'),(3,'l'),(4,'l'),(5,'l'),(6,'l'),(7,'l')]
    
    
    with open (params_path, 'rb') as file:
        data = pickle.load(file)
        rot = data["rot"]


    rib_l_faces = []
    
    # list of faces for the down line of the available ribs
    ribs_down_faces = []
    # list of faces for the upper line of the available ribs
    ribs_up_faces = []
    with open(faces_list_file_path, 'r') as file:
        lines = file.readlines()
        # lines are in the format a b c d e and I want e
        skel_faces = [int(line.split()[4]) for line in lines]
    for rib in available_ribs:
        path_down = ribs_path_prefix+str(rib[0])+"_"+rib[1]+"_down.txt"
        with open(path_down, 'r') as file:
            lines = file.readlines()
            # lines are in the format a b c d e and I want e
            ribs_down_faces.append([int(line.split()[4]) for line in lines])
        path_up = ribs_path_prefix+str(rib[0])+"_"+rib[1]+"_up.txt"
        with open(path_up, 'r') as file:
            lines = file.readlines()
            # lines are in the format a b c d e and I want e
            ribs_up_faces.append([int(line.split()[4]) for line in lines])
        
    
    skel_center_vertex_id = 25736

    skel_model_new  = o3d.t.geometry.TriangleMesh.from_legacy(skel_model)
    color_faces(skel_model_new, skel_faces, [1.0, 0.0, 0.0])

    rib_model_new  = o3d.t.geometry.TriangleMesh.from_legacy(skel_model)
    color_faces(rib_model_new, rib_l_faces, [0.0, 1.0, 0.0])

    o3d.visualization.draw([skel_model_new])
    o3d.visualization.draw([rib_model_new])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(humanoid)
    

    skel_face_vertices_idx = np.asarray(skel_model.triangles)[skel_faces]
    skel_vertex_positions = np.asarray(skel_model.vertices)
    
    # compute the mean of every vertex in the skel faces
    skel_center = np.mean([skel_vertex_positions[skel_face].mean(axis=0) for skel_face in skel_face_vertices_idx],axis=0)
    skel_center = skel_vertex_positions[skel_center_vertex_id]
    
    # compute skel center taking the avg betweem min and max x and z
    print("Skel center is ",skel_center)

    import scipy
    transf_matrix = np.eye(4)
    transf_matrix[:3,3] = skel_center
    # convert rvec to rotation matrix
    rotmat = scipy.spatial.transform.Rotation.from_rotvec(rot).as_matrix()
    transf_matrix[:3,:3] = rotmat

    rays = []
    print("Preparing data for raycasting")
    for i, skel_face in enumerate(tqdm(skel_faces)):

        skel_face_vertices = []
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][0]])
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][1]])
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][2]])

        for j in range(3):
            direction_start = skel_center
            direction_start = skel_center
            
            vertex = skel_face_vertices[j]
            skel_center_in_mesh_ref_frame = np.linalg.inv(transf_matrix) @ np.array([*skel_center,1])
            vertex_in_mesh_ref_frame = np.linalg.inv(transf_matrix) @ np.array([*vertex,1])
            y_offset_in_mesh_ref_frame = vertex_in_mesh_ref_frame[1] - skel_center_in_mesh_ref_frame[1]
            # transform the offset to the skel frame
            y_offset = transf_matrix @ np.array([0,y_offset_in_mesh_ref_frame,0,1])
            direction_start = direction_start + y_offset[:3]

            direction_start[1] = skel_face_vertices[j][1]
            direction_end = skel_face_vertices[j]
            direction = direction_end - direction_start
            rays.append([*skel_face_vertices[j],*direction])
    
    
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    ans = scene.cast_rays(rays)
    distances = ans['t_hit'].cpu().numpy()
    # smpl_faces_intersection.append(ans['primitive_ids'][0].cpu().numpy())
    rays = rays.cpu().numpy()
    
    mesh_projected = o3d.t.geometry.TriangleMesh()
    vertices = np.zeros((len(skel_faces)*3,3))
    faces = np.zeros((len(skel_faces),3))

    ribs_down_vertices = [[] for i in range(len(ribs_down_faces))]
    ribs_up_vertices = [[] for i in range(len(ribs_up_faces))]

    print("RIB l cage: ",rib_l_faces)
    for i, skel_face in enumerate(tqdm(skel_faces)):
        skel_face_vertices = []
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][0]])
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][1]])
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][2]])

        z_distance1 = distances[i*3]
        z_distance2 = distances[i*3+1]
        z_distance3 = distances[i*3+2]
        direction1 = rays[i*3][3:]
        direction2 = rays[i*3+1][3:]
        direction3 = rays[i*3+2][3:]
        
        if z_distance1 != np.inf and z_distance2 != np.inf and z_distance3 != np.inf:
            new_point1 = skel_face_vertices[0] + z_distance1 * direction1
            new_point2 = skel_face_vertices[1] + z_distance2 * direction2
            new_point3 = skel_face_vertices[2] + z_distance3 * direction3
            vertices[i*3][:] = new_point1
            vertices[i*3+1][:] = new_point2
            vertices[i*3+2][:] = new_point3
            faces[i] = [i*3,i*3+1,i*3+2]
            
            # iter available down ribs
            for j,rib_down in enumerate(ribs_down_faces):
                if skel_face in rib_down:
                    # take just the vertices with highest y
                    points = np.array([new_point1,new_point2,new_point3])
                    ribs_down_vertices[j].append(points[np.argmax(points[:,1])])
                    break
            # iter available up ribs
            for j,rib_up in enumerate(ribs_up_faces):
                if skel_face in rib_up:
                    # take just the vertices with lowest y
                    points = np.array([new_point1,new_point2,new_point3])
                    ribs_up_vertices[j].append(points[np.argmin(points[:,1])])
                    break
    
    final_vf = o3d.t.geometry.TriangleMesh()
    for i in range(len(ribs_down_vertices)):
        print("RIB ",i)
        pc_down = o3d.geometry.PointCloud()
        pc_down.points = o3d.utility.Vector3dVector(ribs_down_vertices[i])
        pc_up = o3d.geometry.PointCloud()
        pc_up.points = o3d.utility.Vector3dVector(ribs_up_vertices[i])
        mesh = retrieve_vf_from_rib(pc_down,pc_up, skel_center, transf_matrix)
        # o3d.visualization.draw_geometries([mesh])
        if i == 0:
            final_vf = mesh
            continue
        final_vf += mesh
        
    o3d.visualization.draw_geometries([final_vf])
    return final_vf
