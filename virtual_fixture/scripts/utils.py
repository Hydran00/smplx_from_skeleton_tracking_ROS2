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
def mirror_normals(mesh):
    """
    Mirror the normals of a mesh.
    """
    normals = np.asarray(mesh.triangle_normals)
    mirrored_normals = normals * -1
    mesh.triangle_normals = o3d.utility.Vector3dVector(mirrored_normals)

def compute_torax_projection(mesh):
    """
    Computes the projection of the SKEL torax to the SMPL mesh
    """

    humanoid = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    skel_path =os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj"
    skel_model = o3d.io.read_triangle_mesh(skel_path)
    faces_list_file_path = get_package_share_directory("virtual_fixture")+ '/skel_regions/full_torax.txt'
    ribs_path_prefix = os.path.expanduser('~')+"/SKEL_WS/ros2_ws/src/smplx_from_skeleton_tracking_ROS2/virtual_fixture/skel_regions/rib"
    available_ribs = [(2,'r')] #,(3,'r'),(4,'r'),(5,'r'),(6,'r'),(7,'r'),(2,'l'),(3,'l'),(4,'l'),(5,'l'),(6,'l'),(7,'l')]
    
    projection_method = "radial" # "linear" or "radial"
    
    rib_l_faces = []
    
    # list of faces for the down line of the available ribs
    ribs_down_faces = []
    # list of faces for the upper line of the available ribs
    ribs_up_faces = []
    # consider the top and bottom vertices of each face on Y axis
    ribs_vertices_up = [ [] for i in range(len(available_ribs))]
    ribs_vertices_bottom = [ [] for i in range(len(available_ribs))]
    
    skel_vertices_pos = np.asarray(skel_model.vertices) 
    
    with open(faces_list_file_path, 'r') as file:
        lines = file.readlines()
        # lines are in the format a b c d e and I want e
        skel_faces = [int(line.split()[4]) for line in lines]
    for i, rib in enumerate(available_ribs):
        path_down = ribs_path_prefix+str(rib[0])+"_"+rib[1]+"_down.txt"
        with open(path_down, 'r') as file:
            lines = file.readlines()
            # lines are in the format a b c d e and I want e
            ribs_down_faces.append([int(line.split()[4]) for line in lines])
            for line in lines:
                b_idx = int(line.split()[1])
                c_idx = int(line.split()[2])
                d_idx = int(line.split()[3])
                b = skel_vertices_pos[b_idx]
                c = skel_vertices_pos[c_idx]
                d = skel_vertices_pos[d_idx]
                max_y_idx = 0 #np.argmin([b[1],c[1],d[1]])
                ribs_vertices_bottom[i].append([b_idx,c_idx,d_idx][max_y_idx])
        
        path_up = ribs_path_prefix+str(rib[0])+"_"+rib[1]+"_up.txt"
        with open(path_up, 'r') as file:
            lines = file.readlines()
            # lines are in the format a b c d e and I want e
            ribs_up_faces.append([int(line.split()[4]) for line in lines])
            for line in lines:
                b_idx = int(line.split()[1])
                c_idx = int(line.split()[2])
                d_idx = int(line.split()[3])
                b = skel_vertices_pos[b_idx]
                c = skel_vertices_pos[c_idx]
                d = skel_vertices_pos[d_idx]
                max_y_idx = 0# np.argmax([b[1],c[1],d[1]])
                ribs_vertices_up[i].append([b_idx,c_idx,d_idx][max_y_idx])
    
    
    # remove duplicates
    for i in range(len(ribs_vertices_bottom)):
        ribs_vertices_bottom[i] = list(set(ribs_vertices_bottom[i]))
        ribs_vertices_up[i] = list(set(ribs_vertices_up[i]))
    
    print("Ribs down vertices: ",ribs_vertices_bottom)
    print("Ribs up vertices: ",ribs_vertices_up)
    
    skel_center_vertex_id = 25736

    skel_model_new  = o3d.t.geometry.TriangleMesh.from_legacy(skel_model)
    # color_faces(skel_model_new, skel_faces, [1.0, 0.0, 0.0])
    # color_faces(skel_model_new, ribs_up_faces[10], [1.0, 0.0, 0.0])

    # rib_model_new  = o3d.t.geometry.TriangleMesh.from_legacy(skel_model)
    # color_faces(rib_model_new, rib_l_faces, [0.0, 1.0, 0.0])

    # skel_model_new.compute_vertex_normals()
    
    # reference_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0,0,0])
    
    # o3d.visualization.draw([skel_model_new,reference_frame])
    # o3d.visualization.draw([rib_model_new])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(humanoid)
    
    # rib cage faces and vertices
    # skel_face_vertices_idx = np.asarray(skel_model.triangles)[skel_faces]
    
    # compute the mean of every vertex in the skel faces
    # skel_center = np.mean([skel_vertex_positions[skel_face].mean(axis=0) for skel_face in skel_face_vertices_idx],axis=0)
    skel_center = skel_vertices_pos[skel_center_vertex_id]
    
    # compute skel center taking the avg betweem min and max x and z
    print("Skel center is ",skel_center)

    rays = []
    print("Preparing data for raycasting")
    
    for rib_idx in tqdm(range(len(ribs_vertices_bottom))):
        for i, vertex_idx in enumerate(ribs_vertices_bottom[rib_idx]):
            vertex = skel_vertices_pos[vertex_idx]

            direction_start = skel_center
            direction_start[1] = vertex[1]
            direction_end = vertex
            direction = direction_end - direction_start
            direction = direction / np.linalg.norm(direction)
            rays.append([*vertex,*direction])

        for i, vertex_idx in enumerate(ribs_vertices_up[rib_idx]):

            vertex = skel_vertices_pos[vertex_idx]

            direction_start = skel_center
            direction_start[1] = vertex[1]
            direction_end = vertex
            direction = direction_end - direction_start
            direction = direction / np.linalg.norm(direction)
            rays.append([*vertex,*direction])

    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    ans = scene.cast_rays(rays)
    distances = ans['t_hit'].cpu().numpy()
    # smpl_faces_intersection.append(ans['primitive_ids'][0].cpu().numpy())
    rays = rays.cpu().numpy()
    
    ribs_vertices_up_projected = [[] for i in range(len(ribs_vertices_up))]
    ribs_vertices_down_projected = [[] for i in range(len(ribs_vertices_bottom))]

    print("RIB l cage: ",rib_l_faces)
    ray_idx = 0
    sphere_list = []
    for rib_idx in range(len(ribs_vertices_bottom)):
        for vertex_idx in ribs_vertices_bottom[rib_idx]:
            vertex_bottom = rays[ray_idx][:3]
            direction_bottom = rays[ray_idx][3:]
            distance_bottom = distances[ray_idx]
            ray_idx += 1
            if distance_bottom != np.inf :
                projected_vertex = vertex_bottom + distance_bottom * direction_bottom
                ribs_vertices_down_projected[rib_idx].append(projected_vertex)
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(projected_vertex)
                sphere.paint_uniform_color([0.0, 1.0, 0.0])
                sphere_list.append(sphere)
        for vertex_idx in ribs_vertices_up[rib_idx]:
            vertex_up = rays[ray_idx][:3]
            direction_up = rays[ray_idx][3:]
            distance_up = distances[ray_idx]
            ray_idx += 1
            if distance_up != np.inf :
                projected_vertex = vertex_up + distance_up * direction_up
                ribs_vertices_up_projected[rib_idx].append(projected_vertex)
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(projected_vertex)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                sphere_list.append(sphere)
    
    o3d.visualization.draw_geometries(sphere_list + [humanoid.to_legacy()])
    print("Ray idx: ",ray_idx, " vs ",len(rays))
    final_vf = o3d.t.geometry.TriangleMesh()
    for rib_idx in range(len(ribs_vertices_bottom)):
        print("RIB ",i)
        pc_down = o3d.geometry.PointCloud()
        print("down Adding: ",np.array(ribs_vertices_down_projected[rib_idx]))
        pc_down.points = o3d.utility.Vector3dVector(np.array(ribs_vertices_down_projected[rib_idx]))
        pc_up = o3d.geometry.PointCloud()
        print("up Adding: ",np.array(ribs_vertices_up_projected[rib_idx]))
        pc_up.points = o3d.utility.Vector3dVector(np.array(ribs_vertices_up_projected[rib_idx]))
        mesh = retrieve_vf_from_rib(pc_down,pc_up, skel_center)
        # o3d.visualization.draw_geometries([mesh])
        mesh.compute_triangle_normals()
        mesh.orient_triangles()
        mirror_normals(mesh)
        if i == 0:
            final_vf = mesh
            continue
        final_vf += mesh
    final_vf.paint_uniform_color([1.0, 0.0, 0.0])
    final_vf_new = o3d.t.geometry.TriangleMesh.from_legacy(final_vf)
    # o3d.visualization.draw_geometries([final_vf])
    o3d.visualization.draw([final_vf_new])
    o3d.visualization.draw_geometries([final_vf, skel_model,humanoid.to_legacy()])
    return final_vf


    
    # vertices = mesh_projected.vertex.positions.cpu().numpy()
    # triangles = mesh_projected.triangle.indices.cpu().numpy()
    # def triangle_area(v1, v2, v3):
    #     return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        
    # # Set the area threshold
    # area_threshold = 0.0005
    # edge_threshold = 0.025

    # # Find the triangles that have an area below the threshold
    # valid_triangles = []
    # for tri in triangles:
    #     v1, v2, v3 = vertices[tri]
    #     area = triangle_area(v1, v2, v3)
    #     if area <= area_threshold:
    #         # ensure that edges are smaller than 0.02 meter
    #         if (np.linalg.norm(v1-v2) < edge_threshold) and (np.linalg.norm(v2-v3) < edge_threshold) and (np.linalg.norm(v1-v3) < edge_threshold):
    #             valid_triangles.append(tri)


    # # Create a new mesh with the filtered triangles
    # valid_triangles = np.array(valid_triangles)
    # filtered_mesh = o3d.geometry.TriangleMesh()
    # filtered_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # filtered_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    # filtered_mesh.remove_duplicated_vertices()     
    # filtered_mesh = o3d.t.geometry.TriangleMesh.from_legacy(filtered_mesh)
    # # filtered_mesh.compute_vertex_normals()
    # # # filtered_mesh.paint_uniform_color([1, 0, 0])
    # triangle_colors = np.ones((len(filtered_mesh.triangle.indices), 3))
    # # triangle_colors[::3] = [1, 0, 0]
    # for i in range(0,len(triangle_colors)):
    #     triangle_colors[i] = [1, 0, 0]
    # filtered_mesh.triangle.colors = o3d.core.Tensor(triangle_colors, dtype=o3d.core.float32)
    # filtered_mesh.compute_vertex_normals()
    # # o3d.visualization.draw([filtered_mesh])
    # # filters out nans
    # smpl_points_intersection = [point for point in smpl_points_intersection if not np.isnan(point).any()]
    
    # pcd.point.positions = o3d.core.Tensor(smpl_points_intersection, dtype=o3d.core.Dtype.Float32)
    # pcd.point.colors = o3d.core.Tensor(np.zeros((len(smpl_points_intersection), 3)), dtype=o3d.core.Dtype.Float32)
    # pcd.point.colors[:, 0] = 1.0

    # geometries = [
    # {
    #     "name": "humanoid",
    #     "geometry": humanoid,
    #     "material": mat_skin
    # },
    # {
    #     "name": "skel",
    #     "geometry": filtered_mesh,
    #     # "material": mat_sphere_transparent
    # },
    # # {
    # #     "name": "pcd",
    # #     "geometry": pcd,
    # #     # "material": mat_sphere_transparent
    # # },
    # {
    #     "name": "mesh center",
    #     "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=skel_center),
    # }
    # ]

    # humanoid.compute_vertex_normals()
    # o3d.visualization.draw(geometries)
    # # dump pointcloud to file
    # pcd_legacy = pcd.to_legacy()
    # o3d.io.write_point_cloud("pcd.ply", pcd_legacy, write_ascii=True) 
    # # dump mesh to file
    # print("Filling holes and saving the mesh")
    # filtered_mesh.fill_holes(hole_size=0.02)
    # print("Done")

    # return filtered_mesh

def visualize_mesh_normals(mesh):
    mesh.compute_vertex_normals()

    # Create a LineSet for visualizing normals
    lines = []
    colors = []
    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.triangle_normals)
    # Generate lines for each triangle normal
    for i, triangle in enumerate(np.asarray(mesh.triangles)):
        v0, v1, v2 = triangle
        triangle_center = (points[v0] + points[v1] + points[v2]) / 3
        normal_start = triangle_center
        normal_end = triangle_center + normals[i] * 0.1  # Adjust length of the normal line

        lines.append([len(points) + 2*i, len(points) + 2*i + 1])
        points = np.vstack([points, normal_start, normal_end])
        colors.extend([[1, 0, 0]] * 2)  # Red color for normals
    lines = np.array(lines)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
    