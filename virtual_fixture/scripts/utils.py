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

def compute_torax_projection(mesh):
    """
    Computes the projection of the SKEL torax to the SMPL mesh
    """

    humanoid = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    skel_path =os.path.expanduser('~')+"/SKEL_WS/SKEL/output/smpl_fit/smpl_fit_skel.obj"
    skel_model = o3d.io.read_triangle_mesh(skel_path)
    faces_list_file_path = get_package_share_directory("virtual_fixture")+ '/skel_regions/full_torax.txt'
    projection_method = "radial" # "linear" or "radial"
    
    with open(faces_list_file_path, 'r') as file:
        lines = file.readlines()
        # lines are in the format a b c d e and I want e
        skel_faces = [int(line.split()[4]) for line in lines]
    
    skel_center_vertex_id = 25736

    skel_model_new  = o3d.t.geometry.TriangleMesh.from_legacy(skel_model)
    color_faces(skel_model_new, skel_faces, [1.0, 0.0, 0.0])
    o3d.visualization.draw([skel_model_new])

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(humanoid)
    

    skel_face_vertices_idx = np.asarray(skel_model.triangles)[skel_faces]
    skel_vertex_positions = np.asarray(skel_model.vertices)
    
    # compute the mean of every vertex in the skel faces
    skel_center = np.mean([skel_vertex_positions[skel_face].mean(axis=0) for skel_face in skel_face_vertices_idx],axis=0)
    skel_center = skel_vertex_positions[skel_center_vertex_id]
    
    # compute skel center taking the avg betweem min and max x and z
    print("Skel center is ",skel_center)
    smpl_faces_intersection = []
    smpl_points_intersection = []
    
    pcd = o3d.t.geometry.PointCloud()

    rays = []
    print("Preparing data for raycasting")
    for i, skel_face in enumerate(tqdm(skel_faces)):

        skel_face_vertices = []
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][0]])
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][1]])
        skel_face_vertices.append(skel_vertex_positions[skel_face_vertices_idx[i][2]])

        for j in range(3):
            direction_start = skel_center
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
            smpl_points_intersection.append(new_point1)
            smpl_points_intersection.append(new_point2)
            smpl_points_intersection.append(new_point3)
            vertices[i*3][:] = new_point1
            vertices[i*3+1][:] = new_point2
            vertices[i*3+2][:] = new_point3
            faces[i] = [i*3,i*3+1,i*3+2]

    mesh_projected.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    mesh_projected.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)
    
    vertices = mesh_projected.vertex.positions.cpu().numpy()
    triangles = mesh_projected.triangle.indices.cpu().numpy()
    def triangle_area(v1, v2, v3):
        return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        
    # Set the area threshold
    area_threshold = 0.0005
    edge_threshold = 0.025

    # Find the triangles that have an area below the threshold
    valid_triangles = []
    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        area = triangle_area(v1, v2, v3)
        if area <= area_threshold:
            # ensure that edges are smaller than 0.02 meter
            if (np.linalg.norm(v1-v2) < edge_threshold) and (np.linalg.norm(v2-v3) < edge_threshold) and (np.linalg.norm(v1-v3) < edge_threshold):
                valid_triangles.append(tri)
         
    # Create a new mesh with the filtered triangles
    valid_triangles = np.array(valid_triangles)
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh = o3d.t.geometry.TriangleMesh.from_legacy(filtered_mesh)
    filtered_mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    filtered_mesh.triangle.indices = o3d.core.Tensor(valid_triangles, dtype=o3d.core.Dtype.Int32)
    # filtered_mesh.compute_vertex_normals()
    # # filtered_mesh.paint_uniform_color([1, 0, 0])
    triangle_colors = np.ones((len(filtered_mesh.triangle.indices), 3))
    # triangle_colors[::3] = [1, 0, 0]
    for i in range(0,len(triangle_colors)):
        triangle_colors[i] = [1, 0, 0]
    filtered_mesh.triangle.colors = o3d.core.Tensor(triangle_colors, dtype=o3d.core.float32)
    filtered_mesh.compute_vertex_normals()
    # o3d.visualization.draw([filtered_mesh])
    # filters out nans
    smpl_points_intersection = [point for point in smpl_points_intersection if not np.isnan(point).any()]
    
    pcd.point.positions = o3d.core.Tensor(smpl_points_intersection, dtype=o3d.core.Dtype.Float32)
    pcd.point.colors = o3d.core.Tensor(np.zeros((len(smpl_points_intersection), 3)), dtype=o3d.core.Dtype.Float32)
    pcd.point.colors[:, 0] = 1.0

    geometries = [
    {
        "name": "humanoid",
        "geometry": humanoid,
        "material": mat_skin
    },
    {
        "name": "skel",
        "geometry": filtered_mesh,
        # "material": mat_sphere_transparent
    },
    # {
    #     "name": "pcd",
    #     "geometry": pcd,
    #     # "material": mat_sphere_transparent
    # },
    {
        "name": "mesh center",
        "geometry": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=skel_center),
    }
    ]

    humanoid.compute_vertex_normals()
    # o3d.visualization.draw(geometries)
    # dump pointcloud to file
    pcd_legacy = pcd.to_legacy()
    o3d.io.write_point_cloud("pcd.ply", pcd_legacy, write_ascii=True) 
    # dump mesh to file
    print("Filling holes and saving the mesh")
    filtered_mesh.fill_holes(hole_size=0.02)
    print("Done")

    return filtered_mesh

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

def publish_mesh(publisher,path, id, rgba=[1.0, 0.0, 0.0, 0.6]):
    """
    Publish a mesh to the Rviz visualization.
    """
    marker = Marker()
    # clear marker with id 0
    marker.id = id
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
    print("Publishing mesh with a : ",marker.color.a, " and path ", marker.mesh_resource)
    publisher.publish(marker)
    
def get_flat_surface_point_cloud(points_num,spacing=0.1):
    # Parameters
    num_points_per_side = int(np.sqrt(points_num))  # Assuming a square grid
    side_length = spacing * (num_points_per_side - 1)

    # Generate the points on the XY plane with Z = 0
    x = np.linspace(0, side_length, num_points_per_side)
    y = np.linspace(0, side_length, num_points_per_side)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)  # Flat surface, Z = 0

    # Combine into a single array of points
    points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    # Create a PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Assign the points to the point cloud
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Optionally, assign colors to the points (e.g., all points white)
    colors = np.ones_like(points)  # White color
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud