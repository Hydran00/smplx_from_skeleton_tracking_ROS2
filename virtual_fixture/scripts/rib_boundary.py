import open3d as o3d
import numpy as np
import math_utils
import trimesh
from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt

# read 
mesh = o3d.io.read_triangle_mesh("rib_proj.obj")
mesh.remove_duplicated_vertices()

# not aligned bbox
bbox = mesh.get_oriented_bounding_box()
bbox.color = (0, 1, 0)
# sphere.translate([bbox
reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
bbox_center = bbox.center
bbox_extent = bbox.extent


# get bbox orientation
rot_mat = bbox.R


sphere_tr = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
sphere_tr.compute_vertex_normals()
corner_tr = rot_mat @ np.array([bbox_extent[0], bbox_extent[1], bbox_extent[2]]) 
corner_tr = corner_tr/2 + bbox_center
sphere_tr.translate(corner_tr)
sphere_tr.paint_uniform_color([1,0,0])

sphere_br = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
sphere_br.compute_vertex_normals()
corner_br = rot_mat @ np.array([bbox_extent[0], bbox_extent[1], -bbox_extent[2]])
corner_br = corner_br/2 + bbox_center
sphere_br.translate(corner_br)
sphere_br.paint_uniform_color([0,1,0])

sphere_tl = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
sphere_tl.compute_vertex_normals()
corner_tl = rot_mat @ np.array([-bbox_extent[0], bbox_extent[1], bbox_extent[2]])
corner_tl = corner_tl/2 + bbox_center
sphere_tl.translate(corner_tl)
sphere_tl.paint_uniform_color([0,0,1])

o3d.visualization.draw_geometries([mesh, bbox, sphere_tr, sphere_br, sphere_tl])

border_faces = []
locations = []

# compute faces with at most two neighbors 
for i,face in enumerate(mesh.triangles):
    v0, v1, v2 = face
    matches1 = np.isin(mesh.triangles, [v0,v1])
    matches1[i] = [False, False, False]
    matches1 = np.sum(matches1, axis=1) == 2
    # print(matches1)
    matches2 = np.isin(mesh.triangles, [v1,v2])
    matches2[i] = [False, False, False]
    matches2 = np.sum(matches2, axis=1) == 2
    # print(matches2)
    matches3 = np.isin(mesh.triangles, [v0,v2])
    matches3[i] = [False, False, False]
    matches3 = np.sum(matches3, axis=1) == 2
    print("Face i: ",np.sum(matches1) + np.sum(matches2) + np.sum(matches3))
    print(np.sum(matches1) + np.sum(matches2) + np.sum(matches3))
    if np.sum(matches1) + np.sum(matches2) + np.sum(matches3) <= 2:
        free_edge = []
        if(np.sum(matches1) == 0):
           free_edge.append(0) # V0V1
        if(np.sum(matches2) == 0):
           free_edge.append(1) # V1V2
        if(np.sum(matches3) == 0):
           free_edge.append(2) # V0V2
        border_faces.append((face,free_edge))


# point plane distance, plane is defined by the normal and a point
# Function to compute the distance from a point to a plane
def point_plane_distance(point, normal, point_plane):
    return np.dot(normal, point - point_plane)

plane_up_normal = corner_br - corner_tr
plane_up_normal /= np.linalg.norm(plane_up_normal)
plane_up_point = corner_tr
plane_down_normal = -plane_up_normal
plane_down_point = corner_br

spheres = []
edges = []
# Find the closest plane (up or down) for each vertex
for (face, free_edge) in border_faces:
    for vertex in face[free_edge]:
        point = mesh.vertices[vertex]
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.compute_vertex_normals()
        sphere.translate(point)
        
        distance_up = point_plane_distance(point, plane_up_normal, plane_up_point)
        distance_down = point_plane_distance(point, plane_down_normal, plane_down_point)
        
        # Take the absolute value for distance comparison
        abs_distance_up = abs(distance_up)
        abs_distance_down = abs(distance_down)
        print("Point: ",point)
        print("Distance up: ", distance_up)
        print("Distance down: ", distance_down)
        
        if abs_distance_down > abs_distance_up:
            if(abs_distance_up)<0.005:
                sphere.paint_uniform_color([1, 0, 0])
                spheres.append(sphere)
                edges.append([point,0])
        else:
            if(abs_distance_down)<0.005:
                sphere.paint_uniform_color([0, 1, 0])
                spheres.append(sphere)
                edges.append([point,1])        
    
o3d.visualization.draw_geometries([mesh, bbox, sphere_tr, sphere_br, sphere_tl] + spheres)








# # Step 1: Define the spline
# control_points = np.array([
#     [0, 0, 0],
#     [1, 2, 1],
#     [2, 0, 2],
#     [3, -1, 3],
#     [4, 0, 4]
# ])

# tck, u = splprep(control_points.T, s=0)
# u_fine = np.linspace(0, 1, 50)
# spline_points = np.array(splev(u_fine, tck)).T

# tangents = np.gradient(spline_points, axis=0)
# tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

# # Step 2: Define the ellipse
# def ellipse(a, b, n_points=30):
#     t = np.linspace(0, 2*np.pi, n_points)
#     x = a * np.cos(t)
#     y = b * np.sin(t)
#     z = np.zeros_like(x)  # Ellipse lies initially in the XY plane
#     return np.stack((x, y, z), axis=-1)

# a = 0.1  # Semi-major axis
# b = 0.1  # Semi-minor axis
# n_ellipse_points = 30

# # Step 3: Sweep the ellipse along the spline
# vertices = []
# faces = []

# for i, (point, tangent) in enumerate(zip(spline_points, tangents)):
#     # Compute a perpendicular vector to create the local coordinate system
#     if np.allclose(tangent, [0, 0, 1]):
#         normal = np.array([0, 1, 0])
#     else:
#         normal = np.array([0, 0, 1])
    
#     binormal = np.cross(tangent, normal)
#     binormal /= np.linalg.norm(binormal)
#     normal = np.cross(binormal, tangent)
    
#     # Create the rotation matrix from local to global coordinates
#     R = np.array([binormal, normal, tangent]).T
    
#     # Generate ellipse points in local coordinates
#     ellipse_points = ellipse(a, b)
    
#     # Apply the rotation matrix to align with the spline's tangent
#     ellipse_points_3d = np.dot(ellipse_points, R.T)
    
#     # Translate the ellipse to the current point on the spline
#     ellipse_points_3d += point
    
#     vertices.extend(ellipse_points_3d)
    
#     if i > 0:
#         n = len(vertices)
#         for j in range(n_ellipse_points):
#             faces.append([
#                 n - 2 * n_ellipse_points + j,
#                 n - 2 * n_ellipse_points + (j + 1) % n_ellipse_points,
#                 n - n_ellipse_points + j
#             ])
#             faces.append([
#                 n - n_ellipse_points + j,
#                 n - 2 * n_ellipse_points + (j + 1) % n_ellipse_points,
#                 n - n_ellipse_points + (j + 1) % n_ellipse_points
#             ])

# vertices = np.array(vertices)
# faces = np.array(faces)

# # Step 4: Create the mesh
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# # Step 5: Visualize the mesh
# mesh.show()

# # Optional: Plot using Matplotlib for better customization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='white', edgecolor='black')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()


# # export the mesh to a file
# mesh.export('swept_mesh.obj')

