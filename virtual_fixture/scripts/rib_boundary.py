import open3d as o3d
import numpy as np
# read 
mesh = o3d.io.read_triangle_mesh("rib_proj.obj")


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
sphere_tr.translate(corner_tr/2 + bbox_center)


sphere_br = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
sphere_br.compute_vertex_normals()
corner_br = rot_mat @ np.array([bbox_extent[0], bbox_extent[1], -bbox_extent[2]])
sphere_br.translate(corner_br/2 + bbox_center)


sphere_tl = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
sphere_tl.compute_vertex_normals()
corner_tl = rot_mat @ np.array([-bbox_extent[0], bbox_extent[1], bbox_extent[2]])
sphere_tl.translate(corner_tl/2 + bbox_center)

o3d.visualization.draw_geometries([mesh, bbox, sphere_tr, sphere_br, sphere_tl])

border_faces = []

# compute faces with at most two neighbors 
for i,face in enumerate(mesh.triangles):
    # get the vertices of the face
    v0, v1, v2 = face
    # get the neighbors of the face with at
    neighbors_faces_mask = np.sum( [np.any(mesh.triangles == v0, axis=1),np.any(mesh.triangles == v1, axis=1),np.any(mesh.triangles == v2, axis=1)])
    
    print(neighbors_faces_mask)
    neighbors_faces = np.where(neighbors_faces_mask == 2)[0]

    if len(neighbors_faces) < 2:
        border_faces.append(face)

border_faces = np.array(border_faces)

# create a sphere in each vertex of the border faces
spheres = []
for face in border_faces:
    for vertex in face:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.compute_vertex_normals()
        sphere.translate(mesh.vertices[vertex])
        spheres.append(sphere)

o3d.visualization.draw_geometries([mesh, bbox, sphere_tr, sphere_br, sphere_tl] + spheres)


