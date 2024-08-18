import numpy as np
import collections
from location import Location
import pickle
import os
from tqdm import tqdm
import open3d as o3d

class Mesh:
    def __init__(self, vertices, faces, normals, vertex_adj_list):
        self.vertices = vertices  # List of vertices (vct3 equivalent)
        self.faces = faces        # List of faces (vctInt3 equivalent)
        self.normals = normals    # List of normals (vct3 equivalent)
        self.vertex_adj_list = vertex_adj_list  # Store vertex adjacency data
        self.adjacency_dict = None  # Store adjacency data
        
        # Precompute adjacency information using Open3D
        self.precompute_adjacency()

    def precompute_adjacency(self):
        """Precompute adjacent faces for all faces and store in a dictionary using Open3D's adjacency list."""
        adjacency_list = self.vertex_adj_list
        adjacency_dict = {}
        if os.path.exists('adjacency_dict.pkl'):
            with open('adjacency_dict.pkl', 'rb') as f:
                self.adjacency_dict = pickle.load(f)
                return
        # For each face, find adjacent faces for all locations
        for face_idx, face in enumerate(tqdm(self.faces)):
            for location in Location:
                adjacent_faces = self.get_adjacent_faces(face_idx, location)
                adjacency_dict[(face_idx, location)] = adjacent_faces
        with open('adjacency_dict.pkl', 'wb') as f:
            pickle.dump(adjacency_dict, f)
        print(adjacency_dict)
        self.adjacency_dict = adjacency_dict

    def get_adjacent_faces(self, face_index, location):
        """Find adjacent faces for a specific face and location using the adjacency list."""
        face_vertices = self.faces[face_index]
        
        # Identify vertices based on the location
        if location == Location.V1:
            vertices_to_query = [face_vertices[0]]
        elif location == Location.V2:
            vertices_to_query = [face_vertices[1]]
        elif location == Location.V3:
            vertices_to_query = [face_vertices[2]]
        elif location == Location.V1V2:
            vertices_to_query = [face_vertices[0], face_vertices[1]]
        elif location == Location.V1V3:
            vertices_to_query = [face_vertices[0], face_vertices[2]]
        elif location == Location.V2V3:
            vertices_to_query = [face_vertices[1], face_vertices[2]]
        elif location == Location.IN:
            return []  # No specific edge to query; return empty
        
        # Find adjacent faces based on the adjacency of the vertices
        adjacent_faces = set()
        
        # Filter faces that contain **all** vertices in `vertices_to_query`
        matches = np.isin(self.faces, vertices_to_query)
        # Only retain faces that have all the queried vertices
        valid_faces = np.sum(matches, axis=1) == len(vertices_to_query)
        
        # Get the indices of the valid adjacent faces
        adj_faces = np.where(valid_faces)[0]
        
        for adj_face in adj_faces:
            if adj_face != face_index:
                adjacent_faces.add(adj_face)
        # print("Found adjacent faces:", adjacent_faces)
        return list(adjacent_faces)
    
    def is_locally_convex(self, idx, neighbor, location):
        """Check if a triangle is locally convex based on the location of the closest point."""
        return self.check_convexity(idx, neighbor, location)
    
    def is_locally_concave(self, idx, neighbor, location):
        """Check if a triangle is locally concave based on the location of the closest point."""
        return not self.is_locally_convex(idx, neighbor, location)

    def check_convexity(self, idx, idxNeighbor, location):
        """
        Check if the edge between the triangle at index `idx` and its neighbor at index `idxNeighbor` is convex.
        
        Args:
            idx (int): Index of the current triangle.
            idxNeighbor (int): Index of the neighboring triangle.
            location (Location): Location enum representing the shared edge/vertex.

        Returns:
            bool: True if the edge is convex, False otherwise.
            
        The local convexity is found by 
                        {
            convexity =     1, if N T i,av > 0 
                            0, if N T i,av < 0 
                        }
            where v is the one of the non-shared edges originating from the shared vertex of the neighboring triangle

        """
        # Get the current triangle vertices
        current_triangle = self.faces[idx]
        
        # Compute vectors based on the location
        if location == Location.V1V2:
            vec1 = self.vertices[current_triangle[2]] - self.vertices[current_triangle[0]]
            vec2 = self.vertices[current_triangle[2]] - self.vertices[current_triangle[1]]
        elif location == Location.V1V3:
            vec1 = self.vertices[current_triangle[1]] - self.vertices[current_triangle[0]]
            vec2 = self.vertices[current_triangle[1]] - self.vertices[current_triangle[2]]
        elif location == Location.V2V3:
            vec1 = self.vertices[current_triangle[0]] - self.vertices[current_triangle[1]]
            vec2 = self.vertices[current_triangle[0]] - self.vertices[current_triangle[2]]
            """
            In theory, when the CPi falls on a vertex, there could be many triangles intersecting this point.
            However, it is sufficient to only consider any one of the two adjacent triangles who shares an edge
            and CPi with Ti. The rest of the triangles whose closet points also fall on this vertex can
            be safely discarded as they will result in the same constraint.
            """
        elif location == Location.V1:
            vec1 = self.vertices[current_triangle[1]] - self.vertices[current_triangle[0]]
            vec2 = self.vertices[current_triangle[2]] - self.vertices[current_triangle[0]]
        elif location == Location.V2:
            vec1 = self.vertices[current_triangle[0]] - self.vertices[current_triangle[1]]
            vec2 = self.vertices[current_triangle[2]] - self.vertices[current_triangle[1]]
        elif location == Location.V3:
            vec1 = self.vertices[current_triangle[0]] - self.vertices[current_triangle[2]]
            vec2 = self.vertices[current_triangle[1]] - self.vertices[current_triangle[2]]
        else:
            return False  # Invalid location
        
        # Get the normal of the neighboring triangle
        neighbor_normal = self.normals[idxNeighbor]
        
        # Check convexity by computing dot products
        if np.dot(vec1, neighbor_normal) > 0.0 or np.dot(vec2, neighbor_normal) > 0.0:
            return True
        elif np.dot(vec1, neighbor_normal) < 0.0 or np.dot(vec2, neighbor_normal) < 0.0:
            return False
        
        return False
    
def color_mesh_faces(mesh, face_index, adjacent_faces):
    """ Test method
    Colors the mesh faces with a specific color for the given face and its adjacent faces.
    
    Parameters:
    - mesh: The mesh object.
    - face_index: The index of the face to be highlighted.
    - adjacent_faces: The list of adjacent face indices.
    """
    print("Query face index:", face_index)
    print("Adjacent faces:", adjacent_faces)
    # Create a copy of the mesh to work on
    mesh_copy = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    mesh_new = o3d.t.geometry.TriangleMesh.from_legacy(mesh_copy)
    mesh_new.compute_vertex_normals()
    triangle_colors = np.ones((len(mesh_new.triangle.indices), 3))
    # Color specific faces red
    triangle_colors[face_index] = [1, 0, 0]
    for face in adjacent_faces:
        triangle_colors[face] = [0, 0, 1]

    # Set the colors to the mesh
    mesh_new.triangle.colors = o3d.core.Tensor(triangle_colors, dtype=o3d.core.float32)
    
    return mesh_new


def compute_triangle_xfm(v1, v2, v3):
    # Compute the y-axis as the normalized vector from v1 to v2
    yaxis = (v2 - v1) / np.linalg.norm(v2 - v1)
    
    # Compute the x-axis using the cross product of (v3 - v1) and y-axis, then normalize
    zaxis = np.cross((v3 - v1) / np.linalg.norm(v3 - v1), yaxis) 
    zaxis = zaxis / np.linalg.norm(zaxis)
    
    # Compute the z-axis using the cross product of y-axis and zaxis, then normalize
    xaxis = np.cross(yaxis, zaxis)
    xaxis = xaxis / np.linalg.norm(xaxis)

    # Create the rotation matrix
    R = np.vstack([xaxis, yaxis, zaxis])

    # Compute the translation component by rotating v1 and taking the negative
    T = -np.dot(R, v1)

    # Construct the transformation matrix (4x4 homogeneous transformation)
    xfm = np.eye(4)
    xfm[:3, :3] = R
    xfm[:3, 3] = T

    return xfm

def run_test():
    # dataset = o3d.data.BunnyMesh()
    # data = o3d.io.read_triangle_mesh(dataset.path)
    data = o3d.io.read_triangle_mesh(os.path.expanduser('~') + '/SKEL_WS/ros2_ws/bunny.ply')
    data.compute_vertex_normals()
    data.compute_adjacency_list()
    mesh = Mesh(
        vertices=np.asarray(data.vertices),
        faces=np.asarray(data.triangles),
        normals=np.asarray(data.triangle_normals),
        vertex_adj_list=np.asarray(data.adjacency_list)
    )
    face_index = 10
    location = Location.V1V2
    adjacent_faces = mesh.get_adjacent_faces(face_index, location)
    colored_mesh = color_mesh_faces(mesh, face_index, adjacent_faces)
    o3d.visualization.draw([colored_mesh])
    
    location = Location.V1V3
    adjacent_faces = mesh.get_adjacent_faces(face_index, location)
    colored_mesh = color_mesh_faces(mesh, face_index, adjacent_faces)
    o3d.visualization.draw([colored_mesh])
    
    location = Location.V2V3
    adjacent_faces = mesh.get_adjacent_faces(face_index, location)
    colored_mesh = color_mesh_faces(mesh, face_index, adjacent_faces)
    o3d.visualization.draw([colored_mesh])
    
    location = Location.V1
    adjacent_faces = mesh.get_adjacent_faces(face_index, location)
    colored_mesh = color_mesh_faces(mesh, face_index, adjacent_faces)
    o3d.visualization.draw([colored_mesh])
    
    location = Location.V2
    adjacent_faces = mesh.get_adjacent_faces(face_index, location)
    colored_mesh = color_mesh_faces(mesh, face_index, adjacent_faces)
    o3d.visualization.draw([colored_mesh])
    
    location = Location.V3
    adjacent_faces = mesh.get_adjacent_faces(face_index, location)
    colored_mesh = color_mesh_faces(mesh, face_index, adjacent_faces)
    o3d.visualization.draw([colored_mesh])
    
if __name__ == "__main__":
    run_test()
