import numpy as np
import collections
from location import Location

class Mesh:
    def __init__(self, vertices, faces, normals):
        self.vertices = vertices  # List of vertices (vct3 equivalent)
        self.faces = faces        # List of faces (vctInt3 equivalent)
        self.normals = normals    # List of normals (vct3 equivalent)
        
        # Combined dictionary to store both edge-to-triangle mapping and face neighbors
        self.edge_map = collections.defaultdict(lambda: {"triangles": [], "neighbors": {}})

        # Initialize the edge map and find neighbors
        self.populate_edge_map()

    def vertex_to_key(self, vertex1, vertex2, triangle):
        """Create a unique key for an edge formed by two vertices in a triangle."""
        return f"{min(triangle[vertex1], triangle[vertex2])}-{max(triangle[vertex1], triangle[vertex2])}"

    def populate_edge_map(self):
        """Populate the edge_map with edges, corresponding triangles, and neighbors."""
        for face_idx, triangle in enumerate(self.faces):
            for (v1, v2) in [(0, 1), (0, 2), (1, 2)]:
                edge_key = self.vertex_to_key(v1, v2, triangle)
                self.edge_map[edge_key]["triangles"].append(face_idx)

        self.find_all_face_neighbors()

    def find_all_face_neighbors(self):
        """Find and store neighbors for all triangles."""
        for edge_key, data in self.edge_map.items():
            triangles = data["triangles"]
            if len(triangles) > 1:
                # If more than one triangle shares the same edge, they are neighbors
                for i in range(len(triangles)):
                    for j in range(i + 1, len(triangles)):
                        self.edge_map[edge_key]["neighbors"][triangles[i]] = triangles[j]
                        self.edge_map[edge_key]["neighbors"][triangles[j]] = triangles[i]

    def get_neighbors_of_face(self, face_idx, location):
        """
        Get the list of face indices corresponding to the neighbors of a given face.

        :param face_idx: The index of the face for which neighbors are needed
        :param location: The Location enum specifying which neighbors to retrieve
        :return: A list of neighboring face indices
        """
        neighboring_faces = []

        for (v1, v2) in [(0, 1), (0, 2), (1, 2)]:
            if self.is_valid_location(location, v1, v2):
                edge_key = self.vertex_to_key(v1, v2, self.faces[face_idx])
                neighbor = self.edge_map[edge_key]["neighbors"].get(face_idx)
                if neighbor is not None:
                    neighboring_faces.append(neighbor)

        return neighboring_faces

    def is_valid_location(self, location, v1, v2):
        """Helper method to check if an edge corresponds to the given Location."""
        if location == Location.V1 and v1 in [0] and v2 in [1, 2]:
            return True
        elif location == Location.V2 and v1 in [0, 1] and v2 in [1, 2]:
            return True
        elif location == Location.V3 and v1 in [0, 1] and v2 == 2:
            return True
        elif location == Location.V1V2 and v1 == 0 and v2 == 1:
            return True
        elif location == Location.V1V3 and v1 == 0 and v2 == 2:
            return True
        elif location == Location.V2V3 and v1 == 1 and v2 == 2:
            return True
        return False
    
    def is_locally_convex(self, idx, neighbor, cpLocation):
        """Check if a triangle is locally convex based on the location of the closest point."""
        return self.check_convexity(idx, neighbor, cpLocation)
    
    def is_locally_concave(self, idx, neighbor, cpLocation):
        """Check if a triangle is locally concave based on the location of the closest point."""
        return not self.is_locally_convex(idx, neighbor, cpLocation)

    def check_convexity(self, idx, idxNeighbor, location):
        vec1 = np.zeros(3)
        vec2 = np.zeros(3)
        vertexOffset = 0  # Assuming vertexOffset is 1 as in the C++ code

        if location == Location.V1V2:
            vec1 = self.vertices[faces[idx][2] - vertexOffset] - self.vertices[faces[idx][0] - vertexOffset]
            vec2 = self.vertices[faces[idx][2] - vertexOffset] - self.vertices[faces[idx][1] - vertexOffset]
        elif location == Location.V1V3:
            vec1 = self.vertices[faces[idx][1] - vertexOffset] - self.vertices[faces[idx][0] - vertexOffset]
            vec2 = self.vertices[faces[idx][1] - vertexOffset] - self.vertices[faces[idx][2] - vertexOffset]
        elif location == Location.V2V3:
            vec1 = self.vertices[faces[idx][0] - vertexOffset] - self.vertices[faces[idx][1] - vertexOffset]
            vec2 = self.vertices[faces[idx][0] - vertexOffset] - self.vertices[faces[idx][2] - vertexOffset]
        elif location == Location.V1:
            vec1 = self.vertices[faces[idx][1] - vertexOffset] - self.vertices[faces[idx][0] - vertexOffset]
            vec2 = self.vertices[faces[idx][2] - vertexOffset] - self.vertices[faces[idx][0] - vertexOffset]
        elif location == Location.V2:
            vec1 = self.vertices[faces[idx][0] - vertexOffset] - self.vertices[faces[idx][1] - vertexOffset]
            vec2 = self.vertices[faces[idx][2] - vertexOffset] - self.vertices[faces[idx][1] - vertexOffset]
        elif location == Location.V3:
            vec1 = self.vertices[faces[idx][0] - vertexOffset] - self.vertices[faces[idx][2] - vertexOffset]
            vec2 = self.vertices[faces[idx][1] - vertexOffset] - self.vertices[faces[idx][2] - vertexOffset]

        dot1 = np.dot(vec1, self.normals[idxNeighbor])
        dot2 = np.dot(vec2, self.normals[idxNeighbor])

        if dot1 > 0.0 or dot2 > 0.0:
            return True
        if dot1 < 0.0 or dot2 < 0.0:
            return False
        return False


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

def E(xy, XY, dxdy):
    """Compute the function E based on the formula provided."""
    return round6((xy[0] - XY[0]) * dxdy[1] - (xy[1] - XY[1]) * dxdy[0])


