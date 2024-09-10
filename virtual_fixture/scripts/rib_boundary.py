import open3d as o3d
import numpy as np
import math_utils
from scipy.interpolate import splev, splprep
import matplotlib.pyplot as plt

def retrieve_vf_from_rib(rib_down, rib_up, skel_center, transf_matrix):
    rib_down.remove_duplicated_points()
    rib_up.remove_duplicated_points()

    a1 = np.asarray(rib_up.points)
    a2 = np.asarray(rib_down.points)

    line_size = 10
    # Find the range of x values in a1
    min_a1_x, max_a1_x = min(a1[:, 0]), max(a1[:, 0])
    # Create an evenly spaced array that ranges from the minimum to the maximum
    new_a1_x = np.linspace(min_a1_x, max_a1_x, line_size)
    # Fit a 3rd degree polynomial to your data in the y and z dimensions
    a1_coefs_y = np.polyfit(a1[:, 0], a1[:, 1], 3)
    a1_coefs_z = np.polyfit(a1[:, 0], a1[:, 2], 3)
    # Get your new y and z coordinates from the coefficients of the above polynomial
    new_a1_y = np.polyval(a1_coefs_y, new_a1_x)
    new_a1_z = np.polyval(a1_coefs_z, new_a1_x)

    # Repeat for array 2:
    min_a2_x, max_a2_x = min(a2[:, 0]), max(a2[:, 0])
    new_a2_x = np.linspace(min_a2_x, max_a2_x, line_size)
    a2_coefs_y = np.polyfit(a2[:, 0], a2[:, 1], 3)
    a2_coefs_z = np.polyfit(a2[:, 0], a2[:, 2], 3)
    new_a2_y = np.polyval(a2_coefs_y, new_a2_x)
    new_a2_z = np.polyval(a2_coefs_z, new_a2_x)

    # Calculate the midpoint for x, y, and z coordinates
    midx = [np.mean([new_a1_x[i], new_a2_x[i]]) for i in range(line_size)]
    midy = [np.mean([new_a1_y[i], new_a2_y[i]]) for i in range(line_size)]
    midz = [np.mean([new_a1_z[i], new_a2_z[i]]) for i in range(line_size)]

    # Create a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Scatter plot for the original points
    # ax.scatter(a1[:, 0], a1[:, 1], a1[:, 2], c='red')
    # ax.scatter(a2[:, 0], a2[:, 1], a2[:, 2], c='green')

    # # Plot the midpoints line
    # ax.plot(midx, midy, midz, '--', c='blue')

    # # Set labels for axes
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # Set axis equal
    # # ax.axis('equal')

    # plt.show()

    # Step 1: Define the spline with the middle line
# Function to define an ellipse
    def ellipse(a, b, n_points=30):
        t = np.linspace(0, 2*np.pi, n_points)
        x = a * np.cos(t)
        y = b * np.sin(t)
        z = np.zeros_like(x)  # Ellipse lies initially in the XY plane
        return np.stack((x, y, z), axis=-1)

    # Given control points
    num_vertical_divisions = 40
    control_points = np.array([midx, midy, midz]).T
    tck, u = splprep(control_points.T, s=0)
    u_fine = np.linspace(0, 1, num_vertical_divisions)
    spline_points = np.array(splev(u_fine, tck)).T

    # Calculate tangents along the spline
    tangents = np.gradient(spline_points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

    # x and y axis length
    print("distance center-new_a1:", np.mean(np.linalg.norm(np.array([new_a1_x, new_a1_y, new_a1_z]) - np.array([midx, midy, midz]), axis=0)))
    print("distance center-new_a2:", np.mean(np.linalg.norm(np.array([new_a2_x, new_a2_y, new_a2_z]) - np.array([midx, midy, midz]), axis=0)))
    b = 2.6 * np.mean(np.linalg.norm(np.array([new_a1_x, new_a1_y, new_a1_z]) - np.array([midx, midy, midz]), axis=0))
    a = 1.3 * np.mean(np.linalg.norm(np.array([new_a1_x, new_a1_y, new_a1_z]) - np.array([midx, midy, midz]), axis=0))
    # Define the number of points on each ellipse
    n_ellipse_points = 80

    # Sweep the ellipse along the spline
    vertices = []
    faces = []

    for i, (point, tangent) in enumerate(zip(spline_points, tangents)):
        # Compute a perpendicular vector to create the local coordinate system
        if np.allclose(tangent, [0, 0, 1]):
            normal = np.array([0, 1, 0])
        else:
            normal = np.array([0, 0, 1])

        # normal direction is the same as the projection direction
        # normal = skel_center
        # normal_in_mesh_ref = np.linalg.inv(transf_matrix) @ np.array([*normal,1])
        # point_in_mesh_ref = np.linalg.inv(transf_matrix) @ np.array([*point,1])
        # y_offset_in_mesh_ref_frame = point_in_mesh_ref[1] - normal_in_mesh_ref[1]
        # # transform the offset to the skel frame
        # y_offset = transf_matrix @ np.array([0,y_offset_in_mesh_ref_frame,0,1])
        # normal[1] = point[1] + y_offset[1]

        binormal = np.cross(tangent, normal)
        binormal /= np.linalg.norm(binormal)
        normal = np.cross(binormal, tangent)
        
        # Create the rotation matrix from local to global coordinates
        R = np.array([binormal, normal, tangent]).T
        
        # Generate ellipse points in local coordinates
        ellipse_points = ellipse(a, b, n_points=n_ellipse_points)
        
        # Apply the rotation matrix to align with the spline's tangent
        ellipse_points_3d = np.dot(ellipse_points, R.T)
        
        # Translate the ellipse to the current point on the spline
        ellipse_points_3d += point
        
        vertices.extend(ellipse_points_3d)
        
        if i > 0:
            n = len(vertices)
            for j in range(n_ellipse_points):
                faces.append([
                    n - 2 * n_ellipse_points + j,
                    n - 2 * n_ellipse_points + (j + 1) % n_ellipse_points,
                    n - n_ellipse_points + j
                ])
                faces.append([
                    n - n_ellipse_points + j,
                    n - 2 * n_ellipse_points + (j + 1) % n_ellipse_points,
                    n - n_ellipse_points + (j + 1) % n_ellipse_points
                ])

    vertices = np.array(vertices)
    faces = np.array(faces)

    # Optional: Plot using Matplotlib for better customization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], color='white', edgecolor='black')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.axis('equal')
    # plt.show()
    
    # Step 4: Create the Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    return mesh



