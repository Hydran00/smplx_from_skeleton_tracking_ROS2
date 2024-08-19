import numpy as np
from math_utils import compute_triangle_xfm
from location import Location
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def E(xy, XY, dxdy):
    """
    Edge equation from "3D Distance from a Point to a Triangle" of  Mark W. Jones (1995) 
    """
    return round((xy[0] - XY[0]) * dxdy[1] - (xy[1] - XY[1]) * dxdy[0], 6)

def find_closest_point_on_triangle(point, triXfm, triXfmInv, v1, v2, v3):
    # Transform point to local triangle coordinates
    point_homogeneous = np.append(point, [1.0])
    pt_local = np.dot(triXfm, point_homogeneous)
    x, y = pt_local[:2]
    
    pt_2d = np.array([x, y])
    
    # Transform vertices to local 2D coordinates
    v1_homogeneous = np.append(v1, [1.0])
    v2_homogeneous = np.append(v2, [1.0])
    v3_homogeneous = np.append(v3, [1.0])
    
    P1 = np.zeros(2)
    P2 = np.dot(triXfm, v2_homogeneous)[:2]
    P3 = np.dot(triXfm, v3_homogeneous)[:2]
    
    P1P3 = P3 - P1
    P2P3 = P3 - P2
    
    E12 = np.array([-1.0, 0.0])
    E13 = np.array([P3[1], -P3[0]])
    E23 = np.array([-P2P3[1], P2P3[0]])

    norm_E13 = np.linalg.norm(E13)
    norm_E23 = np.linalg.norm(E23)
    if norm_E13 != 0:
        E13 /= norm_E13
    if norm_E23 != 0:
        E23 /= norm_E23
    
    # Check which edge or vertex the point is closest to
    if x <= 0.0:  # Outside edge P1P2
        if y <= 0.0:
            if E(pt_2d, P1, E13) >= 0.0:
                closest_local = np.array([P1[0], P1[1], 0.0])
                closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
                return closestPoint, Location.V1
        elif y >= P2[1]:
            if E(pt_2d, P2, E23) <= 0.0:
                closest_local = np.array([P2[0], P2[1], 0.0])
                closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
                return closestPoint, Location.V2
        else:
            closest_local = np.array([0.0, y, 0.0])
            closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
            return closestPoint, Location.V1V2
    
    if E(pt_2d, P1, P1P3) >= 0.0:
        if E(pt_2d, P1, E13) >= 0.0:
            closest_local = np.array([P1[0], P1[1], 0.0])
            closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
            return closestPoint, Location.V1
        elif E(pt_2d, P3, E13) <= 0.0:
            if E(pt_2d, P3, E23) >= 0.0:
                closest_local = np.array([P3[0], P3[1], 0.0])
                closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
                return closestPoint, Location.V3
        else:
            projection = np.dot(pt_2d, P1P3) * P1P3
            closest_local = np.array([projection[0], projection[1], 0.0])
            closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
            return closestPoint, Location.V1V3
    
    if E(pt_2d, P2, P2P3) <= 0:
        if E(pt_2d, P2, E23) <= 0:
            closest_local = np.array([P2[0], P2[1], 0.0])
            closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
            return closestPoint, Location.V2
        elif E(pt_2d, P3, E23) >= 0:
            closest_local = np.array([P3[0], P3[1], 0.0])
            closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
            return closestPoint, Location.V3
        else:
            projection = np.dot(pt_2d - P2, P2P3) / np.dot(P2P3, P2P3) * P2P3 + P2
            closest_local = np.array([projection[0], projection[1], 0.0])
            closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
            return closestPoint, Location.V2V3
    
    closest_local = np.array([pt_2d[0], pt_2d[1], 0.0])
    closestPoint = np.dot(triXfmInv, np.append(closest_local, [1.0]))[:3]
    return closestPoint, Location.IN




def visualize_test_case(P1, P2, P3, point, closestPoint, location):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the triangle edges
    ax.plot([P1[0], P2[0]], [P1[1], P2[1]], [P1[2], P2[2]], 'k-', label='Edge P1P2')  # Edge P1P2
    ax.plot([P2[0], P3[0]], [P2[1], P3[1]], [P2[2], P3[2]], 'k-', label='Edge P2P3')  # Edge P2P3
    ax.plot([P3[0], P1[0]], [P3[1], P1[1]], [P3[2], P1[2]], 'k-', label='Edge P1P3')  # Edge P1P3

    # Plot the triangle face
    ax.plot_trisurf([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]], color='lightgray', alpha=0.5)

    # Plot the point and the closest point
    ax.scatter(point[0], point[1], point[2], color='blue', s=100, label='Point')
    ax.scatter(closestPoint[0], closestPoint[1], closestPoint[2], color='red', s=100, label='Closest Point')

    # Add text annotations
    ax.text(point[0], point[1], point[2], '  Point', color='blue', verticalalignment='bottom', horizontalalignment='right')
    ax.text(closestPoint[0], closestPoint[1], closestPoint[2], f'  Closest Point\n{location}', color='red', verticalalignment='bottom', horizontalalignment='right')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Closest Point on Triangle Test Case\nClosest Point: {location}')

    # Set the legend and grid
    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()

# Example test cases with visualization
def run_tests():

    # Test Case 1: Point exactly at vertex P1
    # point = np.array([0.0, 0.0, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)

    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(triXfm, closestPoint, location)
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 2: Point exactly at vertex P2
    # point = np.array([1.0, 0.0, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)
    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(closestPoint, location)
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 3: Point exactly at vertex P3
    # point = np.array([0.0, 1.0, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)
    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(closestPoint, location)
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 4: Point on edge P1P2 but not at vertices
    # point = np.array([0.5, 0.0, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)
    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(closestPoint, location)
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 5: Point on edge P1P3 but not at vertices
    # point = np.array([0.0, 0.5, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)
    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(closestPoint, location)
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 6: Point on edge P2P3 but not at vertices
    # point = np.array([0.5, 0.5, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)
    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(closestPoint, location)
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 7: Point inside the triangle
    # point = np.array([0.25, 0.25, 0.0])
    # P1 = np.array([0.0, 0.0, 0.0])
    # P2 = np.array([1.0, 0.0, 0.0])
    # P3 = np.array([0.0, 1.0, 0.0])
    # triXfm = compute_triangle_xfm(P1, P2, P3)
    # triXfmInv = np.linalg.inv(triXfm)
    # closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    # print(closestPoint, location)   
    # visualize_test_case(P1, P2, P3, point, closestPoint, location)

    # # Test Case 8: Point outside the triangle
    point = np.array([0.4, 1.1, -0.0])
    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([1.0, 0.0, 0.0])
    P3 = np.array([0.0, 1.0, 0.0])
    triXfm = compute_triangle_xfm(P1, P2, P3)
    triXfmInv = np.linalg.inv(triXfm)
    closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    print(closestPoint, location)
    visualize_test_case(P1, P2, P3, point, closestPoint, location)

    point = np.array([-0.5, -1.5, 0.0])
    closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    print(closestPoint, location)
    visualize_test_case(P1, P2, P3, point, closestPoint, location)

    point = np.array([-0.5, 0.2, 0.3])
    closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    print(closestPoint, location)
    visualize_test_case(P1, P2, P3, point, closestPoint, location)

    point = np.array([1.5, 1.0, 0.3])
    closestPoint, location = find_closest_point_on_triangle(point, triXfm, triXfmInv, P1, P2, P3)
    print(closestPoint, location)
    visualize_test_case(P1, P2, P3, point, closestPoint, location)    

if __name__ == "__main__":
    run_tests()