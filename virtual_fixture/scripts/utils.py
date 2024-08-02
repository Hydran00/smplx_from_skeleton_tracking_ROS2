# LUNG_US_SMPL_VERTICES = {
#     "right_basal_midclavicular" : 928, #661,#929      # 13
#     "right_upper_midclavicular": 594, #595,#596       # 14
#     "left_basal_midclavicular" : 4415, #4414,#4417  # 11
#     "left_upper_midclavicular": 4082, #4084,#4085   # 12
# }
LUNG_US_SMPL_FACES = {
    "left_basal_midclavicular" : 13345, #4414,#4417  # 11
    "left_upper_midclavicular": 13685, #4084,#4085   # 12
    "right_basal_midclavicular" : 6457, #661,#929      # 13
    "right_upper_midclavicular": 884, #595,#596       # 14
}
import numpy as np
def to_z_up(mesh):
    R = mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    mesh.rotate(R, center=[0, 0, 0])
    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
    # mesh.rotate(R, center=[0, 0, 0])  