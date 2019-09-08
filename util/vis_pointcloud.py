import numpy as np
import os

cube_vert_raw = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)

# todo: use four edge face
cube_face = np.array([[1, 7, 5], [1, 3, 7], [1, 4, 3], [1, 2, 4], [3, 8, 7], [3, 4, 8], [5, 7, 8], [5, 8, 6], [1, 5, 6], [1, 6, 2], [2, 6, 8], [2, 8, 4]])


def save_points(position, save_file, depth=5, offset=0):
  with open(save_file, 'w') as f:
    n_vert = np.shape(position)[0]
    position -= offset
    vert_offset = 0
    cube_vert = (cube_vert_raw-0.5) / (2**depth * 2) 
    for i in range(n_vert):
      for j in range(8):
        x = position[i][0] + cube_vert[j][0]
        y = position[i][1] + cube_vert[j][1]
        z = position[i][2] + cube_vert[j][2]
        f.write("v {} {} {}\n".format(x, y, z))
      faces = cube_face + vert_offset
      for j in range(12):
        f.write("f {} {} {}\n".format(faces[j][0], faces[j][1], faces[j][2]))
      vert_offset += 8
