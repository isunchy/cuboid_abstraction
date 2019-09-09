import numpy as np
import quaternion


ply_cube_template = """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
element face %d
property list uchar int vertex_index
end_header
%s"""

cube_vert = np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]], dtype=float)
cube_face = np.array([[0, 1, 2, 3], [0, 4, 5, 1], [0, 3, 7, 4], [6, 5, 4, 7], [6, 7, 3, 2], [6, 2, 1, 5]])


def compute_affine_cube_vertex(translation, scale, rotation=None):
  '''
  Compute cube vectex after affine transformation.

  :param translation: cube center [n, 3]
  :param scale: cube scale [n, 3]
  :param rotation: cube rotation [n, 4]
  :return: cube vertex after affine transformation [n*8, 3]
  '''
  n = translation.shape[0]
  vertex = np.ndarray([n*8, 3])
  for i in range(n):
    vert = cube_vert
    vert = vert * scale[i]
    if rotation is not None:
      rot = rotation[i]
      rot = np.quaternion(rot[0], rot[1], rot[2], rot[3])
      rot = quaternion.as_rotation_matrix(rot)
      vert = np.transpose(np.matmul(rot, np.transpose(vert)))
    vert = vert + translation[i]
    vertex[i*8:(i+1)*8, :] = vert
  return vertex


def save_ply(points, filename, scale=None, rotation=None):
  filename = filename + '.ply'
  assert(points.shape[1] == 3)
  n = points.shape[0]
  if not isinstance(scale, (np.ndarray)):
    scale = np.ones([n, 3]) * (0.125 if scale == None else scale)
  points = compute_affine_cube_vertex(points, scale, rotation)
  with open(filename, 'w') as f:
    f.write(ply_cube_template % (8*n, 6*n, ''))
    for i in range(n*8):
      f.write('%f %f %f\n' % (points[i][0], points[i][1], points[i][2]))
    index = 0
    for i in range(n):
      f.write('4 {} {} {} {}\n'.format(index, index + 1, index + 2, index + 3))
      f.write('4 {} {} {} {}\n'.format(index, index + 4, index + 5, index + 1))
      f.write('4 {} {} {} {}\n'.format(index, index + 3, index + 7, index + 4))
      f.write('4 {} {} {} {}\n'.format(index + 6, index + 5, index + 4, index + 7))
      f.write('4 {} {} {} {}\n'.format(index + 6, index + 7, index + 3, index + 2))
      f.write('4 {} {} {} {}\n'.format(index + 6, index + 2, index + 1, index + 5))
      index += 8


def save_obj(points, filename, scale=None, rotation=None, material=None):
  filename = filename + '.obj'
  assert(points.shape[1] == 3)
  n = points.shape[0]
  if not isinstance(scale, (np.ndarray)):
    scale = np.ones([n, 3]) * (0.125 if scale == None else scale)
  affine_points = compute_affine_cube_vertex(points, scale, rotation)
  with open(filename, 'w') as f:
    if material is not None:
      material_filename = filename.replace('.obj', '.mtl')
      save_material_file(material['mtl_filename'], material_filename)
      f.write('mtllib {}\n'.format(material_filename[filename.rfind('/')+1:]))
    for i in range(n*8):
      if (material is not None) and (i%8 == 0):
        if rotation is not None:
          f.write('# {} {} {} {} {} {} {} {} {} {}\n'.format(
              scale[i//8][0], scale[i//8][1], scale[i//8][2],
              rotation[i//8][0], rotation[i//8][1], rotation[i//8][2], rotation[i//8][3],
              points[i//8][0], points[i//8][1], points[i//8][2]))
        f.write('usemtl m{}\n'.format(material['mtl_index'][i//8]))
      f.write('v {} {} {}\n'.format(affine_points[i][0], affine_points[i][1], affine_points[i][2]))
    index = 1
    for i in range(n):
      f.write('f {} {} {} {}\n'.format(index, index + 1, index + 2, index + 3))
      f.write('f {} {} {} {}\n'.format(index, index + 4, index + 5, index + 1))
      f.write('f {} {} {} {}\n'.format(index, index + 3, index + 7, index + 4))
      f.write('f {} {} {} {}\n'.format(index + 6, index + 5, index + 4, index + 7))
      f.write('f {} {} {} {}\n'.format(index + 6, index + 7, index + 3, index + 2))
      f.write('f {} {} {} {}\n'.format(index + 6, index + 2, index + 1, index + 5))
      index += 8


def save_material_file(src_filename, des_filename):
  with open(src_filename) as f1, open(des_filename, 'w') as f2:
    for line in f1:
      f2.write(line)


def points2cube(points, filename, scale=None, rotation=None, transpose=False,
    type='obj', material=None, verbose=False):
  if transpose:
    points = np.transpose(points)
  if type == 'obj':
    save_obj(points, filename, scale=scale, rotation=rotation,
        material=material)
    if verbose:
      print(points.shape, '--> {}.obj'.format(filename))
  elif type == 'ply':
    save_ply(points, filename, scale=scale, rotation=rotation)
    if verbose:
      print(points.shape, '--> {}.ply'.format(filename))
  else:
    raise ValueError('File extension not support!!!')


if __name__ == '__main__':
  points = np.array([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0],
                     [0, 1, 1],
                     [1, 0, 0],
                     [1, 0, 1],
                     [1, 1, 0],
                     [1, 1, 1]])
  filename = 'sample_points'
  points2cube(points, filename, type='ply', verbose=True)
  points2cube(points, filename, type='obj', verbose=True)