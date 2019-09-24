import numpy as np
import quaternion


color_palette = {
  'L1': [[158,   1,  66],
         [176,  21,  70],
         [194,  41,  74],
         [213,  62,  79],
         [223,  77,  75],
         [233,  93,  71],
         [244, 109,  67],
         [247, 130,  77],
         [250, 152,  87],
         [253, 174,  97],
         [253, 190, 111],
         [253, 207, 125],
         [254, 224, 139],
         [246, 231, 143],
         [238, 238, 147],
         [230, 245, 152],
         [210, 237, 156],
         [190, 229, 160],
         [171, 221, 164],
         [148, 212, 164],
         [125, 203, 164],
         [102, 194, 165],
         [ 84, 174, 173],
         [ 67, 155, 181],
         [ 50, 136, 189],
         [ 64, 117, 180],
         [ 79,  98, 171],
         [ 94,  79, 162],
         [117,  79, 152],
         [140,  79, 142],
         [163,  79, 132],
         [161,  53, 110]],
  'L2': [[242, 198,   4],
         [252, 218, 123],
         [ 77, 146,  33],
         [161, 206, 107],
         [ 41, 125, 198],
         [126, 193, 221],
         [198,  31,  40],
         [252, 136, 123],
         [  5, 112, 103],
         [ 87, 193, 177],
         [107,  53, 168],
         [139, 117, 198],
         [206,  37, 135],
         [247, 155, 222],
         [196,  98,  13],
         [253, 184,  99]],
  'L3': [[246,  83,  20],
         [124, 187,   0],
         [  0, 161, 241],
         [255, 187,   0],
         [ 11, 239, 239],
         [247, 230,  49],
         [255,  96, 165],
         [178,  96, 255]]
}


cube_vert = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
cube_vert = cube_vert * 2 - 1
cube_face = np.array([[1, 3, 7, 5], [1, 2, 4, 3], [3, 4, 8, 7], [5, 7, 8, 6], [1, 5, 6, 2], [2, 6, 8, 4]])


def save_parts(cube_params, output_file, level='1'):
  n_part = np.shape(cube_params)[0]
  mtl_filename = output_file.replace('.obj', '.mtl')
  with open(mtl_filename, 'w') as f:
    palette = color_palette['L{}'.format(level)]
    n_color = len(palette)
    for i in range(n_part):
      color_index = i
      if level == '1':
        step = n_color/float(n_part)
        color_index = int(i*step)
      part_color = palette[color_index]
      f.write('newmtl m{}\nKd {} {} {}\nKa 0 0 0\n'.format(i, float(part_color[0])/256, float(part_color[1])/256, float(part_color[2])/256))

  with open(output_file, 'w') as f:
    vert_offset = 0
    n_vert = np.shape(cube_vert)[0]
    f.write('mtllib {}\n'.format(mtl_filename.split('/')[-1]))
    for i in range(n_part):
      verts = cube_vert
      scale = cube_params[i][0:3]
      rot = cube_params[i][3:7]
      translation = cube_params[i][7:10]
      verts = verts * scale
      quat = np.quaternion(rot[0], rot[1], rot[2], rot[3])
      rotation = quaternion.as_rotation_matrix(quat)
      verts = np.transpose(np.matmul(rotation, np.transpose(verts)))
      verts = verts + translation

      faces = cube_face + vert_offset
      f.write('# {} {} {} {} {} {} {} {} {} {}\n'.format(
          scale[0], scale[1], scale[2],
          rot[0], rot[1], rot[2], rot[3],
          translation[0], translation[1], translation[2])
      )
      f.write('usemtl m{}\n'.format(i))
      for j in range(n_vert):
        sigma = 0.1
        f.write('v {} {} {}\n'.format(verts[j][0], verts[j][1], verts[j][2]))
      for j in range(np.shape(cube_face)[0]):
        f.write('f {} {} {} {}\n'.format(faces[j][0], faces[j][1], faces[j][2], faces[j][3]))
      vert_offset += n_vert
