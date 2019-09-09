import os

class AssembleObj:
  def __init__(self):
    pass

  @classmethod
  def assemble_obj(cls, src_file_dir, des_file_dir, save_filename, choose_flag, verbose=True):
    assert(os.path.exists(src_file_dir))
    if not os.path.exists(des_file_dir):
      os.makedirs(des_file_dir)
      if verbose: 'Create directory: {}'.format(des_file_dir)
    obj_filename = save_filename + '.obj'
    mtl_filename = save_filename + '.mtl'
    mtl_dict = {}
    with open(os.path.join(des_file_dir, obj_filename), 'w') as f1, \
        open(os.path.join(des_file_dir, mtl_filename), 'w') as f2:
      f1.write('mtllib {}\n'.format(mtl_filename))
      face_offset = 0
      for i in [2, 1, 0]:
        for j in range(len(choose_flag[i])):  # cf[2], cf[1], cf[0]
          if choose_flag[i][j] == 1:
            face_count = 0
            with open(os.path.join(src_file_dir, 'cube_{}_{}_part.obj').format(i, j)) as fin:
              for line in fin:
                if line[:6] == 'mtllib':
                  continue
                elif line[:6] == 'usemtl':
                  mtl_id = line.split()[1][:1] + '_' + str(i) + '_' + line.split()[1][1:]
                  f1.write('{} {}\n'.format(line.split()[0], mtl_id))
                elif line[0] == 'f':
                  number = tuple(int(v) + face_offset for v in line.split()[1:])
                  f1.write('f {} {} {} {}\n'.format(number[0], number[1], number[2], number[3]))
                else:
                  f1.write(line)
                face_count += 1 if line[0] == 'v' else 0
            face_offset += face_count
            mtl_content = open(os.path.join(src_file_dir, 'cube_{}_{}_part.mtl').format(i, j)).read().splitlines()
            for k in range(len(choose_flag[i])):
              if mtl_content[k*3] == 'newmtl m{}'.format(j):
                mtl_id = mtl_content[k*3].split()[1][:1] + '_' + str(i) + '_' + mtl_content[k*3].split()[1][1:]
                if mtl_id not in mtl_dict:
                  f2.write('newmtl {}\n{}\n{}\n'.format(mtl_id, mtl_content[k*3+1], mtl_content[k*3+2]))
                mtl_dict[mtl_id] = True  # insert mtl_id to dict

    if verbose: print('Save to {}.obj'.format(os.path.join(des_file_dir, save_filename)))
