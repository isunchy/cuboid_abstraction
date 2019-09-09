import os
import numpy as np

from hierarchical_primitive import assemble_obj
from hierarchical_primitive import cube_inclusion
from hierarchical_primitive import points2cube


class HierarchicalPrimitive(assemble_obj.AssembleObj):
  '''
    L3:                       root                          #L3 == 1

    L2:                   N2_0    N2_1                      #L2 = 2
  
    L1:           N1_0    N1_1    N1_2    N1_3              #L1 = 4

    L0:  N0_0  N0_1  N0_2  N0_3  N0_4  N0_5  N0_6  N0_7     NL0 = 8
  '''
  def __init__(self, max_level=3):
    self.point_cloud = None
    self.max_level = max_level
    self.level_cube_number = np.ones(max_level + 1, dtype=int)  # [#L0, #L1, #L2, 1]
    self.level_cube_param = [None]*self.max_level
    self.level_point_cloud_index = [None]*self.max_level
    self.root_folder = ''
    self.level_index_relation = {}  # {'32': [0]*#L2, '21': ..., '10': ...}

  def load_cube(self, filename, level):
    assert(level < self.max_level)
    cube_count = 0
    mtl_filename = ''
    cube_mtl_index = []
    cube_param = {'z':[], 'q':[], 't':[]}
    vertex = []
    if self.root_folder == '':
      self.root_folder = filename[0:filename.rfind('/')+1]
    with open(filename) as f:
      for line in f:
        if line[0:6] == 'mtllib':
          mtl_filename = self.root_folder + line.split()[1]
        elif line[0:6] == 'usemtl':
          cube_mtl_index.append(int(line.split()[1][1:]))
        elif line[0] == '#':
          param = [float(line.split()[1:][p]) for p in range(10)]
          cube_param['z'].append(param[0:3])
          cube_param['q'].append(param[3:7])
          cube_param['t'].append(param[7:])
          cube_count += 1
        elif line[0] == 'v':
          vertex.append([float(line.split()[1:][p]) for p in range(3)])
        elif line[0] == 'f':
          pass
        else:
          raise ValueError('Parse obj file error!!')
    cube_param['z'] = np.array(cube_param['z'])
    cube_param['q'] = np.array(cube_param['q'])
    cube_param['t'] = np.array(cube_param['t'])
    content_dict = {
      'cube_count': cube_count,
      'mtl_filename': mtl_filename,
      'cube_mtl_index': cube_mtl_index,
      'cube_param': cube_param,
    }
    assert(len(cube_param['z']) ==
           len(cube_param['q']) ==
           len(cube_param['t']) ==
           len(cube_mtl_index) ==
           cube_count)
    self.level_cube_number[level] = cube_count
    self.level_cube_param[level] = content_dict
    if level == self.max_level - 1:
      self.level_index_relation['{}{}'.format(self.max_level,
          self.max_level - 1)] = np.zeros([cube_count], dtype=int)  # {'32'}

  def traverse_level(self, level, save_dir='tmp', verbose=False):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    for i in range(self.level_cube_number[level]):      
      material = {
          'mtl_filename': self.level_cube_param[level]['mtl_filename'],
          'mtl_index': np.array([i], dtype=np.int32)}    
      points2cube.points2cube(
          np.expand_dims(self.level_cube_param[level]['cube_param']['t'][i], axis=0),
          save_dir + '/cube_{}_{}_part'.format(level, i),
          scale=np.expand_dims(self.level_cube_param[level]['cube_param']['z'][i], axis=0),
          rotation=np.expand_dims(self.level_cube_param[level]['cube_param']['q'][i], axis=0),
          type='obj', material=material, verbose=verbose)


def correct_choose_flag(choose_flag, cube_param_0, cube_param_1, cube_param_2, verbose=False):
  index_relation_01 = cube_inclusion.cube_inclusion(cube_param_0, cube_param_1)
  index_relation_12 = cube_inclusion.cube_inclusion(cube_param_1, cube_param_2)
  if verbose: print('index_relation_01:', index_relation_01)
  if verbose: print('index_relation_12:', index_relation_12)
  if verbose: print('choose_flag:', choose_flag)
  n_cube_0 = cube_param_0['z'].shape[0]  # 32
  n_cube_1 = cube_param_1['z'].shape[0]  # 18
  n_cube_2 = cube_param_2['z'].shape[0]  # 9
  # when level 2 is picked as 1, if it has no children in level 1, or his
  # children has no children in level 0, set level 2 mask as 0
  for i in range(n_cube_2):
    if choose_flag[2][i] == 1:
      choose_flag[2][i] = 0
      for j in range(n_cube_1):
        if index_relation_12[j] == i:
          for k in range(n_cube_0):
            if index_relation_01[k] == j:
              choose_flag[2][i] = 1
              break
          if choose_flag[2][i] == 1:
            break

  # when level 1 is picked as 1, if it has no children in level 0, set its mask as 0
  for i in range(n_cube_1):
    if choose_flag[1][i] == 1:
      choose_flag[1][i] = 0
      for j in range(n_cube_0):
        if index_relation_01[j] == i:
          choose_flag[1][i] = 1
          break

  # when level 2 is picked as 1, set his next two level children as 0
  for i in range(n_cube_2):
    if choose_flag[2][i] == 1:
      for j in range(n_cube_1):
        if index_relation_12[j] == i:
          choose_flag[1][j] = 0
          for k in range(n_cube_0):
            if index_relation_01[k] == j:
              choose_flag[0][k] = 0

  # when level 1 is picked as 1, set his children as 0
  for i in range(n_cube_1):
    if choose_flag[1][i] == 1:
      for j in range(n_cube_0):
        if index_relation_01[j] == i:
          choose_flag[0][j] = 0

  # complete tree
  for i in range(n_cube_0):
    # the cube index of the path from leaf to root w.r.t. the ith cube in level 0
    flag_0_index = i
    flag_1_index = index_relation_01[i]
    flag_2_index = index_relation_12[index_relation_01[i]]
    flag_index_list = [flag_0_index, flag_1_index, flag_2_index]
    fill_level = -1
    if verbose: print('{} {} {}  {} {} {}'.format(choose_flag[0][flag_0_index], choose_flag[1][flag_1_index], choose_flag[2][flag_2_index],
                                                  flag_0_index, flag_1_index, flag_2_index))
    if choose_flag[0][flag_0_index] + choose_flag[1][flag_1_index] + choose_flag[2][flag_2_index] == 0:
      fill_level = 1
      # scan level 0 to find if the level 1 cube has other picked children
      for j in range(n_cube_0):
        if j != flag_0_index and index_relation_01[j] == flag_1_index:
          if choose_flag[0][j] == 1:
            fill_level = 0
            break
    # if level 1 cube has no other picked children, go to check level 2 cube
    if fill_level == 1:
      fill_level = 2
      for j in range(n_cube_1):
        if j != flag_1_index and index_relation_12[j] == flag_2_index:
          if choose_flag[1][j] == 1:
            fill_level = 1
            break
          for k in range(n_cube_0):
            if index_relation_01[k] == j:
              if choose_flag[0][k] == 1:
                fill_level = 1
                break
    if fill_level != -1:
      choose_flag[fill_level][flag_index_list[fill_level]] = 1
      if verbose: print('--> {} {} {}'.format(i, fill_level, flag_index_list[fill_level]))

  # if the path from bottom to top is a one cube branch, set the top level as 1 and its all children to 0
  for i in range(n_cube_0):
    # the cube index of the path from leaf to root w.r.t. the ith cube in level 0
    flag_0_index = i
    flag_1_index = index_relation_01[i]
    flag_2_index = index_relation_12[index_relation_01[i]]
    level_0_cube_number = 0
    level_1_cube_number = 0
    # count level 1 co-parent cube number
    for j in range(n_cube_1):
      if index_relation_12[j] == flag_2_index:
        level_1_cube_number += 1
    # count level 0 co-parent cube number
    for j in range(n_cube_0):
      if index_relation_01[j] == flag_1_index:
        level_0_cube_number += 1
    # if level 2 has one children and level 1 has one children
    if level_1_cube_number == 1 and level_0_cube_number == 1:
      if choose_flag[0][flag_0_index] == 1 or choose_flag[1][flag_1_index] == 1 or choose_flag[2][flag_2_index] == 1:
        choose_flag[2][flag_2_index] = 1
        choose_flag[0][flag_0_index] = 0
        choose_flag[1][flag_1_index] = 0
    # if only level 2 has one children
    if level_1_cube_number == 1 and level_0_cube_number > 1:
      if choose_flag[1][flag_1_index] == 1 or choose_flag[2][flag_2_index] == 1:
        choose_flag[2][flag_2_index] = 1
        choose_flag[1][flag_1_index] = 0
    # if only level 1 has one children
    if level_1_cube_number > 1 and level_0_cube_number == 1:
      if choose_flag[0][flag_0_index] == 1 or choose_flag[1][flag_1_index] == 1:
        choose_flag[1][flag_1_index] = 1
        choose_flag[0][flag_0_index] = 0

  if verbose: print('choose_flag:', choose_flag)
  return choose_flag


def vis_assembly_cube(cube_file_dir, cube_file_name, choose_flag_dir, choose_flag_name, des_dir, choose_flag_prefix=None, with_correction=False, verbose=False):
  hp = HierarchicalPrimitive()
  hp.load_cube('{}/cube_1_{}.obj'.format(cube_file_dir, cube_file_name), level=0)
  hp.load_cube('{}/cube_2_{}.obj'.format(cube_file_dir, cube_file_name), level=1)
  hp.load_cube('{}/cube_3_{}.obj'.format(cube_file_dir, cube_file_name), level=2)
  hp.traverse_level(2, save_dir=os.path.join(des_dir, 'tmp'), verbose=False)
  hp.traverse_level(1, save_dir=os.path.join(des_dir, 'tmp'), verbose=False)
  hp.traverse_level(0, save_dir=os.path.join(des_dir, 'tmp'), verbose=False)
  choose_flag = [None]*3
  if choose_flag_prefix is None: choose_flag_prefix = 'predict'
  choose_flag[0] = np.loadtxt('{}/{}_mask_1_{}.txt'.format(choose_flag_dir, choose_flag_prefix, choose_flag_name))
  choose_flag[1] = np.loadtxt('{}/{}_mask_2_{}.txt'.format(choose_flag_dir, choose_flag_prefix, choose_flag_name))
  choose_flag[2] = np.loadtxt('{}/{}_mask_3_{}.txt'.format(choose_flag_dir, choose_flag_prefix, choose_flag_name))
  if verbose: print('choose_flag:\n', choose_flag)
  hp.assemble_obj(os.path.join(des_dir, 'tmp'), des_dir, '{}_assembly_cube_{}'.format(choose_flag_prefix, cube_file_name), choose_flag, verbose=verbose)
  if with_correction:
    choose_flag = correct_choose_flag(choose_flag,
                                      hp.level_cube_param[0]['cube_param'],
                                      hp.level_cube_param[1]['cube_param'],
                                      hp.level_cube_param[2]['cube_param'],
                                      verbose=verbose)
    choose_flag_prefix += '_correction'
    hp.assemble_obj(os.path.join(des_dir, 'tmp'), des_dir, '{}_assembly_cube_{}'.format(choose_flag_prefix, cube_file_name), choose_flag, verbose=verbose)
