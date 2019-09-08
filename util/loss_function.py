import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('..')
from cext import primitive_coverage_loss_v2
from cext import primitive_cube_coverage_loss_v3
from cext import primitive_consistency_loss_v2
from cext import primitive_mutex_loss_v3
from cext import primitive_aligning_loss_v2
from cext import primitive_symmetry_loss_v3
from cext import primitive_cube_area_average_loss
from cext import primitive_cube_volume_v2
from cext import primitive_group_points_v3
from cext import primitive_points_suffix_index


def coverage_loss_v2(cube_params, node_position):
  with tf.name_scope('coverage'):
    distance = primitive_coverage_loss_v2(cube_params[0], cube_params[1],
        cube_params[2], node_position)
    distance = tf.reduce_sum(distance)
    volume = primitive_cube_volume_v2(cube_params[0])
    volume = tf.reduce_sum(volume)
  return distance, volume


def cube_coverage_loss_v5(src_cube_params, des_cube_params, n_src_cube,
    node_position):
  with tf.name_scope('cube_coverage'):
    points_index = primitive_group_points_v3(src_cube_params[0],
        src_cube_params[1], src_cube_params[2], node_position)
    volume = primitive_cube_volume_v2(des_cube_params[0])
    volume = tf.reduce_sum(volume)
    distance = primitive_cube_coverage_loss_v3(des_cube_params[0],
        des_cube_params[1], des_cube_params[2], node_position, points_index,
        n_src_cube=n_src_cube)
    distance = tf.reduce_sum(distance)
  return distance, volume


def consistency_loss_v2(cube_params, node_position, num_sample=26):
  with tf.name_scope('consistency'):
    distance = primitive_consistency_loss_v2(cube_params[0], cube_params[1],
        cube_params[2], node_position, scale=1, num_sample=num_sample)
    distance = tf.reduce_sum(distance)
  return distance


def mutex_loss_v2(cube_params):
  with tf.name_scope('mutex'):
    distance = primitive_mutex_loss_v3(cube_params[0], cube_params[1],
        cube_params[2], scale=0.8)
    distance = tf.reduce_sum(distance)
  return distance


def aligning_loss_v2(cube_params):
  with tf.name_scope('aligning'):
    up_direction = tf.constant([0.0, 1.0, 0.0], shape=[3, 1])
    cosine_distance_up = primitive_aligning_loss_v2(cube_params[1], up_direction)
    front_direction = tf.constant([1.0, 0.0, 0.0], shape=[3, 1])
    cosine_distance_front = primitive_aligning_loss_v2(cube_params[1],
        front_direction)
    distance = (cosine_distance_up + cosine_distance_front) / 2
    distance = tf.reduce_sum(distance)
  return distance


def symmetry_loss_v3(cube_params):
  with tf.name_scope('symmetry'):
    distance = primitive_symmetry_loss_v3(cube_params[0], cube_params[1],
        cube_params[2], scale=1, depth=0)  # depth == 0, means symmetry plane is 0
    distance = tf.reduce_sum(distance)
  return distance


def cube_area_average_loss(cube_params):
  with tf.name_scope('cube_area_average'):
    distance = primitive_cube_area_average_loss(cube_params[0])
    distance = tf.reduce_sum(distance)
  return distance


def compute_loss_phase_one(cube_params, node_position):
  with tf.name_scope('compute_loss_phase_one'):
    coverage_distance, volume = coverage_loss_v2(cube_params, node_position)
    consistency_distance = consistency_loss_v2(cube_params, node_position)
    mutex_distance = mutex_loss_v2(cube_params)
    aligning_distance = aligning_loss_v2(cube_params)
    symmetry_distance = symmetry_loss_v3(cube_params)
    cube_area_average_distance = cube_area_average_loss(cube_params)
  return [coverage_distance, volume, consistency_distance, mutex_distance,
      aligning_distance, symmetry_distance, cube_area_average_distance]


def compute_loss_phase_merge(src_cube_params, des_cube_params, num_part_src, 
    node_position, phase='two'):
  with tf.name_scope('compute_loss_phase_' + phase):
    coverage_distance, volume = cube_coverage_loss_v5(src_cube_params,
        des_cube_params, num_part_src, node_position)
    if phase == 'two':
      consistency_distance = consistency_loss_v2(des_cube_params, node_position)
    else:
      consistency_distance = consistency_loss_v2(des_cube_params, node_position,
          num_sample=96)
    mutex_distance = mutex_loss_v2(des_cube_params)
    aligning_distance = aligning_loss_v2(des_cube_params)
    symmetry_distance = symmetry_loss_v3(des_cube_params)
  return [coverage_distance, volume, consistency_distance, mutex_distance,
      aligning_distance, symmetry_distance]
