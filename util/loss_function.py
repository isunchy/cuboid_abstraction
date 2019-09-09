import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('..')
# initial training
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
# mask prediction
from cext import primitive_cube_coverage_loss_v4
from cext import primitive_coverage_split_loss_v3
from cext import primitive_consistency_split_loss
from cext import primitive_tree_generation


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


def cube_coverage_loss_v6(src_latent_code, des_latent_code, n_src_cube,
    node_position):
  with tf.name_scope('cube_coverage'):
    points_index = primitive_group_points_v3(src_latent_code[0],
        src_latent_code[1], src_latent_code[2], node_position)
    _, relation = primitive_cube_coverage_loss_v4(des_latent_code[0],
        des_latent_code[1], des_latent_code[2], node_position, points_index,
        n_src_cube=n_src_cube)
  return relation


def coverage_split_loss_v2(latent_code, node_position):
  with tf.name_scope('coverage'):
    ## The output `distance` of one cube, is the distance summation of all
    ## points belong to the cube. The number of points one cube contains is
    ## stored in `point_count`.
    distance, point_count = primitive_coverage_split_loss_v3(latent_code[0],
        latent_code[1], latent_code[2], node_position)
  return distance, point_count


def consistency_split_loss(latent_code, node_position, num_sample=26):
  with tf.name_scope('consistency'):
    distance = primitive_consistency_split_loss(latent_code[0], latent_code[1],
        latent_code[2], node_position, scale=1, num_sample=num_sample)
  return distance


def mask_sparseness_loss(logit_1, logit_2, logit_3):
  with tf.name_scope('mask_sparseness_loss'):
    logit = tf.concat([logit_1, logit_2, logit_3], axis=1)
    loss = tf.reduce_mean(logit)
  return loss


def shape_similarity_loss(logit_1, logit_2, logit_3, cube_params_1,
    cube_params_2, cube_params_3, node_position):
  with tf.name_scope('shape_similarity_loss'):
    relation_12 = cube_coverage_loss_v6(cube_params_1, cube_params_2, n_part_1,
        node_position) # [bs, n_part_1]
    relation_23 = cube_coverage_loss_v6(cube_params_2, cube_params_3, n_part_2,
        node_position) # [bs, n_part_2]
    coverage_loss_1, point_count_1 = coverage_split_loss_v2(cube_params_1, node_position) # [bs, n_part_1]
    coverage_loss_2, point_count_2 = coverage_split_loss_v2(cube_params_2, node_position) # [bs, n_part_2]
    coverage_loss_3, point_count_3 = coverage_split_loss_v2(cube_params_3, node_position) # [bs, n_part_3]
    coverage_loss = tf.concat([coverage_loss_1, coverage_loss_2, coverage_loss_3], axis=1) # [bs, n1+n2+n3]
    point_count = tf.cast(tf.concat([point_count_1, point_count_2, point_count_3], axis=1), tf.float32) # [bs, n1+n2+n3]
    consistency_loss_1 = consistency_split_loss(cube_params_1, node_position, num_sample=26) # [bs, n_part_1]
    consistency_loss_2 = consistency_split_loss(cube_params_2, node_position, num_sample=26) # [bs, n_part_2]
    consistency_loss_3 = consistency_split_loss(cube_params_3, node_position, num_sample=26) # [bs, n_part_3]
    consistency_loss = tf.concat([consistency_loss_1, consistency_loss_2, consistency_loss_3], axis=1) # [bs, n1+n2+n3]

    def compute_chamfer_distance(logit):
      mean_coverage_loss = tf.reduce_sum(coverage_loss*logit, axis=1, keepdims=True)/tf.reduce_sum(logit*point_count, axis=1, keepdims=True) # [bs, 1]
      mean_consistency_loss = tf.reduce_sum(consistency_loss*logit, axis=1, keepdims=True)/tf.reduce_sum(logit, axis=1, keepdims=True) # [bs, 1]
      mean_chamfer_distance = mean_coverage_loss + mean_consistency_loss # [bs, 1]
      return mean_chamfer_distance

    logit = tf.concat([logit_1, logit_2, logit_3], axis=1) # [bs, n1+n2+n3]
    chamfer_distance = compute_chamfer_distance(logit) # [bs, 1]
    max_logit_1 = tf.zeros_like(logit_1)
    max_logit_2 = tf.zeros_like(logit_2)
    max_logit_3 = tf.ones_like(logit_3)
    max_logit = tf.concat([max_logit_1, max_logit_2, max_logit_3], axis=1) # [bs, n1+n2+n3]
    max_chamfer_distance = compute_chamfer_distance(max_logit) # [bs, 1]
    min_logit_1 = tf.ones_like(logit_1)
    min_logit_2 = tf.zeros_like(logit_2)
    min_logit_3 = tf.zeros_like(logit_3)
    min_logit = tf.concat([min_logit_1, min_logit_2, min_logit_3], axis=1) # [bs, n1+n2+n3]
    min_chamfer_distance = compute_chamfer_distance(min_logit) # [bs, 1]
    normalized_chamfer_distance = 2 * chamfer_distance / (max_chamfer_distance + min_chamfer_distance) # [bs, 1]
    loss = tf.reduce_mean(normalized_chamfer_distance) # [1]
  return loss, relation_12, relation_23


def mask_completeness_loss(logit_1, logit_2, logit_3, relation_12, relation_23):
  with tf.name_scope('mask_completeness_loss'):
    L1 = logit_1
    L2 = tf.batch_gather(logit_2, relation_12)
    L3 = tf.batch_gather(logit_3, tf.batch_gather(relation_23, relation_12))
    loss = tf.reduce_mean((L1 + L2 + L3 - 1)**2)
  return loss
