import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker

sys.path.append('../..')
from cext import primitive_cube_coverage_loss_v3
from cext import primitive_group_points_v3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveCubeCoverageLossTest(test.TestCase):

  def _VerifyValuesNew(self, src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
      n_src_cube, expected):
    with self.test_session() as sess:
      sz = constant_op.constant(src_z)
      sq = constant_op.constant(src_q)
      st = constant_op.constant(src_t)
      dz = constant_op.constant(des_z)
      dq = constant_op.constant(des_q)
      dt = constant_op.constant(des_t)
      pos = constant_op.constant(in_pos)
      points_index = primitive_group_points_v3(sz, sq, st, pos)
      data_out = primitive_cube_coverage_loss_v3(dz, dq, dt, pos, points_index,
          n_src_cube=n_src_cube)
      [actual, pi] = sess.run([data_out, points_index])
      # print("points_index: ", pi)
    self.assertAllClose(expected, actual.flatten(), atol=1e-8)

  def _VerifyGradientsNew(self, src_z, src_q, src_t, des_z, des_q, des_t,
      in_pos, n_src_cube, n_des_cube, batch_size):
    with self.test_session():
      sz = constant_op.constant(src_z, shape=[batch_size, 3*n_src_cube])
      sq = constant_op.constant(src_q, shape=[batch_size, 4*n_src_cube])
      st = constant_op.constant(src_t, shape=[batch_size, 3*n_src_cube])
      dz = constant_op.constant(des_z, shape=[batch_size, 3*n_des_cube])
      dq = constant_op.constant(des_q, shape=[batch_size, 4*n_des_cube])
      dt = constant_op.constant(des_t, shape=[batch_size, 3*n_des_cube])
      pos = constant_op.constant(in_pos)
      points_index = primitive_group_points_v3(sz, sq, st, pos)
      data_out = primitive_cube_coverage_loss_v3(dz, dq, dt, pos, points_index,
          n_src_cube=n_src_cube)
      ret = gradient_checker.compute_gradient(
          [dz, dq, dt],
          [[batch_size, 3*n_des_cube], [batch_size, 4*n_des_cube], [batch_size, 3*n_des_cube]],
          data_out,
          [1],
          x_init_value=[np.asfarray(des_z).reshape([batch_size, 3*n_des_cube]),
                        np.asfarray(des_q).reshape([batch_size, 4*n_des_cube]),
                        np.asfarray(des_t).reshape([batch_size, 3*n_des_cube])]
          )
      # print(ret)
      self.assertAllClose(ret[0][0], ret[0][1], atol=5e-5)
      self.assertAllClose(ret[1][0], ret[1][1], atol=5e-5)
      self.assertAllClose(ret[2][0], ret[2][1], atol=5e-5)

  def testForward_degenerate(self):
    # one src cube cover no point
    src_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    src_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    src_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    n_src_cube = 2
    des_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    des_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    des_t = [[0.2, 0.2, 0.2, 0.8, 0.8, 0.8], [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]]
    in_pos = [[0.6, 0.8, 0.6, 0.8],
              [0.6, 0.8, 0.6, 0.8],
              [0.6, 0.8, 0.6, 0.8],
              [0.0, 0.0, 1.0, 1.0]]
    expected = [0.0075]
    self._VerifyValuesNew(src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
        n_src_cube, expected)

  def testForward_0(self):
    # two point for two group
    src_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    src_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    src_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    n_src_cube = 2
    des_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    des_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    des_t = [[0.3, 0.3, 0.3, 0.8, 0.8, 0.8], [0.3, 0.3, 0.3, 0.8, 0.8, 0.8]]
    in_pos = [[0.1, 0.8, 0.1, 0.8],
              [0.1, 0.8, 0.1, 0.8],
              [0.1, 0.8, 0.1, 0.8],
              [0.0, 0.0, 1.0, 1.0]]
    expected = [0.015]
    self._VerifyValuesNew(src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
        n_src_cube, expected)

  def testForward_1(self):
    # random q
    src_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    src_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    src_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    n_src_cube = 2
    des_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    des_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    des_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_pos = [[0.1, 0.7, 0.1, 0.7],
              [0.1, 0.8, 0.1, 0.8],
              [0.1, 0.9, 0.1, 0.9],
              [0.0, 0.0, 1.0, 1.0]]
    expected = [0.07]
    self._VerifyValuesNew(src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
        n_src_cube, expected)


  def testBackward_degenerate(self):
    # one src cube cover no point
    src_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    src_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    src_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    n_src_cube = 2
    des_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    des_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    des_t = [[0.2, 0.2, 0.2, 0.8, 0.8, 0.8], [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]]
    n_des_cube = 2
    batch_size = 2
    in_pos = [[0.6, 0.8, 0.6, 0.8],
              [0.6, 0.8, 0.6, 0.8],
              [0.6, 0.8, 0.6, 0.8],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
        n_src_cube, n_des_cube, batch_size)

  def testBackward_0(self):
    # two point, two group, two cube
    src_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    src_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    src_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    n_src_cube = 2
    des_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    des_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    des_t = [[0.2, 0.2, 0.2, 0.8, 0.8, 0.8], [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]]
    n_des_cube = 2
    batch_size = 2
    in_pos = [[0.4, 0.6, 0.4, 0.6],
              [0.4, 0.6, 0.4, 0.6],
              [0.4, 0.6, 0.4, 0.6],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
        n_src_cube, n_des_cube, batch_size)

  def testBackward_1(self):
    # test q
    src_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    src_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    src_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    n_src_cube = 2
    des_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    des_q = [[0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5]]
    des_t = [[0.3, 0.3, 0.3, 0.8, 0.8, 0.8], [0.3, 0.3, 0.3, 0.8, 0.8, 0.8]]
    n_des_cube = 2
    batch_size = 2
    in_pos = [[0.1, 0.6, 0.1, 0.6],
              [0.1, 0.6, 0.1, 0.6],
              [0.1, 0.6, 0.1, 0.6],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(src_z, src_q, src_t, des_z, des_q, des_t, in_pos,
        n_src_cube, n_des_cube, batch_size)


if __name__ == '__main__':
  test.main()