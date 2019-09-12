import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker

sys.path.append('../..')
from cext import primitive_coverage_select_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveCoverageSelectLossTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, in_q, in_t, in_mask, in_pos, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      q = constant_op.constant(in_q)
      t = constant_op.constant(in_t)
      mask = constant_op.constant(in_mask)
      pos = constant_op.constant(in_pos)
      data_out = primitive_coverage_select_loss(z, q, t, mask, pos)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-8)

  def _VerifyGradientsNew(self, in_z, in_q, in_t, in_mask, in_pos, n_cube, batch_size):
    with self.test_session():
      z = constant_op.constant(in_z, shape=[batch_size, 3*n_cube])
      q = constant_op.constant(in_q, shape=[batch_size, 4*n_cube])
      t = constant_op.constant(in_t, shape=[batch_size, 3*n_cube])
      mask = constant_op.constant(in_mask)
      pos = constant_op.constant(in_pos)
      data_out = primitive_coverage_select_loss(z, q, t, mask, pos)
      ret = gradient_checker.compute_gradient(
          [z, q, t],
          [[batch_size, 3*n_cube], [batch_size, 4*n_cube], [batch_size, 3*n_cube]],
          data_out,
          [1],
          x_init_value=[np.asfarray(in_z).reshape([batch_size, 3*n_cube]),
                        np.asfarray(in_q).reshape([batch_size, 4*n_cube]),
                        np.asfarray(in_t).reshape([batch_size, 3*n_cube])]
          )
      # print(ret)
      self.assertAllClose(ret[0][0], ret[0][1], atol=5e-5)
      self.assertAllClose(ret[1][0], ret[1][1], atol=5e-5)
      self.assertAllClose(ret[2][0], ret[2][1], atol=5e-5)

  def testForward_0(self):
    # point outside one cube
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_mask = [[1], [1]]
    in_pos = [[0.5, 0.7, 0.5, 0.7],
              [0.5, 0.8, 0.5, 0.8],
              [0.5, 0.9, 0.5, 0.9],
              [0.0, 0.0, 1.0, 1.0]]
    expected = [0.685]
    self._VerifyValuesNew(in_z, in_q, in_t, in_mask, in_pos, expected)

  def testForward_1(self):
    # point inside one cube
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_mask = [[1], [1]]
    in_pos = [[0.2, 0.2],
              [0.2, 0.2],
              [0.2, 0.2],
              [0.0, 1.0]]
    expected = [0.0]
    self._VerifyValuesNew(in_z, in_q, in_t, in_mask, in_pos, expected)

  def testForward_2(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    in_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_mask = [[1, 1], [1, 1]]
    in_pos = [[0.2, 0.3, 0.7, 0.2, 0.3, 0.7],
              [0.2, 0.3, 0.8, 0.2, 0.3, 0.8],
              [0.2, 0.3, 0.9, 0.2, 0.3, 0.9],
              [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    expected = [0.04666667]
    self._VerifyValuesNew(in_z, in_q, in_t, in_mask, in_pos, expected)

  def testForward_3(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    in_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_mask = [[1, 0], [0, 1]]
    in_pos = [[0.2, 0.3, 0.7, 0.2, 0.3, 0.7],
              [0.2, 0.3, 0.8, 0.2, 0.3, 0.8],
              [0.2, 0.3, 0.9, 0.2, 0.3, 0.9],
              [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    ### [0, 0.03, 1.1, 0, 0, 0.14]
    expected = [0.21166667]
    self._VerifyValuesNew(in_z, in_q, in_t, in_mask, in_pos, expected)

  def testBackward_0(self):
    # one cube, one point, test q
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_mask = [[1], [1]]
    batch_size = 2
    n_cube = 1
    in_pos = [[0.5, 0.7, 0.5, 0.7],
              [0.5, 0.8, 0.5, 0.8],
              [0.5, 0.9, 0.5, 0.9],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(in_z, in_q, in_t, in_mask, in_pos, n_cube, batch_size)

  def testBackward_1(self):
    # two point to two cube
    in_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    in_mask = [[1, 0], [0, 1]]
    batch_size = 2
    n_cube = 2
    in_pos = [[0.3, 0.6, 0.3, 0.6],
              [0.3, 0.6, 0.3, 0.6],
              [0.3, 0.6, 0.3, 0.6],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(in_z, in_q, in_t, in_mask, in_pos, n_cube, batch_size)


if __name__ == '__main__':
  test.main()
