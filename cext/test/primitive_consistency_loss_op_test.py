import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker

sys.path.append('../..')
from cext import primitive_consistency_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveConsistencyLossTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, in_q, in_t, in_pos, scale, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      q = constant_op.constant(in_q)
      t = constant_op.constant(in_t)
      pos = constant_op.constant(in_pos)
      data_out = primitive_consistency_loss(z, q, t, pos, scale=scale)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-8)

  def _VerifyGradientsNew(self, in_z, in_q, in_t, in_pos, scale, n_cube,
      batch_size):
    with self.test_session():
      z = constant_op.constant(in_z, shape=[batch_size, 3*n_cube])
      q = constant_op.constant(in_q, shape=[batch_size, 4*n_cube])
      t = constant_op.constant(in_t, shape=[batch_size, 3*n_cube])
      pos = constant_op.constant(in_pos)
      data_out = primitive_consistency_loss(z, q, t, pos, scale=scale)
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
      self.assertAllClose(ret[0][0], ret[0][1], atol=5e-4)
      self.assertAllClose(ret[1][0], ret[1][1], atol=5e-4)
      self.assertAllClose(ret[2][0], ret[2][1], atol=5e-4)


  def testForward_0(self):
    # one cube, one point
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    scale = 1.0
    in_pos = [[0.5, 0.5],
              [0.5, 0.5],
              [0.5, 0.5],
              [0.0, 1.0]]
    expected = [0.500769]
    self._VerifyValuesNew(in_z, in_q, in_t, in_pos, scale, expected)

  def testForward_1(self):
    # one cube, two point
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    scale = 0.8
    in_pos = [[0.5, 0.7, 0.5, 0.7],
              [0.5, 0.8, 0.5, 0.8],
              [0.5, 0.9, 0.5, 0.9],
              [0.0, 0.0, 1.0, 1.0]]
    expected = [0.493292]
    self._VerifyValuesNew(in_z, in_q, in_t, in_pos, scale, expected)

  def testForward_2(self):
    # two cube, two point
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    in_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    scale = 0.8
    in_pos = [[0.5, 0.7, 0.5, 0.7],
              [0.5, 0.8, 0.5, 0.8],
              [0.5, 0.9, 0.5, 0.9],
              [0.0, 0.0, 1.0, 1.0]]
    expected = [0.380892]
    self._VerifyValuesNew(in_z, in_q, in_t, in_pos, scale, expected)

  def testBackward_0(self):
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    scale = 0.8
    batch_size = 2
    n_cube = 1
    in_pos = [[0.5, 0.7, 0.5, 0.7],
              [0.5, 0.8, 0.5, 0.8],
              [0.5, 0.9, 0.5, 0.9],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(in_z, in_q, in_t, in_pos, scale, n_cube, batch_size)

  def testBackward_1(self):
    # test q
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[5.0, 4.0, 3.0, 1.0], [5.0, 4.0, 3.0, 1.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    scale = 0.8
    batch_size = 2
    n_cube = 1
    in_pos = [[0.2, 0.0, 0.2, 0.0],
              [0.2, 0.0, 0.2, 0.0],
              [0.2, 0.0, 0.2, 0.0],
              [0.0, 0.0, 1.0, 1.0]]
    self._VerifyGradientsNew(in_z, in_q, in_t, in_pos, scale, n_cube, batch_size)


if __name__ == '__main__':
  test.main()
