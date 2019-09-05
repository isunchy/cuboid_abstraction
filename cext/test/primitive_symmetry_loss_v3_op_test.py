import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker

sys.path.append('../..')
from cext import primitive_symmetry_loss_v3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveSymmetryLossTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, in_q, in_t, scale, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      q = constant_op.constant(in_q)
      t = constant_op.constant(in_t)
      data_out = primitive_symmetry_loss_v3(z, q, t, scale=scale, depth=5)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-6)

  def _VerifyGradientsNew(self, in_z, in_q, in_t, scale, n_cube, batch_size):
    with self.test_session() as sess:
      z = constant_op.constant(in_z, shape=[batch_size, 3*n_cube])
      q = constant_op.constant(in_q, shape=[batch_size, 4*n_cube])
      t = constant_op.constant(in_t, shape=[batch_size, 3*n_cube])
      data_out = primitive_symmetry_loss_v3(z, q, t, scale=scale, depth=5)
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
    # one cube, on symmetry plane
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.484375, 0.484375, 0.484375], [0.484375, 0.484375, 0.484375]]
    scale = 1.0
    expected = [0]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testForward_1(self):
    in_z = [[0.2425, 0.1222, 0.4111], [0.2425, 0.1222, 0.4111]]
    in_q = [[1.5, 0.4, 1.3, 2.2], [1.5, 0.4, 1.3, 2.2]]
    in_t = [[0.0710, 0.4125, 0.3224], [0.0710, 0.4125, 0.3224]]
    scale = 0.9
    expected = [0.08312]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testForward_2(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    in_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    scale = 0.8
    expected = [0.019409]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testBackward(self):
    # test q
    in_z = [[0.2425, 0.1222, 0.4111], [0.2425, 0.1222, 0.4111]]
    in_q = [[1.5, 0.4, 1.3, 2.2], [1.5, 0.4, 1.3, 2.2]]
    in_t = [[0.0710, 0.4125, 0.3224], [0.0710, 0.4125, 0.3224]]
    scale = 1.0
    n_cube = 1
    batch_size = 2
    self._VerifyGradientsNew(in_z, in_q, in_t, scale, n_cube, batch_size)


if __name__ == '__main__':
  test.main()
