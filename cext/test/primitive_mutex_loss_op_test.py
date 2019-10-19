import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker

sys.path.append('../..')
from cext import primitive_mutex_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class PrimitiveMutexLossTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, in_q, in_t, scale, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      q = constant_op.constant(in_q)
      t = constant_op.constant(in_t)
      data_out = primitive_mutex_loss(z, q, t, scale=scale)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-6)

  def _VerifyGradientsNew(self, in_z, in_q, in_t, scale, n_cube, batch_size):
    with self.test_session() as sess:
      z = constant_op.constant(in_z, shape=[batch_size, 3*n_cube])
      q = constant_op.constant(in_q, shape=[batch_size, 4*n_cube])
      t = constant_op.constant(in_t, shape=[batch_size, 3*n_cube])
      data_out = primitive_mutex_loss(z, q, t, scale=scale)
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

  def testForward_degenerate(self):
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    scale = 1
    expected = [0.0]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testForward_0(self):
    in_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    scale = 1
    expected = [0.003704]  # 0.1 / 27 == 0.003704
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testForward_1(self):
    in_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    scale = 0.9
    expected = [0.013333]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testForward_2(self):
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]]
    in_t = [[0.1, 0.1, 0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.4]]
    scale = 1
    expected = [0.005556]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testForward_3(self):
    in_z = [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.2, 0.3, 0.28, 0.56, 0.84], [0.1, 0.2, 0.3, 0.28, 0.56, 0.84]]
    scale = 1
    expected = [0.000741]
    self._VerifyValuesNew(in_z, in_q, in_t, scale, expected)

  def testBackward_degenerate(self):
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    scale = 1
    n_cube = 1
    batch_size = 2
    self._VerifyGradientsNew(in_z, in_q, in_t, scale, n_cube, batch_size)

  def testBackward_0(self):
    # carefully design the distance along each axis
    # when add delta, the minimal distance axis should not change
    # x axis
    in_z = [[0.1, 0.2, 0.3, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.2, 0.3, 0.28, 0.56, 0.84], [0.1, 0.2, 0.3, 0.28, 0.56, 0.84]]
    scale = 1
    n_cube = 2
    batch_size = 2
    self._VerifyGradientsNew(in_z, in_q, in_t, scale, n_cube, batch_size)

  def testBackward_1(self):
    # carefully design the distance along each axis
    # when add delta, the minimal distance axis should not change
    # y axis
    in_z = [[0.2, 0.1, 0.3, 0.2, 0.1, 0.3], [0.2, 0.1, 0.3, 0.2, 0.1, 0.3]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.2, 0.1, 0.3, 0.56, 0.28, 0.84], [0.2, 0.1, 0.3, 0.56, 0.28, 0.84]]
    scale = 1
    n_cube = 2
    batch_size = 2
    self._VerifyGradientsNew(in_z, in_q, in_t, scale, n_cube, batch_size)

  def testBackward_2(self):
    # carefully design the distance along each axis
    # when add delta, the minimal distance axis should not change
    # z axis
    in_z = [[0.3, 0.2, 0.1, 0.3, 0.2, 0.1], [0.3, 0.2, 0.1, 0.3, 0.2, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.3, 0.2, 0.1, 0.84, 0.56, 0.28], [0.3, 0.2, 0.1, 0.84, 0.56, 0.28]]
    scale = 1
    n_cube = 2
    batch_size = 2
    self._VerifyGradientsNew(in_z, in_q, in_t, scale, n_cube, batch_size)


if __name__ == '__main__':
  test.main()
