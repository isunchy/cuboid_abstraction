import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker


sys.path.append('../..')
from cext import primitive_cube_area_average_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveCubeAreaTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      data_out = primitive_cube_area_average_loss(z)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-8)

  def _VerifyGradientsNew(self, in_z, n_cube, batch_size):
    with self.test_session():
      z = constant_op.constant(in_z, shape=[batch_size, 3*n_cube])
      data_out = primitive_cube_area_average_loss(z)
      ret = gradient_checker.compute_gradient(
          z,
          [batch_size, 3*n_cube],
          data_out,
          [1],
          x_init_value=np.asfarray(in_z).reshape([batch_size, 3*n_cube])
          )
      # print(ret)
      self.assertAllClose(ret[0], ret[1], atol=5e-5)

  def testForward_0(self):
    # one cube
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    expected = [0.0]
    self._VerifyValuesNew(in_z, expected)

  def testForward_1(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]]
    expected = [0.0018]
    self._VerifyValuesNew(in_z, expected)

  def testForward_2(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    expected = [0.10215]
    self._VerifyValuesNew(in_z, expected)

  def testBackward_0(self):
    # one cube
    in_z = in_z = [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]]
    batch_size = 2
    n_cube = 2
    self._VerifyGradientsNew(in_z, n_cube, batch_size)

  def testBackward_1(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    batch_size = 1
    n_cube = 5
    self._VerifyGradientsNew(in_z, n_cube, batch_size)

if __name__ == '__main__':
  test.main()
