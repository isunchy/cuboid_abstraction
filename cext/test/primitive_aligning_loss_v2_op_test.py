import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.ops import gradient_checker

sys.path.append('../..')
from cext import primitive_aligning_loss_v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveAligningLossTest(test.TestCase):

  def _VerifyValuesNew(self, in_q, in_dir, expected):
    with self.test_session() as sess:
      q = constant_op.constant(in_q)
      direction = constant_op.constant(in_dir)
      data_out = primitive_aligning_loss_v2(q, direction)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-6)

  def _VerifyGradientsNew(self, in_q, in_dir, n_cube, batch_size):
    with self.test_session() as sess:
      q = constant_op.constant(in_q, shape=[batch_size, 4*n_cube])
      direction = constant_op.constant(in_dir)
      data_out = primitive_aligning_loss_v2(q, direction)
      ret = gradient_checker.compute_gradient(
          q,
          [batch_size, 4*n_cube],
          data_out,
          [1],
          x_init_value=np.asfarray(in_q).reshape([batch_size, 4*n_cube])
          )
      # print(ret)
      self.assertAllClose(ret[0], ret[1], atol=5e-4)

  def testForward_0(self):
    # upright direction
    in_q = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    in_dir = [0.0, 1.0, 0.0]
    expected = [0.0]
    self._VerifyValuesNew(in_q, in_dir, expected)

  def testForward_1(self):
    # random q
    in_q = [[1.5, 0.4, 1.3, 2.2], [1.5, 0.4, 1.3, 2.2]]
    in_dir = [0.0, 1.0, 0.0]
    expected = [1.118568]
    self._VerifyValuesNew(in_q, in_dir, expected)

  def testForward_2(self):
    # two cube, front direction
    in_q = [[1.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2], [1.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.3, 0.2]]
    in_dir = [1.0, 0.0, 0.0]
    expected = [0.240741]
    self._VerifyValuesNew(in_q, in_dir, expected)

  def testBackward(self):
    # test q
    in_q = [[1.5, 0.4, 1.3, 2.2], [1.5, 0.4, 1.3, 2.2]]
    in_dir = [0.0, 1.0, 0.0]
    n_cube = 1
    batch_size = 2
    self._VerifyGradientsNew(in_q, in_dir, n_cube, batch_size)


if __name__ == '__main__':
  test.main()
