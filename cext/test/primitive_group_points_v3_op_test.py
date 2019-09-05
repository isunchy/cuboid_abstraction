import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

sys.path.append('../..')
from cext import primitive_group_points_v3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveGroupPointsTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, in_q, in_t, in_pos, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      q = constant_op.constant(in_q)
      t = constant_op.constant(in_t)
      pos = constant_op.constant(in_pos)
      data_out = primitive_group_points_v3(z, q, t, pos)
      actual = sess.run(data_out)
    self.assertAllEqual(expected, actual.flatten())

  def testForward_0(self):
    # two cube, multiple points
    in_z = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    in_q = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
    in_t = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.8], [0.1, 0.1, 0.1, 0.8, 0.8, 0.8]]
    in_pos = [[0.0, 0.1, 0.6, 0.5, 0.7, 0.2, 0.0, 0.1, 0.6, 0.5, 0.7, 0.2],
              [0.0, 0.1, 0.6, 0.5, 0.8, 0.2, 0.0, 0.1, 0.6, 0.5, 0.8, 0.2],
              [0.0, 0.1, 0.6, 0.5, 0.9, 0.2, 0.0, 0.1, 0.6, 0.5, 0.9, 0.2],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    expected = [0, 0, 1, 1, 1, 0, 2, 2, 3, 3, 3, 2]
    self._VerifyValuesNew(in_z, in_q, in_t, in_pos, expected)


if __name__ == '__main__':
  test.main()
