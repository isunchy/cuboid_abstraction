import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

sys.path.append('../..')
from cext import primitive_points_suffix_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveGroupPointsTest(test.TestCase):

  def _VerifyValuesNew(self, in_pos, expected):
    with self.test_session() as sess:
      pos = constant_op.constant(in_pos)
      data_out = primitive_points_suffix_index(pos)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual, atol=1e-6)

  def testForward_0(self):
    # batch size 2, two points
    in_pos = [[0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
              [0.6, 0.7, 0.6, 0.7, 0.6, 0.7]]
    expected = np.array([[0.0, 0.1, 0.6, 0.7],
                         [0.0, 0.1, 0.6, 0.7],
                         [0.0, 0.1, 0.6, 0.7],
                         [0.0, 0.0, 1.0, 1.0]])

    self._VerifyValuesNew(in_pos, expected)

  def testForward_1(self):
    # batch size 1, two points
    in_pos = [[0.0, 0.1, 0.0, 0.1, 0.0, 0.1]]
    expected = np.array([[0.0, 0.1],
                         [0.0, 0.1],
                         [0.0, 0.1],
                         [0.0, 0.0]])

    self._VerifyValuesNew(in_pos, expected)

if __name__ == '__main__':
  test.main()
