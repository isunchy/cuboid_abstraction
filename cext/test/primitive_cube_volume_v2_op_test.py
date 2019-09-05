import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

sys.path.append('../..')
from cext import primitive_cube_volume_v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveCubeVolumeTest(test.TestCase):

  def _VerifyValuesNew(self, in_z, expected):
    with self.test_session() as sess:
      z = constant_op.constant(in_z)
      data_out = primitive_cube_volume_v2(z)
      actual = sess.run(data_out)
    self.assertAllClose(expected, actual.flatten(), atol=1e-8)

  def testForward_0(self):
    # one cube
    in_z = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
    expected = [0.008]
    self._VerifyValuesNew(in_z, expected)

  def testForward_1(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]]
    expected = [0.072]
    self._VerifyValuesNew(in_z, expected)

  def testForward_2(self):
    # two cube
    in_z = [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    expected = [0.044]
    self._VerifyValuesNew(in_z, expected)


if __name__ == '__main__':
  test.main()
