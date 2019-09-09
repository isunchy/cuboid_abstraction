import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

sys.path.append('../..')
from cext import primitive_tree_generation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class PrimitiveGroupPointsTest(test.TestCase):

  def _VerifyValuesNew(self, n1, n2, n3, in_mask, in_relation_1, in_relation_2,
      expected):
    with self.test_session() as sess:
      mask = constant_op.constant(in_mask)
      relation_1 = constant_op.constant(in_relation_1)
      relation_2 = constant_op.constant(in_relation_2)
      data_out = primitive_tree_generation(mask, relation_1, relation_2,
          n_part_1=n1, n_part_2=n2, n_part_3=n3)
      actual = sess.run(data_out)
    self.assertAllEqual(expected[0], actual[0])
    self.assertAllEqual(expected[1], actual[1])
    self.assertAllEqual(expected[2], actual[2])

  def testForward_0(self):
    # complete
    n1 = 8
    n2 = 4
    n3 = 2
    mask = [
        [1, 1,     0, 0, 0,     0,     0, 0,
            0,       1,           0,    0,
                0,                  1]
    ]
    relation_1 = [[0, 0, 1, 1, 1, 2, 3, 3]]
    relation_2 = [[0, 0, 1, 1]]
    out_mask_1 = mask
    out_mask_2 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            1,       1,           0,    0,
                0,                  1]
    ]
    out_mask_3 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    expected = [out_mask_1, out_mask_2, out_mask_3]
    self._VerifyValuesNew(n1, n2, n3, mask, relation_1, relation_2, expected)

  def testForward_1(self):
    # fill level 1
    n1 = 8
    n2 = 4
    n3 = 2
    mask = [
        [1, 1,     1, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                0,                  1]
    ]
    relation_1 = [[0, 0, 1, 1, 1, 2, 3, 3]]
    relation_2 = [[0, 0, 1, 1]]
    out_mask_1 = [
        [1, 1,     1, 1, 1,     0,     0, 0,
            0,       0,           0,    0,
                0,                  1]
    ]
    out_mask_2 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            1,       1,           0,    0,
                0,                  1]
    ]
    out_mask_3 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    expected = [out_mask_1, out_mask_2, out_mask_3]
    self._VerifyValuesNew(n1, n2, n3, mask, relation_1, relation_2, expected)

  def testForward_2(self):
    # fill level 2 and delete children
    n1 = 8
    n2 = 4
    n3 = 2
    mask = [
        [1, 1,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                0,                  1],
        [1, 1,     0, 1, 1,     0,     0, 1,
            0,       0,           1,    0,
                0,                  1]
    ]
    relation_1 = [[0, 0, 1, 1, 1, 2, 3, 3],
                  [0, 0, 1, 1, 1, 2, 3, 3]]
    relation_2 = [[0, 0, 1, 1],
                  [0, 0, 1, 1]]
    out_mask_1 = [
        [1, 1,     0, 0, 0,     0,     0, 0,
            0,       1,           0,    0,
                0,                  1],
        [1, 1,     1, 1, 1,     0,     0, 0,
            0,       0,           0,    0,
                0,                  1]
    ]
    out_mask_2 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            1,       1,           0,    0,
                0,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            1,       1,           0,    0,
                0,                  1]
    ]
    out_mask_3 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    expected = [out_mask_1, out_mask_2, out_mask_3]
    self._VerifyValuesNew(n1, n2, n3, mask, relation_1, relation_2, expected)

  def testForward_3(self):
    # random
    n1 = 8
    n2 = 4
    n3 = 2
    mask = [
        [0, 1,     0, 1, 0,     1,     0, 1,
            0,       1,           0,    1,
                0,                  1],
        [0, 1,     1, 1, 0,     1,     0, 0,
            1,       0,           1,    0,
                1,                  0]
    ]
    relation_1 = [[0, 0, 1, 1, 1, 2, 3, 3],
                  [0, 0, 1, 1, 1, 2, 3, 3]]
    relation_2 = [[0, 0, 1, 1],
                  [0, 0, 1, 1]]
    out_mask_1 = [
        [1, 1,     0, 0, 0,     0,     0, 0,
            0,       1,           0,    0,
                0,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           1,    1,
                1,                  0]
    ]
    out_mask_2 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            1,       1,           0,    0,
                0,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           1,    1,
                1,                  0]
    ]
    out_mask_3 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    expected = [out_mask_1, out_mask_2, out_mask_3]
    self._VerifyValuesNew(n1, n2, n3, mask, relation_1, relation_2, expected)

  def testForward_4(self):
    # all 1 and all 0
    n1 = 8
    n2 = 4
    n3 = 2
    mask = [
        [1, 1,     1, 1, 1,     1,     1, 1,
            1,       1,           1,    1,
                1,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                0,                  0]
    ]
    relation_1 = [[0, 0, 1, 1, 1, 2, 3, 3],
                  [0, 0, 1, 1, 1, 2, 3, 3]]
    relation_2 = [[0, 0, 1, 1],
                  [0, 0, 1, 1]]
    out_mask_1 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    out_mask_2 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    out_mask_3 = [
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1],
        [0, 0,     0, 0, 0,     0,     0, 0,
            0,       0,           0,    0,
                1,                  1]
    ]
    expected = [out_mask_1, out_mask_2, out_mask_3]
    self._VerifyValuesNew(n1, n2, n3, mask, relation_1, relation_2, expected)

  def testForward_5(self):
    # parent cube has no children cube
    n1 = 8
    n2 = 4
    n3 = 2
    mask = [
        [        1, 0, 0,         0, 0, 0,   1, 1,
                     0,       1,      1,       0,
            1,                    0],
        [0,   0, 1, 0,   1,   1, 1, 0,
         1,       1,     0,     0,
         0,              0]
    ]
    relation_1 = [[0, 0, 0, 2, 2, 2, 3, 3],
                  [0, 1, 1, 1, 2, 3, 3, 3]]
    relation_2 = [[1, 1, 1, 1],
                  [0, 1, 1, 1]]
    out_mask_1 = [
        [        1, 1, 1,         0, 0, 0,   1, 1,
                     0,       0,      1,       0,
            0,                    0],
        [0,   0, 0, 0,   0,   1, 1, 1,
         0,       1,     1,     0,
         1,              0]
    ]
    out_mask_2 = [
        [        0, 0, 0,         0, 0, 0,   0, 0,
                     1,       0,      1,       1,
            0,                    0],
        [0,   0, 0, 0,   0,   0, 0, 0,
         0,       1,     1,     1,
         1,              0]
    ]
    out_mask_3 = [
        [        0, 0, 0,         0, 0, 0,   0, 0,
                     0,       0,      0,       0,
            0,                    1],
        [0,   0, 0, 0,   0,   0, 0, 0,
         0,       0,     0,     0,
         1,              1]
    ]
    expected = [out_mask_1, out_mask_2, out_mask_3]
    self._VerifyValuesNew(n1, n2, n3, mask, relation_1, relation_2, expected)


if __name__ == '__main__':
  test.main()
