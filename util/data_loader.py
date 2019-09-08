import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('..')
from cext import octree_database
from cext import primitive_points_suffix_index

def _add_data_to_queue(data, octree, points, test):
  queue = tf.FIFOQueue(capacity=100,
      dtypes=[tf.float32, tf.int32, tf.float32])
  enqueue_op = queue.enqueue([data, octree, points])
  numberOfThreads = 1 if test else 50
  qr = tf.train.QueueRunner(queue, [enqueue_op]*numberOfThreads)
  tf.train.add_queue_runner(qr)
  [data, octree, points] = queue.dequeue()
  return data, octree, points


def read_and_decode(filename_queue, batch_size, n_points, test=False):
  reader = tf.TFRecordReader()
  keys, serialized_examples = reader.read_up_to(filename_queue, batch_size)
  feature = {'octree': tf.FixedLenFeature([], tf.string),
             'points': tf.FixedLenFeature([n_points*3], tf.float32)}
  features = tf.parse_example(serialized_examples, features=feature)
  octree = features['octree']
  points = features['points']
  [data, octree, _] = octree_database(octree)
  return _add_data_to_queue(data, octree, points, test)


def data_loader(dataset, batch_size, n_points=5000, test=False):
  with tf.name_scope('read_and_decode'):
    filename_queue = tf.train.string_input_producer([dataset])
    data, octree, points = read_and_decode(filename_queue, batch_size, n_points, test)
    node_position = primitive_points_suffix_index(points)
  return data, octree, node_position
