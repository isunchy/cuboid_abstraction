import sys
import tensorflow as tf

sys.path.append('..')
from cext import octree_conv
from cext import octree_pooling


def encoder(data, octree, is_training=True, reuse=None):
  with tf.variable_scope('encoder', reuse=reuse):
    # octconv1
    with tf.variable_scope('octconv1'):
      kernel = tf.get_variable('weights', shape=[16, 3, 3**3], dtype=tf.float32,
          initializer=tf.contrib.layers.variance_scaling_initializer())
      octconv1 = octree_conv(data, kernel, octree, curr_depth=5,
          num_output=16, kernel_size=3, stride=1)
      octconv1 = tf.nn.relu(octconv1)

    # octpool1
    [octpool1, _] = octree_pooling(octconv1, octree, curr_depth=5,
        name='octpool1')

    # octconv2
    with tf.variable_scope('octconv2'):
      kernel = tf.get_variable('weights', shape=[32, 16, 3**3], dtype=tf.float32,
          initializer=tf.contrib.layers.variance_scaling_initializer())
      octconv2 = octree_conv(octpool1, kernel, octree, curr_depth=4,
          num_output=32, kernel_size=3, stride=1)
      octconv2 = tf.nn.relu(octconv2)

    # octpool2
    [octpool2, _] = octree_pooling(octconv2, octree, curr_depth=4,
        name='octpool2')

    # octconv3
    with tf.variable_scope('octconv3'):
      kernel = tf.get_variable('weights', shape=[64, 32, 3**3], dtype=tf.float32,
          initializer=tf.contrib.layers.variance_scaling_initializer())
      octconv3 = octree_conv(octpool2, kernel, octree, curr_depth=3,
          num_output=64, kernel_size=3, stride=1)
      octconv3 = tf.nn.relu(octconv3)

    # octpool3
    [octpool3, _] = octree_pooling(octconv3, octree, curr_depth=3,
        name='octpool3')

    # octconv4
    with tf.variable_scope('octconv4'):
      kernel = tf.get_variable('weights', shape=[128, 64, 3**3], dtype=tf.float32,
          initializer=tf.contrib.layers.variance_scaling_initializer())
      octconv4 = octree_conv(octpool3, kernel, octree, curr_depth=2,
          num_output=128, kernel_size=3, stride=1)
      octconv4 = tf.nn.relu(octconv4)

    # octpool4
    [octpool4, _] = octree_pooling(octconv4, octree, curr_depth=2,
        name='octpool4')

    # conv5
    with tf.variable_scope('conv5'):
      conv5 = tf.layers.conv2d(octpool4, 256, kernel_size=[8, 1],
          strides=[8, 1], data_format='channels_first', use_bias=False,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          trainable=is_training)
      conv5 = tf.nn.relu(conv5)

    # [1, C, BS, 1] --> [BS, C, 1, 1] --> [BS, C]
    conv5 = tf.transpose(conv5, perm=[2, 1, 0, 3])
    conv5 = tf.squeeze(conv5, [2, 3])

    # latent code
    with tf.variable_scope('latent_code'):
      latent_code = tf.layers.dense(conv5, 128, activation=tf.nn.tanh,
          use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.zeros_initializer(), trainable=is_training)

  return latent_code
