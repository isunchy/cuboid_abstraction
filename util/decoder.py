import numpy as np
import tensorflow as tf


def decoder(latent_code, n_part, shape_bias, name='decoder', is_training=True,
    reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    # fc1
    with tf.variable_scope('fc1'):
      fc1 = tf.layers.dense(latent_code, 128, activation=tf.nn.relu,
          use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.zeros_initializer(), trainable=is_training)

    # fc2
    with tf.variable_scope('fc2'):
      fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.zeros_initializer(), trainable=is_training)

    # z
    with tf.variable_scope('z'):
      z = tf.layers.dense(fc2, n_part*3, activation=None, use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.constant_initializer(-3/shape_bias),
          trainable=is_training)
      z = tf.nn.sigmoid(z*shape_bias)*0.5  # z \in [0, 0.5]

    # q
    with tf.variable_scope('q'):
      value = np.array([1, 0, 0, 0]*n_part)
      q = tf.layers.dense(fc2, n_part*4, activation=None, use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.constant_initializer(value),
          trainable=is_training)
      q = tf.reshape(q, [-1, n_part, 4])
      q = tf.nn.l2_normalize(q, axis=2)  # normalize [bs, n_part, 4]
      q = tf.reshape(q, [-1, n_part*4])

    # t
    with tf.variable_scope('t'):
      t = tf.layers.dense(fc2, n_part*3, activation=None, use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.zeros_initializer(), trainable=is_training)
      t = tf.nn.tanh(t)*0.5  # t \in [-0.5, 0.5]

  return z, q, t
