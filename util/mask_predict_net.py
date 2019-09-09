import tensorflow as tf


def mask_predict_net(latent_code, n_part, name='phase', is_training=True,
    reuse=False):
  with tf.variable_scope('mask_predict', reuse=reuse):
    with tf.variable_scope(name):
      fc1 = tf.layers.dense(latent_code, 128, activation=tf.nn.tanh, use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.zeros_initializer(), trainable=is_training,
          name='fc1', reuse=reuse)
      fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.tanh, use_bias=True,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          bias_initializer=tf.zeros_initializer(), trainable=is_training,
          name='fc2', reuse=reuse)
      fc_out = tf.layers.dense(fc2, n_part, activation=tf.nn.sigmoid,
          use_bias=False,
          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
          trainable=is_training, name='fc_out', reuse=reuse)

  return fc_out
