import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('util')
import vis_primitive
import vis_pointcloud
from data_loader import *
from encoder import *
from decoder import *
from loss_function import *


tf.app.flags.DEFINE_string('log_dir', 'log/initial_training/PGen_xxxx/0_16_8_4_airplane_0',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('train_data', 
                           'data/airplane_octree_points_d5_train.tfrecords',
                           """Train data location.""")
tf.app.flags.DEFINE_string('test_data', 
                           'data/airplane_octree_points_d5_test_100.tfrecords',
                           """Test data location.""")
tf.app.flags.DEFINE_integer('train_batch_size', 32,
                            """Mini-batch size for the training.""")
tf.app.flags.DEFINE_integer('test_batch_size', 1,
                            """Mini-batch size for the testing.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_integer('max_iter', 100010,
                            """Maximum training iterations.""")
tf.app.flags.DEFINE_integer('test_every_n_steps', 1000,
                            """Test model every n training steps.""")
tf.app.flags.DEFINE_integer('test_iter', 100,
                            """Test steps in testing phase.""")
tf.app.flags.DEFINE_integer('disp_every_n_steps', 5000,
                            """Generate mesh every n training steps.""")
tf.app.flags.DEFINE_integer('n_part_1', 16,
                            """Number of cuboids to generate.""")
tf.app.flags.DEFINE_integer('n_part_2', 8,
                            """Number of cuboids to generate in phase two.""")
tf.app.flags.DEFINE_integer('n_part_3', 4,
                            """Number of cuboids to generate in phase three.""")
tf.app.flags.DEFINE_float('coverage_weight', 1,
                          """Weight of coverage loss""")
tf.app.flags.DEFINE_float('consistency_weight', 1,
                          """Weight of consistency loss""")
tf.app.flags.DEFINE_float('mutex_weight', 1,
                          """Weight of mutex loss""")
tf.app.flags.DEFINE_float('aligning_weight', 0.001,
                          """Weight of aligning loss""")
tf.app.flags.DEFINE_float('symmetry_weight', 0.1,
                          """Weight of symmetry loss""")
tf.app.flags.DEFINE_float('area_average_weight', 5,
                          """Weight of cube surface area average loss""")
tf.app.flags.DEFINE_float('shape_bias_1', 0.01, """phase one shape bias""")
tf.app.flags.DEFINE_float('shape_bias_2', 0.005, """phase two shape bias""")
tf.app.flags.DEFINE_float('shape_bias_3', 0.001, """phase three shape bias""")
tf.app.flags.DEFINE_string('cache_folder', 'test',
                           """Directory where to dump immediate data.""")
tf.app.flags.DEFINE_string('ckpt', 'None',
                           """Restore weights from checkpoint file.""")
tf.app.flags.DEFINE_boolean('test', False, """Test only flags.""")
tf.app.flags.DEFINE_string('gpu', '0', """GPU id.""")
tf.app.flags.DEFINE_integer('num_points_in_points_file', 5000,
                            """Number of points sampled on original shape surface.""")


FLAGS = tf.app.flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

max_iter = FLAGS.max_iter
test_iter = FLAGS.test_iter
n_part_1 = FLAGS.n_part_1
n_part_2 = FLAGS.n_part_2
n_part_3 = FLAGS.n_part_3
shape_bias_1 = FLAGS.shape_bias_1
shape_bias_2 = FLAGS.shape_bias_2
shape_bias_3 = FLAGS.shape_bias_3
n_points = FLAGS.num_points_in_points_file

for key, value in tf.app.flags.FLAGS.flag_values_dict().items():
  print('{}: {}'.format(key, value))
print('====')
sys.stdout.flush()


def initial_loss_function(cube_params_1, cube_params_2, cube_params_3,
    node_position):
  with tf.name_scope('initial_loss_function'):
    [coverage_distance_1,
     cube_volume_1,
     consistency_distance_1,
     mutex_distance_1,
     aligning_distance_1,
     symmetry_distance_1,
     cube_area_average_distance_1
    ] = compute_loss_phase_one(cube_params_1, node_position)

    loss_1 = (coverage_distance_1 * FLAGS.coverage_weight +
              consistency_distance_1 * FLAGS.consistency_weight +
              mutex_distance_1 * FLAGS.mutex_weight +
              aligning_distance_1 * FLAGS.aligning_weight +
              symmetry_distance_1 * FLAGS.symmetry_weight +
              cube_area_average_distance_1 * FLAGS.area_average_weight)

    [coverage_distance_2,
     cube_volume_2,
     consistency_distance_2,
     mutex_distance_2,
     aligning_distance_2,
     symmetry_distance_2
    ] = compute_loss_phase_merge(cube_params_1, cube_params_2, n_part_1,
        node_position, phase='two')

    loss_2 = (coverage_distance_2 * FLAGS.coverage_weight +
              consistency_distance_2 * FLAGS.consistency_weight +
              mutex_distance_2 * FLAGS.mutex_weight +
              aligning_distance_2 * FLAGS.aligning_weight +
              symmetry_distance_2 * FLAGS.symmetry_weight)

    [coverage_distance_3,
     cube_volume_3,
     consistency_distance_3,
     mutex_distance_3,
     aligning_distance_3,
     symmetry_distance_3
    ] = compute_loss_phase_merge(cube_params_2, cube_params_3, n_part_2,
        node_position, phase='three')

    loss_3 = (coverage_distance_3 * FLAGS.coverage_weight +
              consistency_distance_3 * FLAGS.consistency_weight +
              mutex_distance_3 * FLAGS.mutex_weight +
              aligning_distance_3 * FLAGS.aligning_weight +
              symmetry_distance_3 * FLAGS.symmetry_weight)

  return [loss_1,
          coverage_distance_1,
          cube_volume_1,
          consistency_distance_1,
          mutex_distance_1,
          aligning_distance_1,
          symmetry_distance_1,
          cube_area_average_distance_1,
          loss_2,
          coverage_distance_2,
          cube_volume_2,
          consistency_distance_2,
          mutex_distance_2,
          aligning_distance_2,
          symmetry_distance_2,
          loss_3,
          coverage_distance_3,
          cube_volume_3,
          consistency_distance_3,
          mutex_distance_3,
          aligning_distance_3,
          symmetry_distance_3
          ]


def train_network():
  data, octree, node_position = data_loader(FLAGS.train_data,
      FLAGS.train_batch_size, n_points)

  latent_code = encoder(data, octree, is_training=True, reuse=False)
  cube_params_1 = decoder(latent_code, n_part_1, shape_bias_1,
      name='decoder_phase_one', is_training=True, reuse=False)
  cube_params_2 = decoder(latent_code, n_part_2, shape_bias_2,
      name='decoder_phase_two', is_training=True, reuse=False)
  cube_params_3 = decoder(latent_code, n_part_3, shape_bias_3,
      name='decoder_phase_three', is_training=True, reuse=False)

  [train_loss_1,
   coverage_distance_1,
   cube_volume_1,
   consistency_distance_1,
   mutex_distance_1,
   aligning_distance_1,
   symmetry_distance_1,
   cube_area_average_distance_1,
   train_loss_2,
   coverage_distance_2,
   cube_volume_2,
   consistency_distance_2,
   mutex_distance_2,
   aligning_distance_2,
   symmetry_distance_2,
   train_loss_3,
   coverage_distance_3,
   cube_volume_3,
   consistency_distance_3,
   mutex_distance_3,
   aligning_distance_3,
   symmetry_distance_3
  ] = initial_loss_function(cube_params_1, cube_params_2, cube_params_3,
      node_position)

  train_loss = train_loss_1 + train_loss_2 + train_loss_3

  with tf.name_scope('train_summary'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      tvars = tf.trainable_variables()
      encoder_vars = [var for var in tvars if 'encoder' in var.name]
      decoder_1_vars = [var for var in tvars if 'phase_one' in var.name]
      decoder_2_vars = [var for var in tvars if 'phase_two' in var.name]
      decoder_3_vars = [var for var in tvars if 'phase_three' in var.name]
      
      var_list = encoder_vars + decoder_1_vars + decoder_2_vars + decoder_3_vars
      
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      solver = optimizer.minimize(train_loss, var_list=var_list)
      lr = optimizer._lr
      summary_lr_scheme = tf.summary.scalar('learning_rate', lr)
      summary_train_loss = tf.summary.scalar('train_loss', train_loss)

    summary_coverage_distance_1 = tf.summary.scalar('coverage_distance_1', coverage_distance_1)
    summary_cube_volume_1 = tf.summary.scalar('cube_volume_1', cube_volume_1)
    summary_consistency_distance_1 = tf.summary.scalar('consistency_distance_1', consistency_distance_1)
    summary_mutex_distance_1 = tf.summary.scalar('mutex_distance_1', mutex_distance_1)
    summary_aligning_distance_1 = tf.summary.scalar('aligning_distance_1', aligning_distance_1)
    summary_symmetry_distance_1 = tf.summary.scalar('symmetry_distance_1', symmetry_distance_1)
    summary_cube_area_average_distance_1 = tf.summary.scalar('cube_area_average_distance_1', cube_area_average_distance_1)
    summary_list_phase_one = [summary_coverage_distance_1, 
                              summary_cube_volume_1,
                              summary_consistency_distance_1,
                              summary_mutex_distance_1,
                              summary_aligning_distance_1,
                              summary_symmetry_distance_1,
                              summary_cube_area_average_distance_1]

    summary_coverage_distance_2 = tf.summary.scalar('coverage_distance_2', coverage_distance_2)
    summary_cube_volume_2 = tf.summary.scalar('cube_volume_2', cube_volume_2)
    summary_consistency_distance_2 = tf.summary.scalar('consistency_distance_2', consistency_distance_2)
    summary_mutex_distance_2 = tf.summary.scalar('mutex_distance_2', mutex_distance_2)
    summary_aligning_distance_2 = tf.summary.scalar('aligning_distance_2', aligning_distance_2)
    summary_symmetry_distance_2 = tf.summary.scalar('symmetry_distance_2', symmetry_distance_2)
    summary_list_phase_two = [summary_coverage_distance_2, 
                              summary_cube_volume_2,
                              summary_consistency_distance_2,
                              summary_mutex_distance_2,
                              summary_aligning_distance_2,
                              summary_symmetry_distance_2]

    summary_coverage_distance_3 = tf.summary.scalar('coverage_distance_3', coverage_distance_3)
    summary_cube_volume_3 = tf.summary.scalar('cube_volume_3', cube_volume_3)
    summary_consistency_distance_3 = tf.summary.scalar('consistency_distance_3', consistency_distance_3)
    summary_mutex_distance_3 = tf.summary.scalar('mutex_distance_3', mutex_distance_3)
    summary_aligning_distance_3 = tf.summary.scalar('aligning_distance_3', aligning_distance_3)
    summary_symmetry_distance_3 = tf.summary.scalar('symmetry_distance_3', symmetry_distance_3)
    summary_list_phase_three = [summary_coverage_distance_3, 
                                summary_cube_volume_3,
                                summary_consistency_distance_3,
                                summary_mutex_distance_3,
                                summary_aligning_distance_3,
                                summary_symmetry_distance_3]

    total_summary_list = [summary_train_loss, summary_lr_scheme] + \
        summary_list_phase_one + summary_list_phase_two + summary_list_phase_three
    train_merged = tf.summary.merge(total_summary_list)

  return train_merged, solver


def test_network():
  data, octree, node_position = data_loader(FLAGS.test_data,
      FLAGS.test_batch_size, n_points, test=True)

  latent_code = encoder(data, octree, is_training=False, reuse=True)
  cube_params_1 = decoder(latent_code, n_part_1, shape_bias_1,
      name='decoder_phase_one', is_training=False, reuse=True)
  cube_params_2 = decoder(latent_code, n_part_2, shape_bias_2,
      name='decoder_phase_two', is_training=False, reuse=True)
  cube_params_3 = decoder(latent_code, n_part_3, shape_bias_3,
      name='decoder_phase_three', is_training=False, reuse=True)

  [test_loss_1, _, _, _, _, _, _, _,
   test_loss_2, _, _, _, _, _, _,
   test_loss_3, _, _, _, _, _, _
  ] = initial_loss_function(cube_params_1, cube_params_2, cube_params_3,
      node_position)

  test_loss = test_loss_1 + test_loss_2 + test_loss_3

  with tf.name_scope('test_summary'):
    average_test_loss = tf.placeholder(tf.float32)
    summary_test_loss = tf.summary.scalar('average_test_loss',
        average_test_loss)
    test_merged = tf.summary.merge([summary_test_loss])

  return_list = [test_merged,
      average_test_loss, test_loss,
      node_position,
      latent_code,
      cube_params_1, cube_params_2, cube_params_3]
  return return_list


def main(argv=None):
  train_summary, solver = train_network()

  [test_summary,
      average_test_loss, test_loss,
      test_node_position,
      test_latent_code,
      cube_params_1, cube_params_2, cube_params_3
      ] = test_network()

  # checkpoint
  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  start_iters = 0 if not ckpt else int(ckpt[ckpt.find('iter') + 4:-5]) + 1

  # saver
  tvars = tf.trainable_variables()
  encoder_vars = [var for var in tvars if 'encoder' in var.name]
  decoder_1_vars = [var for var in tvars if 'phase_one' in var.name]
  decoder_2_vars = [var for var in tvars if 'phase_two' in var.name]
  decoder_3_vars = [var for var in tvars if 'phase_three' in var.name]

  restore_vars = encoder_vars + decoder_1_vars + decoder_2_vars + decoder_3_vars
  save_vars = encoder_vars + decoder_1_vars + decoder_2_vars + decoder_3_vars

  tf_saver = tf.train.Saver(var_list=save_vars, max_to_keep=100)
  if ckpt:
    assert(os.path.exists(FLAGS.ckpt))
    tf_restore_saver = tf.train.Saver(var_list=restore_vars, max_to_keep=100)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # tf summary
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    if ckpt:
      tf_restore_saver.restore(sess, ckpt)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    dump_dir = os.path.join('dump', FLAGS.cache_folder)
    if not os.path.exists(dump_dir): os.makedirs(dump_dir)
    obj_dir = os.path.join('obj', FLAGS.cache_folder)
    if not os.path.exists(obj_dir): os.makedirs(obj_dir)

    if FLAGS.test:
      for it in range(test_iter):
        [test_loss_total_value,
            cube_params_1_value, cube_params_2_value, cube_params_3_value,
            node_position_value,
            latent_code_value
            ] = sess.run([
                test_loss,
                cube_params_1, cube_params_2, cube_params_3,
                test_node_position,
                test_latent_code
                ])
        print('Iter {} loss: {}'.format(it, test_loss_total_value))

        with open(os.path.join(dump_dir, 'cube_1_{:04d}.txt'.format(it)), 'w') as f:
          z = np.reshape(cube_params_1_value[0], [n_part_1, 3])
          q = np.reshape(cube_params_1_value[1], [n_part_1, 4])
          t = np.reshape(cube_params_1_value[2], [n_part_1, 3])
          for j in range(n_part_1):
            f.write('{} {} {} '.format(z[j][0], z[j][1], z[j][2]))
            f.write('{} {} {} {} '.format(q[j][0], q[j][1], q[j][2], q[j][3]))
            f.write('{} {} {}\n'.format(t[j][0], t[j][1], t[j][2]))
        cube_params = np.loadtxt(os.path.join(dump_dir, 'cube_1_{:04d}.txt'.format(it)))
        obj_filename = os.path.join(obj_dir, 'cube_1_{:04d}.obj'.format(it))
        vis_primitive.save_parts(cube_params, obj_filename)

        with open(os.path.join(dump_dir, 'cube_2_{:04d}.txt'.format(it)), 'w') as f:
          z = np.reshape(cube_params_2_value[0], [n_part_2, 3])
          q = np.reshape(cube_params_2_value[1], [n_part_2, 4])
          t = np.reshape(cube_params_2_value[2], [n_part_2, 3])
          for j in range(n_part_2):
            f.write('{} {} {} '.format(z[j][0], z[j][1], z[j][2]))
            f.write('{} {} {} {} '.format(q[j][0], q[j][1], q[j][2], q[j][3]))
            f.write('{} {} {}\n'.format(t[j][0], t[j][1], t[j][2]))
        cube_params = np.loadtxt(os.path.join(dump_dir, 'cube_2_{:04d}.txt'.format(it)))
        obj_filename = os.path.join(obj_dir, 'cube_2_{:04d}.obj'.format(it))
        vis_primitive.save_parts(cube_params, obj_filename)

        with open(os.path.join(dump_dir, 'cube_3_{:04d}.txt'.format(it)), 'w') as f:
          z = np.reshape(cube_params_3_value[0], [n_part_3, 3])
          q = np.reshape(cube_params_3_value[1], [n_part_3, 4])
          t = np.reshape(cube_params_3_value[2], [n_part_3, 3])
          for j in range(n_part_3):
            f.write('{} {} {} '.format(z[j][0], z[j][1], z[j][2]))
            f.write('{} {} {} {} '.format(q[j][0], q[j][1], q[j][2], q[j][3]))
            f.write('{} {} {}\n'.format(t[j][0], t[j][1], t[j][2]))
        cube_params = np.loadtxt(os.path.join(dump_dir, 'cube_3_{:04d}.txt'.format(it)))
        obj_filename = os.path.join(obj_dir, 'cube_3_{:04d}.obj'.format(it))
        vis_primitive.save_parts(cube_params, obj_filename)

        np.savetxt(os.path.join(dump_dir, 'node_position_{:04d}.txt'.format(it)), node_position_value)
        np.savetxt(os.path.join(dump_dir, 'latent_code_{:04d}.txt'.format(it)), np.reshape(latent_code_value, [-1]))
        # pc_filename = os.path.join(obj_dir, 'pc_{:04d}.obj'.format(it))
        # vis_pointcloud.save_points(np.transpose(node_position_value),
        #     pc_filename, depth=6)

    else:
      # start training
      for i in range(start_iters, max_iter):
        if coord.should_stop():
            break

        if i % FLAGS.test_every_n_steps == 0:
          avg_test_loss = 0
          for it in range(test_iter):
            [test_loss_total_value,
                cube_params_1_value, cube_params_2_value, cube_params_3_value,
                node_position_value
                ] = sess.run([
                    test_loss,
                    cube_params_1, cube_params_2, cube_params_3,
                    test_node_position])
            avg_test_loss += test_loss_total_value

            if i % FLAGS.disp_every_n_steps == 0:
              with open(os.path.join(dump_dir, 'cube_1_{:06d}_{:04d}.txt'.format(i, it)), 'w') as f:
                z = np.reshape(cube_params_1_value[0], [n_part_1, 3])
                q = np.reshape(cube_params_1_value[1], [n_part_1, 4])
                t = np.reshape(cube_params_1_value[2], [n_part_1, 3])
                for j in range(n_part_1):
                  f.write('{} {} {} '.format(z[j][0], z[j][1], z[j][2]))
                  f.write('{} {} {} {} '.format(q[j][0], q[j][1], q[j][2], q[j][3]))
                  f.write('{} {} {}\n'.format(t[j][0], t[j][1], t[j][2]))
              cube_params = np.loadtxt(os.path.join(dump_dir, 'cube_1_{:06d}_{:04d}.txt'.format(i, it)))
              obj_filename = os.path.join(obj_dir, 'cube_1_{:06d}_{:04d}.obj'.format(i, it))
              vis_primitive.save_parts(cube_params, obj_filename)

              with open(os.path.join(dump_dir, 'cube_2_{:06d}_{:04d}.txt'.format(i, it)), 'w') as f:
                z = np.reshape(cube_params_2_value[0], [n_part_2, 3])
                q = np.reshape(cube_params_2_value[1], [n_part_2, 4])
                t = np.reshape(cube_params_2_value[2], [n_part_2, 3])
                for j in range(n_part_2):
                  f.write('{} {} {} '.format(z[j][0], z[j][1], z[j][2]))
                  f.write('{} {} {} {} '.format(q[j][0], q[j][1], q[j][2], q[j][3]))
                  f.write('{} {} {}\n'.format(t[j][0], t[j][1], t[j][2]))
              cube_params = np.loadtxt(os.path.join(dump_dir, 'cube_2_{:06d}_{:04d}.txt'.format(i, it)))
              obj_filename = os.path.join(obj_dir, 'cube_2_{:06d}_{:04d}.obj'.format(i, it))
              vis_primitive.save_parts(cube_params, obj_filename)

              with open(os.path.join(dump_dir, 'cube_3_{:06d}_{:04d}.txt'.format(i, it)), 'w') as f:
                z = np.reshape(cube_params_3_value[0], [n_part_3, 3])
                q = np.reshape(cube_params_3_value[1], [n_part_3, 4])
                t = np.reshape(cube_params_3_value[2], [n_part_3, 3])
                for j in range(n_part_3):
                  f.write('{} {} {} '.format(z[j][0], z[j][1], z[j][2]))
                  f.write('{} {} {} {} '.format(q[j][0], q[j][1], q[j][2], q[j][3]))
                  f.write('{} {} {}\n'.format(t[j][0], t[j][1], t[j][2]))
              cube_params = np.loadtxt(os.path.join(dump_dir, 'cube_3_{:06d}_{:04d}.txt'.format(i, it)))
              obj_filename = os.path.join(obj_dir, 'cube_3_{:06d}_{:04d}.obj'.format(i, it))
              vis_primitive.save_parts(cube_params, obj_filename)

              pc_filename = os.path.join(obj_dir, 'pc_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(np.transpose(node_position_value),
                  pc_filename, depth=6)

          avg_test_loss /= test_iter

          summary = sess.run(test_summary, feed_dict={average_test_loss: avg_test_loss})
          summary_writer.add_summary(summary, i)
          if i % (FLAGS.disp_every_n_steps) == 0:
            tf_saver.save(sess, os.path.join(FLAGS.log_dir, 'model/iter{:06d}.ckpt'.format(i)))

        summary, _ = sess.run([train_summary, solver])
        summary_writer.add_summary(summary, i)

    # finished training
    coord.request_stop()
    coord.join(threads=threads)


if __name__ == '__main__':
  tf.app.run()
