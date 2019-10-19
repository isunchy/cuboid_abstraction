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
from mask_predict_net import *
from hierarchical_primitive.hierarchical_primitive import vis_assembly_cube


tf.app.flags.DEFINE_string('log_dir', 'log/mask_predict/PGenMask_xxxx/0_16_8_4_airplane_0',
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
tf.app.flags.DEFINE_integer('max_iter', 50010,
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
tf.app.flags.DEFINE_float('sparseness_weight', 0.4,
                          """Weight of mask sparseness loss""")
tf.app.flags.DEFINE_float('completeness_weight', 1,
                          """Weight of mask completeness loss""")
tf.app.flags.DEFINE_float('similarity_weight', 0.1,
                          """Weight of shape similarity loss""")
tf.app.flags.DEFINE_float('selected_tree_weight', 1,
                          """Weight of selected tree, weight of original tree is 1""")
tf.app.flags.DEFINE_float('mask_weight', 0.02,
                          """Weight of mask loss, fitting loss is 1""")
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
                            """Number of points sampled on original shape.""")
tf.app.flags.DEFINE_string('stage', 'mask_predict',
                            """Which stage of the network, [mask_predict,
                            cube_update, finetune]""")


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

  return loss_1 + loss_2 + loss_3


def mask_predict_loss_function(logit_1, logit_2, logit_3, cube_params_1,
    cube_params_2, cube_params_3, node_position):
  with tf.name_scope('mask_predict_loss_function'):
    sparseness_loss = mask_sparseness_loss(logit_1, logit_2, logit_3)
    similarity_loss, relation_12, relation_23 = shape_similarity_loss(logit_1,
        logit_2, logit_3, cube_params_1, cube_params_2, cube_params_3,
        node_position, n_part_1, n_part_2)
    completeness_loss = mask_completeness_loss(logit_1, logit_2, logit_3,
        relation_12, relation_23)
    loss = (FLAGS.sparseness_weight*sparseness_loss +
            FLAGS.similarity_weight*similarity_loss + 
            FLAGS.completeness_weight*completeness_loss)
  return [loss, sparseness_loss, similarity_loss, completeness_loss]


def cube_update_loss_function(logit_1, logit_2, logit_3, cube_params_1,
    cube_params_2, cube_params_3, node_position):
  with tf.name_scope('cube_update_loss_function'):
    logit = tf.concat([logit_1, logit_2, logit_3], axis=1)
    mask = tf.cast(logit > 0.5, tf.int32)
    _, _, relation_12 = cube_coverage_loss(cube_params_1, cube_params_2, n_part_1,
        node_position) # [bs, n_part_1]
    _, _, relation_23 = cube_coverage_loss(cube_params_2, cube_params_3, n_part_2,
        node_position) # [bs, n_part_2]
    mask_1, mask_2, mask_3 = primitive_tree_generation(mask, relation_12,
        relation_23, n_part_1, n_part_2, n_part_3)
    cube_params_z = tf.concat([cube_params_1[0], cube_params_2[0], cube_params_3[0]], axis=1)
    cube_params_q = tf.concat([cube_params_1[1], cube_params_2[1], cube_params_3[1]], axis=1)
    cube_params_t = tf.concat([cube_params_1[2], cube_params_2[2], cube_params_3[2]], axis=1)
    cube_params = [cube_params_z, cube_params_q, cube_params_t]
    [selected_coverage_distance_1, selected_consistency_distance_1,
        selected_mutex_distance_1] = compute_selected_tree_loss(
            cube_params, mask_1, node_position, phase='one')
    selected_tree_loss_1 = (selected_coverage_distance_1 * FLAGS.coverage_weight +
                            selected_consistency_distance_1 * FLAGS.consistency_weight +
                            selected_mutex_distance_1 * FLAGS.mutex_weight)
    [selected_coverage_distance_2, selected_consistency_distance_2,
        selected_mutex_distance_2] = compute_selected_tree_loss(
            cube_params, mask_2, node_position, phase='two')
    selected_tree_loss_2 = (selected_coverage_distance_2 * FLAGS.coverage_weight +
                            selected_consistency_distance_2 * FLAGS.consistency_weight +
                            selected_mutex_distance_2 * FLAGS.mutex_weight)
    [selected_coverage_distance_3, selected_consistency_distance_3,
        selected_mutex_distance_3] = compute_selected_tree_loss(
            cube_params, mask_3, node_position, phase='three')
    selected_tree_loss_3 = (selected_coverage_distance_3 * FLAGS.coverage_weight +
                            selected_consistency_distance_3 * FLAGS.consistency_weight +
                            selected_mutex_distance_3 * FLAGS.mutex_weight)
  return [selected_tree_loss_1,
          selected_coverage_distance_1,
          selected_consistency_distance_1,
          selected_mutex_distance_1,
          selected_tree_loss_2,
          selected_coverage_distance_2,
          selected_consistency_distance_2,
          selected_mutex_distance_2,
          selected_tree_loss_3,
          selected_coverage_distance_3,
          selected_consistency_distance_3,
          selected_mutex_distance_3,
          mask_1, mask_2, mask_3
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

  logit_1 = mask_predict_net(latent_code, n_part_1, name='phase_1',
      is_training=True, reuse=False)
  logit_2 = mask_predict_net(latent_code, n_part_2, name='phase_2',
      is_training=True, reuse=False)
  logit_3 = mask_predict_net(latent_code, n_part_3, name='phase_3',
      is_training=True, reuse=False)

  mask_predict_loss, sparseness_loss, similarity_loss, completeness_loss = \
      mask_predict_loss_function(
          logit_1, logit_2, logit_3,
          cube_params_1, cube_params_2, cube_params_3,
          node_position
          )
  original_tree_loss = initial_loss_function(cube_params_1, cube_params_2,
      cube_params_3, node_position)
  [selected_tree_loss_1,
   selected_coverage_distance_1,
   selected_consistency_distance_1,
   selected_mutex_distance_1,
   selected_tree_loss_2,
   selected_coverage_distance_2,
   selected_consistency_distance_2,
   selected_mutex_distance_2,
   selected_tree_loss_3,
   selected_coverage_distance_3,
   selected_consistency_distance_3,
   selected_mutex_distance_3,
   _, _, _
  ] = cube_update_loss_function(logit_1, logit_2, logit_3, cube_params_1,
      cube_params_2, cube_params_3, node_position)
  selected_tree_loss = selected_tree_loss_1 + selected_tree_loss_2 + selected_tree_loss_3  
  fitting_loss = selected_tree_loss * FLAGS.selected_tree_weight + original_tree_loss

  tvars = tf.trainable_variables()
  encoder_vars = [var for var in tvars if 'encoder' in var.name]
  decoder_vars = [var for var in tvars if 'decoder' in var.name]
  mask_predict_vars = [var for var in tvars if 'mask_predict' in var.name]
  
  if FLAGS.stage == 'mask_predict':
    train_loss = mask_predict_loss
    var_list = mask_predict_vars
  elif FLAGS.stage == 'cube_update':
    train_loss = fitting_loss
    var_list = decoder_vars
  elif FLAGS.stage == 'finetune':
    train_loss = fitting_loss + mask_predict_loss*FLAGS.mask_weight
    var_list = encoder_vars# + decoder_vars
  else:
    raise ValueError('[{}] is an invalid training stage'.format(FLAGS.stage))

  with tf.name_scope('train_summary'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      solver = optimizer.minimize(train_loss, var_list=var_list)
      lr = optimizer._lr
      summary_lr_scheme = tf.summary.scalar('learning_rate', lr)
      summary_train_loss = tf.summary.scalar('train_loss', train_loss)

    summary_sparseness_loss = tf.summary.scalar('sparseness_loss', sparseness_loss)
    summary_similarity_loss = tf.summary.scalar('similarity_loss', similarity_loss)
    summary_completeness_loss = tf.summary.scalar('completeness_loss', completeness_loss)
    summary_selected_tree_loss = tf.summary.scalar('selected_tree_loss', selected_tree_loss)
    summary_original_tree_loss = tf.summary.scalar('original_tree_loss', original_tree_loss)

    summary_logit_1_histogram = tf.summary.histogram('logit_1', logit_1)
    summary_logit_2_histogram = tf.summary.histogram('logit_2', logit_2)
    summary_logit_3_histogram = tf.summary.histogram('logit_3', logit_3)

    summary_selected_coverage_distance_1 = tf.summary.scalar('selected_coverage_distance_1', selected_coverage_distance_1)
    summary_selected_consistency_distance_1 = tf.summary.scalar('selected_consistency_distance_1', selected_consistency_distance_1)
    summary_selected_mutex_distance_1 = tf.summary.scalar('selected_mutex_distance_1', selected_mutex_distance_1)
    summary_list_phase_one = [summary_selected_coverage_distance_1,
                              summary_selected_consistency_distance_1,
                              summary_selected_mutex_distance_1]

    summary_selected_coverage_distance_2 = tf.summary.scalar('selected_coverage_distance_2', selected_coverage_distance_2)
    summary_selected_consistency_distance_2 = tf.summary.scalar('selected_consistency_distance_2', selected_consistency_distance_2)
    summary_selected_mutex_distance_2 = tf.summary.scalar('selected_mutex_distance_2', selected_mutex_distance_2)
    summary_list_phase_two = [summary_selected_coverage_distance_2,
                              summary_selected_consistency_distance_2,
                              summary_selected_mutex_distance_2]

    summary_selected_coverage_distance_3 = tf.summary.scalar('selected_coverage_distance_3', selected_coverage_distance_3)
    summary_selected_consistency_distance_3 = tf.summary.scalar('selected_consistency_distance_3', selected_consistency_distance_3)
    summary_selected_mutex_distance_3 = tf.summary.scalar('selected_mutex_distance_3', selected_mutex_distance_3)
    summary_list_phase_three = [summary_selected_coverage_distance_3,
                                summary_selected_consistency_distance_3,
                                summary_selected_mutex_distance_3]

    total_summary_list = [
        summary_train_loss,
        summary_lr_scheme,
        summary_sparseness_loss,
        summary_similarity_loss,
        summary_completeness_loss,
        summary_selected_tree_loss,
        summary_original_tree_loss,
        summary_logit_1_histogram,
        summary_logit_2_histogram,
        summary_logit_3_histogram
        ] + summary_list_phase_one + summary_list_phase_two + summary_list_phase_three
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

  logit_1 = mask_predict_net(latent_code, n_part_1, name='phase_1',
      is_training=False, reuse=True)
  logit_2 = mask_predict_net(latent_code, n_part_2, name='phase_2',
      is_training=False, reuse=True)
  logit_3 = mask_predict_net(latent_code, n_part_3, name='phase_3',
      is_training=False, reuse=True)

  predict_1 = tf.cast(logit_1 > 0.5, tf.int32)
  predict_2 = tf.cast(logit_2 > 0.5, tf.int32)
  predict_3 = tf.cast(logit_3 > 0.5, tf.int32)

  mask_predict_loss, sparseness_loss, similarity_loss, completeness_loss = \
      mask_predict_loss_function(
          logit_1, logit_2, logit_3,
          cube_params_1, cube_params_2, cube_params_3,
          node_position
          )
  original_tree_loss = initial_loss_function(cube_params_1, cube_params_2,
      cube_params_3, node_position)
  [selected_tree_loss_1,
   selected_coverage_distance_1,
   selected_consistency_distance_1,
   selected_mutex_distance_1,
   selected_tree_loss_2,
   selected_coverage_distance_2,
   selected_consistency_distance_2,
   selected_mutex_distance_2,
   selected_tree_loss_3,
   selected_coverage_distance_3,
   selected_consistency_distance_3,
   selected_mutex_distance_3,
   mask_1, mask_2, mask_3
  ] = cube_update_loss_function(logit_1, logit_2, logit_3, cube_params_1,
      cube_params_2, cube_params_3, node_position)
  selected_tree_loss = selected_tree_loss_1 + selected_tree_loss_2 + selected_tree_loss_3  
  fitting_loss = selected_tree_loss * FLAGS.selected_tree_weight + original_tree_loss
  
  if FLAGS.stage == 'mask_predict':
    test_loss = mask_predict_loss
  elif FLAGS.stage == 'cube_update':
    test_loss = fitting_loss
  elif FLAGS.stage == 'finetune':
    test_loss = fitting_loss + mask_predict_loss * FLAGS.mask_weight
  else:
    raise ValueError('[{}] is an invalid training stage'.format(FLAGS.stage))

  with tf.name_scope('test_summary'):
    average_test_loss = tf.placeholder(tf.float32)
    summary_test_loss = tf.summary.scalar('test_loss', average_test_loss)
    average_test_sparseness_loss = tf.placeholder(tf.float32)
    summary_test_sparseness_loss = tf.summary.scalar('sparseness_loss',
        average_test_sparseness_loss)
    average_test_similarity_loss = tf.placeholder(tf.float32)
    summary_test_similarity_loss = tf.summary.scalar('similarity_loss',
        average_test_similarity_loss)
    average_test_completeness_loss = tf.placeholder(tf.float32)
    summary_test_completeness_loss = tf.summary.scalar('completeness_loss',
        average_test_completeness_loss)
    average_test_selected_tree_loss = tf.placeholder(tf.float32)
    summary_test_selected_tree_loss = tf.summary.scalar('selected_tree_loss',
        average_test_selected_tree_loss)
    average_test_original_tree_loss = tf.placeholder(tf.float32)
    summary_test_original_tree_loss = tf.summary.scalar('original_tree_loss',
        average_test_original_tree_loss)
    test_merged = tf.summary.merge([summary_test_loss,
                                    summary_test_sparseness_loss,
                                    summary_test_similarity_loss,
                                    summary_test_completeness_loss,
                                    summary_test_selected_tree_loss,
                                    summary_test_original_tree_loss])

  return_list = [test_merged,
                 logit_1, logit_2, logit_3,
                 predict_1, predict_2, predict_3,
                 sparseness_loss,
                 similarity_loss,
                 completeness_loss,
                 selected_tree_loss,
                 original_tree_loss,
                 test_loss,
                 average_test_sparseness_loss,
                 average_test_similarity_loss,
                 average_test_completeness_loss,
                 average_test_selected_tree_loss,
                 average_test_original_tree_loss,
                 average_test_loss,
                 node_position,
                 latent_code,
                 cube_params_1, cube_params_2, cube_params_3,
                 mask_1, mask_2, mask_3]
  return return_list


def main(argv=None):
  train_summary, solver = train_network()

  [test_summary,
      test_logit_1, test_logit_2, test_logit_3,
      test_predict_1, test_predict_2, test_predict_3,
      test_sparseness_loss,
      test_similarity_loss,
      test_completeness_loss,
      selected_tree_loss,
      original_tree_loss,
      test_loss,
      average_test_sparseness_loss,
      average_test_similarity_loss,
      average_test_completeness_loss,
      average_test_selected_tree_loss,
      average_test_original_tree_loss,
      average_test_loss,
      test_node_position,
      test_latent_code,
      cube_params_1, cube_params_2, cube_params_3,
      mask_1, mask_2, mask_3] = test_network()

  # checkpoint
  assert(os.path.exists(FLAGS.ckpt))
  ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
  start_iters = 0
  if FLAGS.stage == 'mask_predict' and 'mask_predict' in FLAGS.ckpt:
    start_iters = int(ckpt[ckpt.find('iter') + 4:-5]) + 1
  if FLAGS.stage == 'cube_update' and 'cube_update' in FLAGS.ckpt:
    start_iters = int(ckpt[ckpt.find('iter') + 4:-5]) + 1
  if FLAGS.stage == 'finetune' and 'finetune' in FLAGS.ckpt:
    start_iters = int(ckpt[ckpt.find('iter') + 4:-5]) + 1

  # saver
  tvars = tf.trainable_variables()
  encoder_vars = [var for var in tvars if 'encoder' in var.name]
  decoder_vars = [var for var in tvars if 'decoder' in var.name]
  mask_predict_vars = [var for var in tvars if 'mask_predict' in var.name]

  restore_vars = decoder_vars + encoder_vars
  # in the first round of mask predict stage, decoder variables are unaviliable
  if 'initial_training' not in FLAGS.ckpt:
    restore_vars += mask_predict_vars
  save_vars = encoder_vars + decoder_vars + mask_predict_vars

  tf_saver = tf.train.Saver(var_list=save_vars, max_to_keep=100)
  tf_restore_saver = tf.train.Saver(var_list=restore_vars, max_to_keep=100)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # tf summary
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    tf_restore_saver.restore(sess, ckpt)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    dump_dir = os.path.join('dump', FLAGS.cache_folder)
    if not os.path.exists(dump_dir): os.makedirs(dump_dir)
    obj_dir = os.path.join('obj', FLAGS.cache_folder)
    if not os.path.exists(obj_dir): os.makedirs(obj_dir)

    if FLAGS.test:
      output_latent_codes = []
      output_logit_1 = []
      output_logit_2 = []
      output_logit_3 = []
      output_predict_1 = []
      output_predict_2 = []
      output_predict_3 = []
      output_key_1 = []
      output_key_2 = []
      output_key_3 = []

      for it in range(test_iter):
        [test_logit_1_value, test_logit_2_value, test_logit_3_value,
            test_predict_1_value, test_predict_2_value, test_predict_3_value,
            test_loss_value,
            test_node_position_value,
            test_latent_code_value,
            cube_params_1_value,
            cube_params_2_value,
            cube_params_3_value,
            mask_1_value,
            mask_2_value,
            mask_3_value] = sess.run([
                test_logit_1, test_logit_2, test_logit_3,
                test_predict_1, test_predict_2, test_predict_3,
                test_loss,
                test_node_position,
                test_latent_code,
                cube_params_1, cube_params_2, cube_params_3,
                mask_1, mask_2, mask_3])
        print('Iter {} loss: {}'.format(it, test_loss_value))
        sys.stdout.flush()

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
        vis_primitive.save_parts(cube_params, obj_filename, level='1')

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
        vis_primitive.save_parts(cube_params, obj_filename, level='2')

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
        vis_primitive.save_parts(cube_params, obj_filename, level='3')

        output_latent_codes.append(test_latent_code_value.flatten())
        output_logit_1.append(test_logit_1_value.flatten())
        output_logit_2.append(test_logit_2_value.flatten())
        output_logit_3.append(test_logit_3_value.flatten())
        output_predict_1.append(test_predict_1_value.flatten())
        output_predict_2.append(test_predict_2_value.flatten())
        output_predict_3.append(test_predict_3_value.flatten())
        output_key_1.append(mask_1_value.flatten())
        output_key_2.append(mask_2_value.flatten())
        output_key_3.append(mask_3_value.flatten())

        np.savetxt(os.path.join(dump_dir, 'latent_code_{:04d}.txt'.format(it)), np.reshape(test_latent_code_value, [-1]))
        np.savetxt(os.path.join(dump_dir, 'predict_mask_1_{:04d}.txt'.format(it)), test_predict_1_value)
        np.savetxt(os.path.join(dump_dir, 'predict_mask_2_{:04d}.txt'.format(it)), test_predict_2_value)
        np.savetxt(os.path.join(dump_dir, 'predict_mask_3_{:04d}.txt'.format(it)), test_predict_3_value)
        vis_assembly_cube(obj_dir, '{:04d}'.format(it), dump_dir, '{:04d}'.format(it), obj_dir, with_correction=True)
        
        # pc_filename = os.path.join(obj_dir, 'pc_{:04d}.obj'.format(it))
        # vis_pointcloud.save_points(np.transpose(test_node_position_value),
        #     pc_filename, depth=6)

      np.savetxt(os.path.join(dump_dir, 'latent_codes.txt'), np.array(output_latent_codes))
      np.savetxt(os.path.join(dump_dir, 'mask_logit_0.txt'), np.array(output_logit_1))
      np.savetxt(os.path.join(dump_dir, 'mask_logit_1.txt'), np.array(output_logit_2))
      np.savetxt(os.path.join(dump_dir, 'mask_logit_2.txt'), np.array(output_logit_3))
      np.savetxt(os.path.join(dump_dir, 'mask_predict_0.txt'), np.array(output_predict_1, dtype=int))
      np.savetxt(os.path.join(dump_dir, 'mask_predict_1.txt'), np.array(output_predict_2, dtype=int))
      np.savetxt(os.path.join(dump_dir, 'mask_predict_2.txt'), np.array(output_predict_3, dtype=int))
      np.savetxt(os.path.join(dump_dir, 'keys_0.txt'), np.array(output_key_1, dtype=int), fmt='%i')
      np.savetxt(os.path.join(dump_dir, 'keys_1.txt'), np.array(output_key_2, dtype=int), fmt='%i')
      np.savetxt(os.path.join(dump_dir, 'keys_2.txt'), np.array(output_key_3, dtype=int), fmt='%i')

    else:
      # start training
      for i in range(start_iters, max_iter):
        if coord.should_stop():
            break

        if i % FLAGS.test_every_n_steps == 0:
          avg_test_sparseness_loss = 0
          avg_test_similarity_loss = 0
          avg_test_completeness_loss = 0
          avg_test_selected_tree_loss = 0
          avg_test_original_tree_loss = 0
          avg_test_loss = 0
          for it in range(test_iter):
            [test_sparseness_loss_value,
                test_similarity_loss_value,
                test_completeness_loss_value,
                test_selected_tree_loss,
                test_original_tree_loss,
                test_loss_value,
                test_logit_1_value, test_logit_2_value, test_logit_3_value,
                test_predict_1_value, test_predict_2_value, test_predict_3_value,
                test_node_position_value,
                test_latent_code_value,
                cube_params_1_value,
                cube_params_2_value,
                cube_params_3_value] = sess.run([
                    test_sparseness_loss,
                    test_similarity_loss,
                    test_completeness_loss,
                    selected_tree_loss,
                    original_tree_loss,
                    test_loss,
                    test_logit_1, test_logit_2, test_logit_3,
                    test_predict_1, test_predict_2, test_predict_3,
                    test_node_position,
                    test_latent_code,
                    cube_params_1, cube_params_2, cube_params_3
                    ])
            avg_test_sparseness_loss += test_sparseness_loss_value
            avg_test_similarity_loss += test_similarity_loss_value
            avg_test_completeness_loss += test_completeness_loss_value
            avg_test_selected_tree_loss += test_selected_tree_loss
            avg_test_original_tree_loss += test_original_tree_loss
            avg_test_loss += test_loss_value

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
              vis_primitive.save_parts(cube_params, obj_filename, level='1')

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
              vis_primitive.save_parts(cube_params, obj_filename, level='2')

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
              vis_primitive.save_parts(cube_params, obj_filename, level='3')

              pc_filename = os.path.join(obj_dir, 'pc_{:06d}_{:04d}.obj'.format(i, it))
              vis_pointcloud.save_points(np.transpose(test_node_position_value),
                  pc_filename, depth=6)

              np.savetxt(os.path.join(dump_dir, 'predict_logit_1_{:06d}_{:04d}.txt'.format(i, it)), test_logit_1_value)
              np.savetxt(os.path.join(dump_dir, 'predict_logit_2_{:06d}_{:04d}.txt'.format(i, it)), test_logit_2_value)
              np.savetxt(os.path.join(dump_dir, 'predict_logit_3_{:06d}_{:04d}.txt'.format(i, it)), test_logit_3_value)
              np.savetxt(os.path.join(dump_dir, 'predict_mask_1_{:06d}_{:04d}.txt'.format(i, it)), test_predict_1_value)
              np.savetxt(os.path.join(dump_dir, 'predict_mask_2_{:06d}_{:04d}.txt'.format(i, it)), test_predict_2_value)
              np.savetxt(os.path.join(dump_dir, 'predict_mask_3_{:06d}_{:04d}.txt'.format(i, it)), test_predict_3_value)
              vis_assembly_cube(obj_dir, '{:06d}_{:04d}'.format(i, it), dump_dir, '{:06d}_{:04d}'.format(i, it), obj_dir, with_correction=True)

          avg_test_sparseness_loss /= test_iter
          avg_test_similarity_loss /= test_iter
          avg_test_completeness_loss /= test_iter
          avg_test_selected_tree_loss /= test_iter
          avg_test_original_tree_loss /= test_iter
          avg_test_loss /= test_iter

          summary = sess.run(test_summary, 
              feed_dict={average_test_sparseness_loss: avg_test_sparseness_loss,
                         average_test_similarity_loss: avg_test_similarity_loss,
                         average_test_completeness_loss: avg_test_completeness_loss,
                         average_test_selected_tree_loss: avg_test_selected_tree_loss,
                         average_test_original_tree_loss: avg_test_selected_tree_loss,
                         average_test_loss: avg_test_loss
                         })
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
