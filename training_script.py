import argparse
import datetime
import os
import sys

shape_count = {
  'airplane': {
    'train': 2832,
    'test': 808,
    'n_part': 4
  },
  'chair': {
    'train': 4612,
    'test': 1317,
    'n_part': 8
  },
  'table': {
    'train': 5876,
    'test': 1679,
    'n_part': 3
  },
  'animal': {
    'train': 122*20, # random duplication 20 times
    'test': 122,
    'n_part': 5
  }
}

# initial training params
initial_training_iter = 100001
coverage_weight = 1
consistency_weight = 1
mutex_weight = 1
aligning_weight = 0.001
symmetry_weight = 0.1
area_average_weight = 5
# iterative training params
iterative_round = 4
iterative_training_iter = 20001
sparseness_weight = 0.4
completeness_weight = 1
similarity_weight = 0.1
selected_tree_weight = 1
mask_weight = 0.02


def run_training_pipeline(category, gpu):
  batch_size = 32
  n_part_1 = shape_count[category]['n_part']*4
  n_part_2 = shape_count[category]['n_part']*2
  n_part_3 = shape_count[category]['n_part']*1
  train_data = os.path.join('data', '{}_octree_points_d5_train.tfrecords'.format(category))
  test_data = os.path.join('data', '{}_octree_points_d5_test_100.tfrecords'.format(category))
  test_iter = 100
  if category == 'animal':
    train_data = os.path.join('data', 'augment_{}_octree_points_d5.tfrecords'.format(category))
    test_data = os.path.join('data', '{}_octree_points_d5.tfrecords'.format(category))
    test_iter = 122


  ## initial training stage ----------------------------------------------------
  log_folder = 'log/{}_{}_{}_{}/initial_training'.format(category, n_part_1, n_part_2, n_part_3)
  if not os.path.exists(log_folder): os.makedirs(log_folder);
  os.system('python initial_training.py --log_dir {} --train_data {} --test_data {} --max_iter {} --test_iter {} --n_part_1 {} --n_part_2 {} --n_part_3 {} --coverage_weight {} --consistency_weight {} --mutex_weight {} --aligning_weight {} --symmetry_weight {} --area_average_weight {} --cache_folder {} --gpu {} > {} 2>&1'.format(log_folder, train_data, test_data, initial_training_iter, test_iter, n_part_1, n_part_2, n_part_3, coverage_weight, consistency_weight, mutex_weight, aligning_weight, symmetry_weight, area_average_weight, 'test_{}_initial'.format(category), gpu, os.path.join(log_folder, 'initial_training_{}.log'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))))
  former_log_folder = log_folder
  ## ---------------------------------------------------------------------------

  ## iterative training stage --------------------------------------------------
  for i in range(iterative_round):
    global selected_tree_weight
    # mask predict
    log_folder = 'log/{}_{}_{}_{}/iterative_training/mask_predict_{}'.format(category, n_part_1, n_part_2, n_part_3, i)
    if not os.path.exists(log_folder): os.makedirs(log_folder);
    os.system('python iterative_training.py --log_dir {} --train_data {} --test_data {} --max_iter {} --test_iter {} --n_part_1 {} --n_part_2 {} --n_part_3 {} --sparseness_weight {} --completeness_weight {} --similarity_weight {} --cache_folder {} --ckpt {} --stage mask_predict --gpu {} > {} 2>&1'.format(log_folder, train_data, test_data, iterative_training_iter, test_iter, n_part_1, n_part_2, n_part_3, sparseness_weight, completeness_weight, similarity_weight, 'test_{}_iterative_mask_{}'.format(category, i), os.path.join(former_log_folder, 'model'), gpu, os.path.join(log_folder, 'iterative_training_mask_{}_{}.log'.format(i, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))))
    former_log_folder = log_folder
    # cube update
    log_folder = 'log/{}_{}_{}_{}/iterative_training/cube_update_{}'.format(category, n_part_1, n_part_2, n_part_3, i)
    if not os.path.exists(log_folder): os.makedirs(log_folder);
    os.system('python iterative_training.py --log_dir {} --train_data {} --test_data {} --max_iter {} --test_iter {} --n_part_1 {} --n_part_2 {} --n_part_3 {} --selected_tree_weight {} --cache_folder {} --ckpt {} --stage cube_update --gpu {} > {} 2>&1'.format(log_folder, train_data, test_data, iterative_training_iter, test_iter, n_part_1, n_part_2, n_part_3, selected_tree_weight, 'test_{}_iterative_update_{}'.format(category, i), os.path.join(former_log_folder, 'model'), gpu, os.path.join(log_folder, 'iterative_training_update_{}_{}.log'.format(i, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))))
    former_log_folder = log_folder
    selected_tree_weight *= 2
  # finetune
  log_folder = 'log/{}_{}_{}_{}/iterative_training/finutune'.format(category, n_part_1, n_part_2, n_part_3)
  if not os.path.exists(log_folder): os.makedirs(log_folder);
  os.system('python iterative_training.py --log_dir {} --train_data {} --test_data {} --learning_rate {} --max_iter {} --test_iter {} --n_part_1 {} --n_part_2 {} --n_part_3 {} --coverage_weight {} --consistency_weight {} --mutex_weight {} --aligning_weight {} --symmetry_weight {} --area_average_weight {} --selected_tree_weight {} --mask_weight {} --cache_folder {} --ckpt {} --stage finetune --gpu {} > {} 2>&1'.format(log_folder, train_data, test_data, 0.00001, iterative_training_iter, test_iter, n_part_1, n_part_2, n_part_3, coverage_weight, consistency_weight, mutex_weight, aligning_weight, symmetry_weight, area_average_weight, selected_tree_weight/2, mask_weight, 'test_{}_iterative_finetune'.format(category), os.path.join(former_log_folder, 'model'), gpu, os.path.join(log_folder, 'iterative_training_finetune_{}_{}.log'.format(i, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))))  ## ---------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--category',
                      type=str,
                      default='airplane',
                      help='category')
  parser.add_argument('--gpu',
                      type=int,
                      default=0,
                      help='gpu index')
  args = parser.parse_args()
  run_training_pipeline(args.category, args.gpu)
  pass


if __name__ == '__main__':
  main()
