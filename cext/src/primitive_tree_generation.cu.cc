#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

static __global__ void correct_tree_mask_kernal(const int nthreads,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_1, const int* in_relation_2, int* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n_part_sum = n_part_1 + n_part_2 + n_part_3;
    int* mask_ptr = mask + index*n_part_sum;
    int* mask_1_ptr = mask_ptr;
    int* mask_2_ptr = mask_ptr + n_part_1;
    int* mask_3_ptr = mask_ptr + n_part_1 + n_part_2;
    const int* relation_1_ptr = in_relation_1 + index*n_part_1;
    const int* relation_2_ptr = in_relation_2 + index*n_part_2;
    int i, j, k;
    // when level 3 is 1, if it has no children in level 2, or its children has
    // no children in level 1, set level 3 as 0
    for (i = 0; i < n_part_3; ++i) {
      if (mask_3_ptr[i] == 1) {
        mask_3_ptr[i] = 0;
        for (j = 0; j < n_part_2; ++j) {
          if (relation_2_ptr[j] == i) {
            for (k = 0; k < n_part_1; ++k) {
              if (relation_1_ptr[k] == j) {
                mask_3_ptr[i] = 1;
                break;
              }
            }
            if (mask_3_ptr[i] == 1) {
              break;
            }
          }
        }
      }
    }
    // when level 2 is 1, if it has no children in level 1, set level 2 as 0
    for (i = 0; i < n_part_2; ++i) {
      if (mask_2_ptr[i] == 1) {
        mask_2_ptr[i] = 0;
        for (j = 0; j < n_part_1; ++j) {
          if (relation_1_ptr[j] == i) {
            mask_2_ptr[i] = 1;
            break;
          }
        }
      }
    }
    // when level 3 is 1, set its next two level children as 0
    for (i = 0; i < n_part_3; ++i) {
      if (mask_3_ptr[i] == 1) {
        for (j = 0; j < n_part_2; ++j) {
          if (relation_2_ptr[j] == i) {
            mask_2_ptr[j] = 0;
            for (k = 0; k < n_part_1; ++k) {
              if (relation_1_ptr[k] == j) {
                mask_1_ptr[k] = 0;
              }
            }
          }
        }
      }
    }
    // when level 2 is 1, set his children as 0
    for (i = 0; i < n_part_2; ++i) {
      if (mask_2_ptr[i] == 1) {
        for (j = 0; j < n_part_1; ++j) {
          if (relation_1_ptr[j] == i) {
            mask_1_ptr[j] = 0;
          }
        }
      }
    }
    // complete tree
    for (i = 0; i < n_part_1; ++i) {
      // the cube index of the path from leaf to root w.r.t. the ith cube in level 1
      int level_1_index = i;
      int level_2_index = relation_1_ptr[i];
      int level_3_index = relation_2_ptr[level_2_index];
      int fill_level = 0;
      if (mask_1_ptr[level_1_index] + mask_2_ptr[level_2_index] + 
          mask_3_ptr[level_3_index] == 0) {
        fill_level = 2;
        // scan level 1 to find if other sibling cube is picked
        for (j = 0; j < n_part_1; ++j) {
          if (j != level_1_index && relation_1_ptr[j] == level_2_index) {
            if (mask_1_ptr[j] == 1) {
              fill_level = 1;
              break;
            }
          }
        }
      }
      // if no sibling cube is picked in level 1, check level 2
      if (fill_level == 2) {
        fill_level = 3;
        for (j = 0; j < n_part_2; ++j) {
          if (j != level_2_index && relation_2_ptr[j] == level_3_index) {
            if (mask_2_ptr[j] == 1) {
              fill_level = 2;
              break;
            }
            for (k = 0; k < n_part_1; ++k) {
              if (relation_1_ptr[k] == j) {
                if (mask_1_ptr[k] == 1) {
                  fill_level = 2;
                  break;
                }
              }
            }
          }
        }
      }
      if (fill_level != 0) {
        switch(fill_level) {
          case 1: mask_1_ptr[level_1_index] = 1; break;
          case 2: mask_2_ptr[level_2_index] = 1; break;
          case 3: mask_3_ptr[level_3_index] = 1; break;
        }
      }
    }

    // if the path from bottom to top is a one cube branch, set the top level
    // as 1 and its all children to 0
    for (i = 0; i < n_part_1; ++i) {
      // the cube index of the path from leaf to root w.r.t. the ith cube in level 1
      int level_1_index = i;
      int level_2_index = relation_1_ptr[i];
      int level_3_index = relation_2_ptr[level_2_index];
      int level_1_cube_number = 0;
      int level_2_cube_number = 0;
      // count level 2 co-parent cube number
      for (j = 0; j < n_part_2; ++j) {
        if (relation_2_ptr[j] == level_3_index) {
          level_2_cube_number++;
        }
      }
      // count level 1 co-parent cube number
      for (j = 0; j < n_part_1; ++j) {
        if (relation_1_ptr[j] == level_2_index) {
          level_1_cube_number++;
        }
      }
      // if level 3 has one children and level 2 has one children
      if (level_2_cube_number == 1 && level_1_cube_number == 1) {
        if (mask_1_ptr[level_1_index] == 1 || mask_2_ptr[level_2_index] == 1 ||
          mask_3_ptr[level_3_index] == 1) {
          mask_3_ptr[level_3_index] = 1;
          mask_1_ptr[level_1_index] = 0;
          mask_2_ptr[level_2_index] = 0;
        }
      }
      // if level 3 has one children
      if (level_2_cube_number == 1 && level_1_cube_number > 1) {
        if (mask_2_ptr[level_2_index] == 1 || mask_3_ptr[level_3_index] == 1) {
          mask_3_ptr[level_3_index] = 1;
          mask_2_ptr[level_2_index] = 0;
        }
      }
      // if level 2 has one children
      if (level_2_cube_number > 1 && level_1_cube_number == 1) {
        if (mask_1_ptr[level_1_index] == 1 || mask_2_ptr[level_2_index] == 1) {
          mask_2_ptr[level_2_index] = 1;
          mask_1_ptr[level_1_index] = 0;
        }
      }
    }
  }
}

static __global__ void lift_up_tree_mask_level_2(const int nthreads,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_1, int* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n_part_sum = n_part_1 + n_part_2 + n_part_3;
    int* mask_ptr = mask + index*n_part_sum;
    int* mask_1_ptr = mask_ptr;
    int* mask_2_ptr = mask_ptr + n_part_1;
    const int* relation_1_ptr = in_relation_1 + index*n_part_1;
    for (int i = 0; i < n_part_1; ++i) {
      if (mask_1_ptr[i] == 1) {
        mask_1_ptr[i] = 0;
        mask_2_ptr[relation_1_ptr[i]] = 1;
      }
    }
  }
}

static __global__ void lift_up_tree_mask_level_3(const int nthreads,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_2, int* mask) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n_part_sum = n_part_1 + n_part_2 + n_part_3;
    int* mask_ptr = mask + index*n_part_sum;
    int* mask_2_ptr = mask_ptr + n_part_1;
    int* mask_3_ptr = mask_ptr + n_part_1 + n_part_2;
    const int* relation_2_ptr = in_relation_2 + index*n_part_2;
    for (int i = 0; i < n_part_2; ++i) {
      if (mask_2_ptr[i] == 1) {
        mask_2_ptr[i] = 0;
        mask_3_ptr[relation_2_ptr[i]] = 1;
      }
    }
  }
}

void correct_tree_mask(OpKernelContext* context, const int batch_size,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_1, const int* in_relation_2, const int* in_mask,
    int* mask) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // copy mask
  int n_part_sum = n_part_1 + n_part_2 + n_part_3;
  cudaMemcpy(mask, in_mask, sizeof(int)*batch_size*n_part_sum,
      cudaMemcpyDeviceToDevice);

  // correct each tree in batch
  nthreads = batch_size;
  config = GetCudaLaunchConfig(nthreads, d);
  correct_tree_mask_kernal
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_part_1, n_part_2, n_part_3, in_relation_1, in_relation_2,
          mask);
}

void lift_up_tree_mask(OpKernelContext* context, const int batch_size,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_1, const int* in_relation_2, const int* tree_mask_1,
    int* tree_mask_2, int* tree_mask_3) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  int nthreads= batch_size;
  CudaLaunchConfig config = GetCudaLaunchConfig(nthreads, d);

  // copy mask to level 2
  int n_part_sum = n_part_1 + n_part_2 + n_part_3;
  cudaMemcpy(tree_mask_2, tree_mask_1, sizeof(int)*batch_size*n_part_sum,
      cudaMemcpyDeviceToDevice);

  // lift up tree to level 2
  lift_up_tree_mask_level_2
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_part_1, n_part_2, n_part_3, in_relation_1, tree_mask_2);

  // copy mask to level 3
  cudaMemcpy(tree_mask_3, tree_mask_2, sizeof(int)*batch_size*n_part_sum,
      cudaMemcpyDeviceToDevice);

  // lift up tree to level 3
  lift_up_tree_mask_level_3
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_part_1, n_part_2, n_part_3, in_relation_2, tree_mask_3);
}

}  // namespace tensorflow
