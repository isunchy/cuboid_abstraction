#define EIGEN_USE_THREADS

#include "octree.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace octree {

__global__ void max_pooling_forward_kernel(float* top_data, int top_h,
    int* bottom_mask, const float* bottom_data, int bottom_h, int nthreads) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    int h = i % top_h;
    int c = i / top_h;
    int hb = 8 * h;
    int max_idx = hb;
    const float* bottom_tmp = bottom_data + c * bottom_h;
    float max_val = bottom_tmp[hb];
#pragma unroll 7
    for (int idx = hb + 1; idx < hb + 8; ++idx) {
      float value = bottom_tmp[idx];
      if (value > max_val) {
        max_idx = idx;
        max_val = value;
      }
    }
    top_data[i] = max_val;
    bottom_mask[i] = max_idx;
  }
}

__global__ void max_pooling_backward_kernel(float* bottom_diff, int bottom_h,
    const int* bottom_mask, const float* top_diff, int top_h, int nthreads) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    int c = i / top_h;
    bottom_diff[c * bottom_h + bottom_mask[i]] = top_diff[i];
  }
}

void max_pooling_forward(OpKernelContext* ctx, float* top_data, int top_h,
    int* bottom_mask, const float* bottom_data, int bottom_h, int nthreads) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(nthreads, d);
  max_pooling_forward_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          top_data, top_h, bottom_mask, bottom_data, bottom_h, nthreads);
}

void max_pooling_backward(OpKernelContext* ctx, float* bottom_diff,
    int bottom_h, const int* bottom_mask, const float* top_diff, int top_h,
    int nthreads) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(nthreads, d);
  max_pooling_backward_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          bottom_diff, bottom_h, bottom_mask, top_diff, top_h, nthreads);
}

}  // namespace octree

}  // namespace tensorflow

