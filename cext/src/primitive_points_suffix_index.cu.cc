#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

static __global__ void suffix_index_to_points_kernel(const int nthreads,
    const int batch_size, const int n_point, const float* in_points,
    float* out_pos) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_point;
    int point_index = index % n_point;
    out_pos[0 * (batch_size * n_point) + index] = in_points[(batch_index * 3 + 0) * n_point + point_index];
    out_pos[1 * (batch_size * n_point) + index] = in_points[(batch_index * 3 + 1) * n_point + point_index];
    out_pos[2 * (batch_size * n_point) + index] = in_points[(batch_index * 3 + 2) * n_point + point_index];
    out_pos[3 * (batch_size * n_point) + index] = batch_index;
  }
}

void suffix_index_to_points(OpKernelContext* context, const int batch_size,
    const int n_point, const float* in_points, float* out_pos) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  int nthreads = batch_size * n_point;
  CudaLaunchConfig config = GetCudaLaunchConfig(nthreads, d);
  suffix_index_to_points_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, batch_size, n_point, in_points, out_pos);  
}

}  // namespace tensorflow
