#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

static __global__ void get_cube_volume(const int nthreads,
    const int batch_size, const float* in_z, float* volume_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float x = in_z[index * 3 + 0] * 2;
    float y = in_z[index * 3 + 1] * 2;
    float z = in_z[index * 3 + 2] * 2;
    float volume = x * y * z;
    CudaAtomicAdd(volume_ptr, volume / batch_size);
  }
}

void compute_cube_volume(OpKernelContext* context, const int n_cube,
    const int batch_size, const float* in_z, float* out_volume) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // get cube volume
  primitive::gpu_set_zero(context, out_volume, 1);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_cube_volume
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, batch_size, in_z, out_volume);  
}

}  // namespace tensorflow
