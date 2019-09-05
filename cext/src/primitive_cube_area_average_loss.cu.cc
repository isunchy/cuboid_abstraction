#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

static __device__ float smooth_l1(float x) {
  if (abs(x) < 1) {
    return 0.5f*x*x;
  }
  else {
    return abs(x) - 0.5f;
  }
}

static __device__ float smooth_l1_grad(float x) {
  if (x <= -1.0f) {
    return -1.0f;
  }
  else if (x <= 1.0f) {
    return x;
  }
  else {
    return 1.0f;
  }
}

static __global__ void get_cube_surface_mean_area(const int nthreads,
    const int n_cube, const int batch_size, const float* in_z,
    float* cube_surface_mean_area) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_cube;
    float x = in_z[index * 3 + 0] * 2;
    float y = in_z[index * 3 + 1] * 2;
    float z = in_z[index * 3 + 2] * 2;
    float area_xy = x * y;
    float area_xz = x * z;
    float area_yz = y * z;
    CudaAtomicAdd(cube_surface_mean_area + batch_index, 
        (area_xy + area_xz + area_yz) / (3 * n_cube));
  }
}

static __global__ void get_cube_surface_mean_area_loss(const int nthreads,
    const int n_cube, const int batch_size, const float* in_z,
    const float* cube_surface_mean_area, float* area_loss) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_cube;
    float x = in_z[index * 3 + 0] * 2;
    float y = in_z[index * 3 + 1] * 2;
    float z = in_z[index * 3 + 2] * 2;
    float area_xy = x * y;
    float area_xz = x * z;
    float area_yz = y * z;
    float d_xy = smooth_l1(area_xy - cube_surface_mean_area[batch_index]);
    float d_xz = smooth_l1(area_xz - cube_surface_mean_area[batch_index]);
    float d_yz = smooth_l1(area_yz - cube_surface_mean_area[batch_index]);
    CudaAtomicAdd(area_loss, (d_xy + d_xz + d_yz) /
        (3 * batch_size * n_cube));
  }
}

static __global__ void fill_grad_wrt_z(const int nthreads, const int n_cube,
    const int batch_size, const float* in_z, const float* loss,
    const float* cube_surface_mean_area, float* grad_z) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_cube;
    float x = in_z[index * 3 + 0] * 2;
    float y = in_z[index * 3 + 1] * 2;
    float z = in_z[index * 3 + 2] * 2;
    float area_xy = x * y;
    float area_xz = x * z;
    float area_yz = y * z;
    float d_xy = area_xy - cube_surface_mean_area[batch_index];
    float d_xz = area_xz - cube_surface_mean_area[batch_index];
    float d_yz = area_yz - cube_surface_mean_area[batch_index];

    float grad_d = *loss / (3 * batch_size * n_cube);
    float grad_d_xy = grad_d * smooth_l1_grad(d_xy);
    float grad_d_xz = grad_d * smooth_l1_grad(d_xz);
    float grad_d_yz = grad_d * smooth_l1_grad(d_yz);
    float* gz = grad_z + index * 3;
    CudaAtomicAdd(gz + 0, (grad_d_xy * y + grad_d_xz * z) * 2);
    CudaAtomicAdd(gz + 1, (grad_d_xy * x + grad_d_yz * z) * 2);
    CudaAtomicAdd(gz + 2, (grad_d_xz * x + grad_d_yz * y) * 2);
    grad_d = -(grad_d_xy + grad_d_xz + grad_d_yz) / (3 * n_cube);
    for (int i = 0; i < n_cube; ++i) {
      float sx = in_z[(batch_index * n_cube + i) * 3 + 0] * 2;
      float sy = in_z[(batch_index * n_cube + i) * 3 + 1] * 2;
      float sz = in_z[(batch_index * n_cube + i) * 3 + 2] * 2;
      float* gsz = grad_z + (batch_index * n_cube + i) * 3;
      CudaAtomicAdd(gsz + 0, grad_d * (sy + sz) * 2);
      CudaAtomicAdd(gsz + 1, grad_d * (sx + sz) * 2);
      CudaAtomicAdd(gsz + 2, grad_d * (sx + sy) * 2);
    }
  }  
}

void compute_cube_area_average_loss(OpKernelContext* context,
    const int n_cube, const int batch_size, const float* in_z,
    float* loss_ptr) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // get cube surface mean area, regrad each surface as an instance, which
  // means each cube have three surface area
  Tensor cube_surface_mean_area;
  const TensorShape cube_surface_mean_area_shape({batch_size});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              cube_surface_mean_area_shape,
                              &cube_surface_mean_area));
  auto cube_surface_mean_area_ptr = cube_surface_mean_area.flat<float>().data();
  primitive::gpu_set_zero(context, cube_surface_mean_area_ptr, batch_size);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_cube_surface_mean_area
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, in_z, cube_surface_mean_area_ptr);


  // get cube volume loss
  primitive::gpu_set_zero(context, loss_ptr, 1);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_cube_surface_mean_area_loss
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, in_z, cube_surface_mean_area_ptr,
          loss_ptr);  
}

void compute_cube_area_average_loss_grad(OpKernelContext* context,
    const int n_cube, const int batch_size, const float* loss,
    const float* in_z, float* grad_z) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  /// -- prepare forward medial data for gradient computation --
  // get cube surface mean area, regrad each surface as an instance, which
  // means each cube have three surface area
  Tensor cube_surface_mean_area;
  const TensorShape cube_surface_mean_area_shape({batch_size});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              cube_surface_mean_area_shape,
                              &cube_surface_mean_area));
  auto cube_surface_mean_area_ptr = cube_surface_mean_area.flat<float>().data();
  primitive::gpu_set_zero(context, cube_surface_mean_area_ptr, batch_size);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_cube_surface_mean_area
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, in_z, cube_surface_mean_area_ptr);
  /// ----------------------------------------------------------

  // init zero gradient
  primitive::gpu_set_zero(context, grad_z, batch_size * n_cube * 3);

  // gradient w.r.t. z
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_z
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, in_z, loss, cube_surface_mean_area_ptr,
          grad_z);
}

}  // namespace tensorflow
