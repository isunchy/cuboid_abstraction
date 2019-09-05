#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

static __device__ void matvec_kernel(const float* m, float* x, float* y,
    float* z) {
  float tx = m[0] * (*x) + m[1] * (*y) + m[2] * (*z);
  float ty = m[3] * (*x) + m[4] * (*y) + m[5] * (*z);
  float tz = m[6] * (*x) + m[7] * (*y) + m[8] * (*z);
  *x = tx; *y = ty; *z = tz;
}

static __device__ float diag(const float a, const float b) {
  return 1 - 2 * a * a - 2 * b * b;
}

static __device__ float tr_add(const float a, const float b, const float c,
    const float d) {
  return 2 * a * b + 2 * c * d;
}

static __device__ float tr_sub(const float a, const float b, const float c,
    const float d) {
  return 2 * a * b - 2 * c * d;
}

static __device__ void normalize(float* w, float* x, float* y, float* z) {
  float norm = sqrt((*w)*(*w) + (*x)*(*x) + (*y)*(*y) + (*z)*(*z));
  *w /= norm;  *x /= norm;  *y /= norm;  *z /= norm;
}

static __device__ void as_rotation_matrix(float w, float x, float y, float z,
    float* m) {
  normalize(&w, &x, &y, &z);
  m[0] = diag(y, z);  m[1] = tr_sub(x, y, z, w);  m[2] = tr_add(x, z, y, w);
  m[3] = tr_add(x, y, z, w);  m[4] = diag(x, z);  m[5] = tr_sub(y, z, x, w);
  m[6] = tr_sub(x, z, y, w);  m[7] = tr_add(y, z, x, w);  m[8] = diag(x, y);
}

static __device__ void grad_rotation_matrix_to_quaternion(
    const float* grad_rotation_matrix, const float qw, const float qx,
    const float qy, const float qz, float* gqw, float* gqx, float* gqy,
    float* gqz) {
  const float* m = grad_rotation_matrix;
  float w = qw, x = qx, y = qy, z = qz;
  float w2 = w*w, x2 = x*x, y2 = y*y, z2 = z*z;
  float wx = w*x, wy = w*y, wz = w*z, xy = x*y, xz = x*z, yz = y*z;
  float s = 1.0 / (w2 + x2 + y2 + z2);  // devide -> multiple
  float s2 = s*s;
  *gqw =
      m[0] * (4 * w*(y2 + z2)*s2) +
      m[1] * (4 * w*(wz - xy)*s2 - 2 * z*s) +
      m[2] * (2 * y*s - 4 * w*(wy + xz)*s2) +
      m[3] * (2 * z*s - 4 * w*(wz + xy)*s2) +
      m[4] * (4 * w*(x2 + z2)*s2) +
      m[5] * (4 * w*(wx - yz)*s2 - 2 * x*s) +
      m[6] * (4 * w*(wy - xz)*s2 - 2 * y*s) +
      m[7] * (2 * x*s - 4 * w*(wx + yz)*s2) +
      m[8] * (4 * w*(x2 + y2)*s2);
  *gqx =
      m[0] * (4 * x*(y2 + z2)*s2) +
      m[1] * (4 * x*(wz - xy)*s2 + 2 * y*s) +
      m[2] * (2 * z*s - 4 * x*(wy + xz)*s2) +
      m[3] * (2 * y*s - 4 * x*(wz + xy)*s2) +
      m[4] * (4 * x*(x2 + z2)*s2 - 4 * x*s) +
      m[5] * (4 * x*(wx - yz)*s2 - 2 * w*s) +
      m[6] * (4 * x*(wy - xz)*s2 + 2 * z*s) +
      m[7] * (2 * w*s - 4 * x*(wx + yz)*s2) +
      m[8] * (4 * x*(x2 + y2)*s2 - 4 * x*s);
  *gqy =
      m[0] * (4 * y*(y2 + z2)*s2 - 4 * y*s) +
      m[1] * (4 * y*(wz - xy)*s2 + 2 * x*s) +
      m[2] * (2 * w*s - 4 * y*(wy + xz)*s2) +
      m[3] * (2 * x*s - 4 * y*(wz + xy)*s2) +
      m[4] * (4 * y*(x2 + z2)*s2) +
      m[5] * (4 * y*(wx - yz)*s2 + 2 * z*s) +
      m[6] * (4 * y*(wy - xz)*s2 - 2 * w*s) +
      m[7] * (2 * z*s - 4 * y*(wx + yz)*s2) +
      m[8] * (4 * y*(x2 + y2)*s2 - 4 * y*s);
  *gqz =
      m[0] * (4 * z*(y2 + z2)*s2 - 4 * z*s) +
      m[1] * (4 * z*(wz - xy)*s2 - 2 * w*s) +
      m[2] * (2 * x*s - 4 * z*(wy + xz)*s2) +
      m[3] * (2 * w*s - 4 * z*(wz + xy)*s2) +
      m[4] * (4 * z*(x2 + z2)*s2 - 4 * z*s) +
      m[5] * (4 * z*(wx - yz)*s2 + 2 * y*s) +
      m[6] * (4 * z*(wy - xz)*s2 + 2 * x*s) +
      m[7] * (2 * y*s - 4 * z*(wx + yz)*s2) +
      m[8] * (4 * z*(x2 + y2)*s2);
}

static __global__ void get_aligning_loss(const int nthreads, const int n_cube,
    const int batch_size, const float* in_dir, const float* in_q,
    float* loss) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_cube;
    int cube_index = index % n_cube;
    float px = in_dir[0], py = in_dir[1], pz = in_dir[2];
    float raw_px = px, raw_py = py, raw_pz = pz;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    float distance = 1 - (px * raw_px + py * raw_py + pz * raw_pz);
    CudaAtomicAdd(loss, distance / (batch_size * n_cube));
  }
}

static __global__ void fill_grad_wrt_q(const int nthreads, const int n_cube,
    const int batch_size, const float* loss, const float* in_q,
    const float* in_dir, float* grad_q) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_cube;
    int cube_index = index % n_cube;
    float px = in_dir[0], py = in_dir[1], pz = in_dir[2];
    float raw_px = px, raw_py = py, raw_pz = pz;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float tmp_qw = qw, tmp_qx = qx, tmp_qy = qy, tmp_qz = qz;
    float rotation_matrix[9];
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);

    float* gq = grad_q + (batch_index * n_cube + cube_index) * 4;
    float grad_distance = (*loss) / (batch_size * n_cube);
    float gdx = -grad_distance * raw_px;
    float gdy = -grad_distance * raw_py;
    float gdz = -grad_distance * raw_pz;
    // gradient w.r.t. q
    {
      float grad_rotation_matrix[9];
      grad_rotation_matrix[0] = gdx * raw_px;
      grad_rotation_matrix[1] = gdx * raw_py;
      grad_rotation_matrix[2] = gdx * raw_pz;
      grad_rotation_matrix[3] = gdy * raw_px;
      grad_rotation_matrix[4] = gdy * raw_py;
      grad_rotation_matrix[5] = gdy * raw_pz;
      grad_rotation_matrix[6] = gdz * raw_px;
      grad_rotation_matrix[7] = gdz * raw_py;
      grad_rotation_matrix[8] = gdz * raw_pz;
      float gqw, gqx, gqy, gqz;
      grad_rotation_matrix_to_quaternion(grad_rotation_matrix, tmp_qw, tmp_qx,
          tmp_qy, tmp_qz, &gqw, &gqx, &gqy, &gqz);
      CudaAtomicAdd(gq + 0, gqw);
      CudaAtomicAdd(gq + 1, gqx);
      CudaAtomicAdd(gq + 2, gqy);
      CudaAtomicAdd(gq + 3, gqz);
    }
  }
}

void compute_aligning_loss_v2(OpKernelContext* context, const int n_cube,
    const int batch_size, const float* in_q, const float* in_dir,
    float* loss_ptr) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // check in_dir is normalized
  float norm = thrust::reduce(thrust::device, in_dir, in_dir + 3);
  CHECK(norm - 1.0f < 1e-6);

  // get aligning loss
  primitive::gpu_set_zero(context, loss_ptr, 1);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_aligning_loss
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, in_dir, in_q, loss_ptr);  
}

void compute_aligning_loss_grad_v2(OpKernelContext* context, const int n_cube,
    const int batch_size, const float* loss, const float* in_q,
    const float* in_dir, float* grad_q) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // init zero gradient
  primitive::gpu_set_zero(context, grad_q, batch_size * n_cube * 4);

  // gradient w.r.t. q
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_q
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, loss, in_q, in_dir, grad_q);  
}

}  // namespace tensorflow
