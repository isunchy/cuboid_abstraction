#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

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

static __device__ void conjugate(float* w, float* x, float* y, float* z) {
  (*x) = -(*x);  (*y) = -(*y);  (*z) = -(*z);
}

static __global__ void fill_point_cube_distance(const int nthreads,
    const int n_cube, const int n_point, const float* in_z, const float* in_q,
    const float* in_t, const float* in_pos, float* point_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int point_index = index / n_cube;
    int cube_index = index % n_cube;
    float px = in_pos[0 * n_point + point_index];
    float py = in_pos[1 * n_point + point_index];
    float pz = in_pos[2 * n_point + point_index];
    int batch_index = static_cast<int>(in_pos[3 * n_point + point_index]);
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
    px -= t[0];  py -= t[1];  pz -= t[2];
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    conjugate(&qw, &qx, &qy, &qz);
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    float dx = MAX(abs(px) - z[0], 0);
    float dy = MAX(abs(py) - z[1], 0);
    float dz = MAX(abs(pz) - z[2], 0);
    point_cube_distance[point_index * n_cube + cube_index] = dx * dx +
        dy * dy + dz * dz;
  }
}

static __global__ void get_min_distance_cube_index(const int nthreads,
    const int n_cube, const int n_point, const float* point_cube_distance,
    const float* in_pos, int* min_distance_cube_index) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = static_cast<int>(in_pos[3 * n_point + index]);
    const float* distance = point_cube_distance + index * n_cube;
    float min_val = distance[0];
    int min_idx = 0;
    for (int i = 1; i < n_cube; ++i) {
      float d = distance[i];
      if (d < min_val) {
        min_idx = i;
        min_val = d;
      }
    }
    min_distance_cube_index[index] = batch_index * n_cube + min_idx;
  }
}

void group_points(OpKernelContext* context, const int n_point, const int n_cube,
    const float* in_z, const float* in_q, const float* in_t,
    const float* in_pos, int* index) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // fill point to cube distance matrix, [n_point, n_cube]
  Tensor point_cube_distance;
  const TensorShape point_cube_shape({n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, point_cube_shape,
                              &point_cube_distance));
  auto point_cube_distance_ptr = point_cube_distance.flat<float>().data();
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, in_pos,
          point_cube_distance_ptr);

  // get min distance cube index
  nthreads = n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  get_min_distance_cube_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, point_cube_distance_ptr, in_pos, index);
}

}  // namespace tensorflow
