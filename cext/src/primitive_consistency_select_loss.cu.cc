#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

struct my_min_func {
  template <typename T1, typename T2>
  __host__ __device__
  T1 operator()(const T1 t1, const T2 t2) {
    T1 res;
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) {
      thrust::get<0>(res) = thrust::get<0>(t1);
      thrust::get<1>(res) = thrust::get<1>(t1);
    }
    else {
      thrust::get<0>(res) = thrust::get<0>(t2);
      thrust::get<1>(res) = thrust::get<1>(t2);      
    }
    return res;
  }
};

static __device__ void matvec_kernel(const float* m, float* x, float* y,
    float* z) {
  float tx = m[0] * (*x) + m[1] * (*y) + m[2] * (*z);
  float ty = m[3] * (*x) + m[4] * (*y) + m[5] * (*z);
  float tz = m[6] * (*x) + m[7] * (*y) + m[8] * (*z);
  *x = tx; *y = ty; *z = tz;
}

static __device__ void t_matvec_kernel(const float* m, float* x, float* y,
    float* z) {
  float tx = m[0] * (*x) + m[3] * (*y) + m[6] * (*z);
  float ty = m[1] * (*x) + m[4] * (*y) + m[7] * (*z);
  float tz = m[2] * (*x) + m[5] * (*y) + m[8] * (*z);
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

static __global__ void fill_sample_point_object_point_distance(
    const int nthreads, const int n_cube, const int n_sample_point,
    const int n_point, const int batch_size, const float* in_z,
    const float* in_q, const float* in_t, const float* in_pos,
    const float* in_sample_points, float* sample_point_object_point_distance,
    int* sample_point_object_point_key) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int cube_index = index / (n_sample_point * n_point);
    int sample_point_index = (index / n_point) % n_sample_point;
    int point_index = index % n_point;
    float px = in_pos[0 * n_point + point_index];
    float py = in_pos[1 * n_point + point_index];
    float pz = in_pos[2 * n_point + point_index];
    int batch_index = static_cast<int>(in_pos[3 * n_point + point_index]);
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
    float spx = in_sample_points[0 * n_sample_point + sample_point_index];
    float spy = in_sample_points[1 * n_sample_point + sample_point_index];
    float spz = in_sample_points[2 * n_sample_point + sample_point_index];
    spx *= z[0];  spy *= z[1];  spz *= z[2];
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &spx, &spy, &spz);
    spx += t[0];  spy += t[1];  spz += t[2];
    float dx = spx - px;
    float dy = spy - py;
    float dz = spz - pz;
    sample_point_object_point_distance[(cube_index * n_sample_point +
        sample_point_index) * n_point + point_index] = dx * dx + dy * dy +
        dz * dz;
    sample_point_object_point_key[(cube_index * n_sample_point +
        sample_point_index) * n_point + point_index] =
        (cube_index * n_sample_point + sample_point_index) * batch_size +
        batch_index;
  }
}

static __global__ void get_batch_valid_cube_number(const int nthreads,
    const int n_cube, const int* in_mask, int* batch_valid_cube_number) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_cube;
    int cube_index = index % n_cube;
    CudaAtomicAdd(batch_valid_cube_number + batch_index,
        in_mask[batch_index * n_cube + cube_index]);
  }
}

static __global__ void get_consistency_loss(const int nthreads,
    const int n_cube, const int n_sample_point, const int batch_size,
    const int* in_mask, const int* batch_valid_cube_number,
    const float* sample_point_min_distance_ptr, float* loss_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int cube_index = index / (n_sample_point * batch_size);
    int batch_index = index % batch_size;
    if (in_mask[batch_index * n_cube + cube_index]) {
      CudaAtomicAdd(loss_ptr,
          sample_point_min_distance_ptr[index] / 
          (batch_valid_cube_number[batch_index] * n_sample_point * batch_size));
    }
  }
}

static __global__ void fill_grad_sample_point_object_point_distance(
    const int nthreads, const int n_cube, const int n_sample_point,
    const int batch_size, const float* loss,
    const int* in_mask, const int* batch_valid_cube_number,
    const int* sample_point_min_distance_index,
    float* grad_sample_point_object_point_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int cube_index = index / (n_sample_point * batch_size);
    int batch_index = index % batch_size;
    if (in_mask[batch_index * n_cube + cube_index]) {
      grad_sample_point_object_point_distance[
          sample_point_min_distance_index[index]] = (*loss) /
          (batch_valid_cube_number[batch_index] * n_sample_point * batch_size);
    }
  }
}

static __global__ void fill_grad_wrt_zqt(const int nthreads, const int n_cube,
    const int n_sample_point, const int n_point, const float* in_z,
    const float* in_q, const float* in_t, const int* in_mask,
    const float* in_pos, const float* in_sample_points,
    const float* grad_sample_point_object_point_distance, float* grad_z,
    float* grad_q, float* grad_t) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int cube_index = index / (n_sample_point * n_point);
    int sample_point_index = (index / n_point) % n_sample_point;
    int point_index = index % n_point;
    int batch_index = static_cast<int>(in_pos[3 * n_point + point_index]);
    int cube_mask = in_mask[batch_index * n_cube + cube_index];
    if (cube_mask == 1) {
      float px = in_pos[0 * n_point + point_index];
      float py = in_pos[1 * n_point + point_index];
      float pz = in_pos[2 * n_point + point_index];
      const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
      const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
      const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
      float spx = in_sample_points[0 * n_sample_point + sample_point_index];
      float spy = in_sample_points[1 * n_sample_point + sample_point_index];
      float spz = in_sample_points[2 * n_sample_point + sample_point_index];
      float raw_spx = spx, raw_spy = spy, raw_spz = spz;
      spx *= z[0];  spy *= z[1];  spz *= z[2];
      float tmp_spx = spx, tmp_spy = spy, tmp_spz = spz;
      float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
      float rotation_matrix[9];
      float tmp_qw = qw, tmp_qx = qx, tmp_qy = qy, tmp_qz = qz;
      as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
      matvec_kernel(rotation_matrix, &spx, &spy, &spz);
      spx += t[0];  spy += t[1];  spz += t[2];
      float dx = spx - px;
      float dy = spy - py;
      float dz = spz - pz;

      float* gz = grad_z + (batch_index * n_cube + cube_index) * 3;
      float* gq = grad_q + (batch_index * n_cube + cube_index) * 4;
      float* gt = grad_t + (batch_index * n_cube + cube_index) * 3;
      float grad_distance =
          grad_sample_point_object_point_distance[(cube_index * n_sample_point +
              sample_point_index) * n_point + point_index];
      float gdx = grad_distance * 2 * dx;
      float gdy = grad_distance * 2 * dy;
      float gdz = grad_distance * 2 * dz;
      // gradient w.r.t. t
      {
        CudaAtomicAdd(gt + 0, gdx);
        CudaAtomicAdd(gt + 1, gdy);
        CudaAtomicAdd(gt + 2, gdz);
      }
      // gradient w.r.t. q
      {
        float grad_rotation_matrix[9];
        grad_rotation_matrix[0] = gdx * tmp_spx;
        grad_rotation_matrix[1] = gdx * tmp_spy;
        grad_rotation_matrix[2] = gdx * tmp_spz;
        grad_rotation_matrix[3] = gdy * tmp_spx;
        grad_rotation_matrix[4] = gdy * tmp_spy;
        grad_rotation_matrix[5] = gdy * tmp_spz;
        grad_rotation_matrix[6] = gdz * tmp_spx;
        grad_rotation_matrix[7] = gdz * tmp_spy;
        grad_rotation_matrix[8] = gdz * tmp_spz;
        float gqw, gqx, gqy, gqz;
        grad_rotation_matrix_to_quaternion(grad_rotation_matrix, tmp_qw, tmp_qx,
            tmp_qy, tmp_qz, &gqw, &gqx, &gqy, &gqz);
        CudaAtomicAdd(gq + 0, gqw);
        CudaAtomicAdd(gq + 1, gqx);
        CudaAtomicAdd(gq + 2, gqy);
        CudaAtomicAdd(gq + 3, gqz);      
      }
      t_matvec_kernel(rotation_matrix, &gdx, &gdy, &gdz);
      // gradient w.r.t. z
      {
        CudaAtomicAdd(gz + 0, gdx * raw_spx);
        CudaAtomicAdd(gz + 1, gdy * raw_spy);
        CudaAtomicAdd(gz + 2, gdz * raw_spz);
      }
    }
  }
}

static std::vector<float> cube_surface_points_host_8 {
 -1.0, -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0,
 -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0,
 -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0,
};
static std::vector<float> cube_surface_points_host_26 {
  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
  -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,
  -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0,
};
static std::vector<float> cube_surface_points_host_96 {
  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.75, -0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, -0.75, -0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, -0.75, -0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, -0.75, -0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75,
  -0.75, -0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, -0.75, -0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75,
  -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
};

// static std::vector<float> cube_surface_points_host_150 {
//   1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8, -0.8, -0.8, -0.8, -0.8, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8,
//   -0.8, -0.8, -0.8, -0.8, -0.8, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.4, -0.4, -0.4, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8,
//   -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, -0.8, -0.4, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
// };

void compute_consistency_select_loss(OpKernelContext* context, const int n_cube,
    const int n_point, const int batch_size, const float num_sample,
    const float scale, const float* in_z, const float* in_q, const float* in_t,
    const int* in_mask, const float* in_pos, float* loss_ptr) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // sample points on cube surface
  Tensor cube_surface_points;
  int n_sample_point;
  if (num_sample == 8){
    n_sample_point = cube_surface_points_host_8.size() / 3;
  }
  else if (num_sample == 26){
    n_sample_point = cube_surface_points_host_26.size() / 3;
  }
  else if (num_sample == 96) {
    n_sample_point = cube_surface_points_host_96.size() / 3;
  }
  else {
    LOG(ERROR) << "num_sample error in consistency loss layer";
  }
  const TensorShape cube_surface_points_shape({n_sample_point, 3});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              cube_surface_points_shape,
                              &cube_surface_points));
  auto cube_surface_points_ptr = cube_surface_points.flat<float>().data();
  if (num_sample == 8){
    cudaMemcpy(cube_surface_points_ptr, cube_surface_points_host_8.data(),
        sizeof(float) * n_sample_point * 3, cudaMemcpyHostToDevice);
  }
  else if (num_sample == 26){
    cudaMemcpy(cube_surface_points_ptr, cube_surface_points_host_26.data(),
        sizeof(float) * n_sample_point * 3, cudaMemcpyHostToDevice);
  }
  else if (num_sample == 96) {
    cudaMemcpy(cube_surface_points_ptr, cube_surface_points_host_96.data(),
        sizeof(float) * n_sample_point * 3, cudaMemcpyHostToDevice);
  }
  else {
    LOG(ERROR) << "num_sample error in consistency loss layer";
  }
  thrust::transform(thrust::device, cube_surface_points_ptr,
      cube_surface_points_ptr + n_sample_point * 3,
      thrust::make_constant_iterator(scale), cube_surface_points_ptr,
      thrust::multiplies<float>());

  // fill sampled point to object point distance matrix
  // [n_cube * n_sample_point, n_point]
  Tensor sample_point_object_point_distance;
  Tensor sample_point_object_point_index;
  Tensor sample_point_object_point_key;
  const TensorShape sample_point_object_point_distance_shape({
      n_cube * n_sample_point, n_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              sample_point_object_point_distance_shape,
                              &sample_point_object_point_distance));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_object_point_distance_shape,
                              &sample_point_object_point_index));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_object_point_distance_shape,
                              &sample_point_object_point_key));
  auto sample_point_object_point_distance_ptr =
      sample_point_object_point_distance.flat<float>().data();
  auto sample_point_object_point_index_ptr =
      sample_point_object_point_index.flat<int>().data();
  auto sample_point_object_point_key_ptr =
      sample_point_object_point_key.flat<int>().data();
  nthreads = n_cube * n_sample_point * n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_sample_point_object_point_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, n_point, batch_size, in_z, in_q,
          in_t, in_pos, cube_surface_points_ptr,
          sample_point_object_point_distance_ptr,
          sample_point_object_point_key_ptr);

  // get min distance and corresponding point index
  Tensor sample_point_min_distance;
  Tensor sample_point_min_distance_index;
  Tensor sample_point_min_distance_key;
  const TensorShape sample_point_min_distance_shape({
      n_cube * n_sample_point * batch_size});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              sample_point_min_distance_shape,
                              &sample_point_min_distance));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_min_distance_shape,
                              &sample_point_min_distance_index));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_min_distance_shape,
                              &sample_point_min_distance_key));
  auto sample_point_min_distance_ptr =
      sample_point_min_distance.flat<float>().data();
  auto sample_point_min_distance_index_ptr =
      sample_point_min_distance_index.flat<int>().data();
  auto sample_point_min_distance_key_ptr =
      sample_point_min_distance_key.flat<int>().data();
  thrust::sequence(thrust::device, sample_point_object_point_index_ptr,
      sample_point_object_point_index_ptr + n_cube * n_sample_point * n_point);
  auto new_end = thrust::reduce_by_key(thrust::device,
      sample_point_object_point_key_ptr,
      sample_point_object_point_key_ptr + n_cube * n_sample_point * n_point,
      thrust::make_zip_iterator(thrust::make_tuple(
          sample_point_object_point_distance_ptr,
          sample_point_object_point_index_ptr)),
      sample_point_min_distance_key_ptr,
      thrust::make_zip_iterator(thrust::make_tuple(
          sample_point_min_distance_ptr,
          sample_point_min_distance_index_ptr)),
      thrust::equal_to<int>(),
      my_min_func());
  CHECK_EQ(new_end.first - sample_point_min_distance_key_ptr,
      n_cube * n_sample_point * batch_size);

  // get batch valid cube number
  Tensor batch_valid_cube_number;
  const TensorShape batch_valid_cube_number_shape({batch_size});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              batch_valid_cube_number_shape,
                              &batch_valid_cube_number));
  auto batch_valid_cube_number_ptr = batch_valid_cube_number.flat<int>().data();
  primitive::gpu_set_zero(context, batch_valid_cube_number_ptr, batch_size);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_batch_valid_cube_number
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, in_mask, batch_valid_cube_number_ptr);

  // get consistency loss
  nthreads = n_cube * n_sample_point * batch_size;
  config = GetCudaLaunchConfig(nthreads, d);
  primitive::gpu_set_zero(context, loss_ptr, 1);
  get_consistency_loss
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, batch_size, in_mask,
          batch_valid_cube_number_ptr, sample_point_min_distance_ptr, loss_ptr);
}

void compute_consistency_select_loss_grad(OpKernelContext* context,
    const int n_cube, const int n_point, const int batch_size,
    const float num_sample, const float scale, const float* loss,
    const float* in_z, const float* in_q, const float* in_t,
    const int* in_mask, const float* in_pos, float* grad_z, float* grad_q,
    float* grad_t) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  /// -- prepare forward medial data for gradient computation --
  // sample points on cube surface
  Tensor cube_surface_points;
  int n_sample_point;
  if (num_sample == 8){
    n_sample_point = cube_surface_points_host_8.size() / 3;
  }
  else if (num_sample == 26){
    n_sample_point = cube_surface_points_host_26.size() / 3;
  }
  else if (num_sample == 96) {
    n_sample_point = cube_surface_points_host_96.size() / 3;
  }
  else {
    LOG(ERROR) << "num_sample error in consistency loss layer";
  }
  const TensorShape cube_surface_points_shape({n_sample_point, 3});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              cube_surface_points_shape,
                              &cube_surface_points));
  auto cube_surface_points_ptr = cube_surface_points.flat<float>().data();
  if (num_sample == 8){
    cudaMemcpy(cube_surface_points_ptr, cube_surface_points_host_8.data(),
        sizeof(float) * n_sample_point * 3, cudaMemcpyHostToDevice);
  }
  else if (num_sample == 26){
    cudaMemcpy(cube_surface_points_ptr, cube_surface_points_host_26.data(),
        sizeof(float) * n_sample_point * 3, cudaMemcpyHostToDevice);
  }
  else if (num_sample == 96) {
    cudaMemcpy(cube_surface_points_ptr, cube_surface_points_host_96.data(),
        sizeof(float) * n_sample_point * 3, cudaMemcpyHostToDevice);
  }
  else {
    LOG(ERROR) << "num_sample error in consistency loss layer";
  }
  thrust::transform(thrust::device, cube_surface_points_ptr,
      cube_surface_points_ptr + n_sample_point * 3,
      thrust::make_constant_iterator(scale), cube_surface_points_ptr,
      thrust::multiplies<float>());
  // fill sampled point to object point distance matrix
  // [n_cube * n_sample_point, n_point]
  Tensor sample_point_object_point_distance;
  Tensor sample_point_object_point_index;
  Tensor sample_point_object_point_key;
  const TensorShape sample_point_object_point_distance_shape({
      n_cube * n_sample_point, n_point });
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              sample_point_object_point_distance_shape,
                              &sample_point_object_point_distance));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_object_point_distance_shape,
                              &sample_point_object_point_index));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_object_point_distance_shape,
                              &sample_point_object_point_key));
  auto sample_point_object_point_distance_ptr =
      sample_point_object_point_distance.flat<float>().data();
  auto sample_point_object_point_index_ptr =
      sample_point_object_point_index.flat<int>().data();
  auto sample_point_object_point_key_ptr =
      sample_point_object_point_key.flat<int>().data();
  nthreads = n_cube * n_sample_point * n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_sample_point_object_point_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, n_point, batch_size, in_z, in_q,
          in_t, in_pos, cube_surface_points_ptr,
          sample_point_object_point_distance_ptr,
          sample_point_object_point_key_ptr);

  // get min distance and corresponding point index
  Tensor sample_point_min_distance;
  Tensor sample_point_min_distance_index;
  Tensor sample_point_min_distance_key;
  const TensorShape sample_point_min_distance_shape({
      n_cube * n_sample_point * batch_size});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              sample_point_min_distance_shape,
                              &sample_point_min_distance));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_min_distance_shape,
                              &sample_point_min_distance_index));
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              sample_point_min_distance_shape,
                              &sample_point_min_distance_key));
  auto sample_point_min_distance_ptr =
      sample_point_min_distance.flat<float>().data();
  auto sample_point_min_distance_index_ptr =
      sample_point_min_distance_index.flat<int>().data();
  auto sample_point_min_distance_key_ptr =
      sample_point_min_distance_key.flat<int>().data();
  thrust::sequence(thrust::device, sample_point_object_point_index_ptr,
      sample_point_object_point_index_ptr + n_cube * n_sample_point * n_point);
  auto new_end = thrust::reduce_by_key(thrust::device,
      sample_point_object_point_key_ptr,
      sample_point_object_point_key_ptr + n_cube * n_sample_point * n_point,
      thrust::make_zip_iterator(thrust::make_tuple(
          sample_point_object_point_distance_ptr,
          sample_point_object_point_index_ptr)),
      sample_point_min_distance_key_ptr,
      thrust::make_zip_iterator(thrust::make_tuple(
          sample_point_min_distance_ptr,
          sample_point_min_distance_index_ptr)),
      thrust::equal_to<int>(),
      my_min_func());
  CHECK_EQ(new_end.first - sample_point_min_distance_key_ptr,
      n_cube * n_sample_point * batch_size);

  // get batch valid cube number
  Tensor batch_valid_cube_number;
  const TensorShape batch_valid_cube_number_shape({batch_size});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              batch_valid_cube_number_shape,
                              &batch_valid_cube_number));
  auto batch_valid_cube_number_ptr = batch_valid_cube_number.flat<int>().data();
  primitive::gpu_set_zero(context, batch_valid_cube_number_ptr, batch_size);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_batch_valid_cube_number
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, in_mask, batch_valid_cube_number_ptr);
  /// ----------------------------------------------------------

  // splash gradient to sampled point to object point distance
  Tensor grad_sample_point_object_point_distance;
  const TensorShape gspopd_shape({n_cube * n_sample_point, n_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, gspopd_shape,
                              &grad_sample_point_object_point_distance));
  auto gspopd_ptr = grad_sample_point_object_point_distance.flat<float>().data();
  primitive::gpu_set_zero(context, gspopd_ptr,
      n_cube * n_sample_point * n_point);
  nthreads = n_cube * n_sample_point * batch_size;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_sample_point_object_point_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, batch_size, loss,
          in_mask, batch_valid_cube_number_ptr,
          sample_point_min_distance_index_ptr, gspopd_ptr);

  // init zero gradient
  primitive::gpu_set_zero(context, grad_z, batch_size * n_cube * 3);
  primitive::gpu_set_zero(context, grad_q, batch_size * n_cube * 4);
  primitive::gpu_set_zero(context, grad_t, batch_size * n_cube * 3);

  // gradient w.r.t. (z, q, t)
  nthreads = n_cube * n_sample_point * n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_zqt
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, n_point, in_z, in_q, in_t, in_mask,
          in_pos, cube_surface_points_ptr, gspopd_ptr, grad_z, grad_q, grad_t);
}

}  // namespace tensorflow