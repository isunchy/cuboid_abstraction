#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

#define SIGN(a) (((a)>=0)?(1):(-1))

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

static __device__ void conjugate(float* w, float* x, float* y, float* z) {
  (*x) = -(*x);  (*y) = -(*y);  (*z) = -(*z);
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

static __global__ void fill_all_sample_points(const int nthreads,
    const int n_cube, const int n_sample_point, const float* in_sample_points,
    const float* in_z, const float* in_q, const float* in_t,
    float* all_sample_points) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;
    int sample_point_index = index % n_sample_point;
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
    float px = in_sample_points[0 * n_sample_point + sample_point_index];
    float py = in_sample_points[1 * n_sample_point + sample_point_index];
    float pz = in_sample_points[2 * n_sample_point + sample_point_index];
    px *= z[0];  py *= z[1];  pz *= z[2];
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    px += t[0];  py += t[1];  pz += t[2];
    all_sample_points[((batch_index * 3 + 0) * n_cube + cube_index) *
        n_sample_point + sample_point_index] = px;
    all_sample_points[((batch_index * 3 + 1) * n_cube + cube_index) *
        n_sample_point + sample_point_index] = py;
    all_sample_points[((batch_index * 3 + 2) * n_cube + cube_index) *
        n_sample_point + sample_point_index] = pz;
  }
}

static __global__ void flip_points_along_z_axis(const int nthreads,
    const int n_point, const float symmetry_plane, float* points) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_point;
    int point_index = index % n_point;
    float* pz = points + (batch_index * 3 + 2) * n_point + point_index;
    *pz = symmetry_plane - ((*pz) - symmetry_plane);
  }
}

static __global__ void fill_point_cube_distance(const int nthreads,
    const int n_cube, const int n_point, const float* in_z, const float* in_q,
    const float* in_t, const float* in_pos, float* point_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_point * n_cube);
    int point_index = (index / n_cube) % n_point;
    int cube_index = index % n_cube;
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
    float px = in_pos[(batch_index * 3 + 0) * n_point + point_index];
    float py = in_pos[(batch_index * 3 + 1) * n_point + point_index];
    float pz = in_pos[(batch_index * 3 + 2) * n_point + point_index];
    px -= t[0];  py -= t[1];  pz -= t[2];
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    conjugate(&qw, &qx, &qy, &qz);
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    float dx = MAX(abs(px) - z[0], 0);
    float dy = MAX(abs(py) - z[1], 0);
    float dz = MAX(abs(pz) - z[2], 0);
    point_cube_distance[(batch_index * n_point + point_index) * n_cube +
        cube_index] = dx * dx + dy * dy + dz * dz;
  }
}

static __global__ void fill_group_cube_distance(const int nthreads,
    const int n_cube, const int n_sample_point,
    const float* point_cube_distance, float* group_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point * n_cube);
    int src_cube_index = (index / (n_sample_point * n_cube)) % n_cube;
    int sample_point_index = (index / n_cube) % n_sample_point;
    int des_cube_index = index % n_cube;
    CudaAtomicAdd(group_cube_distance + ((batch_index * n_cube) + 
        src_cube_index) * n_cube + des_cube_index,
        point_cube_distance[index] / n_sample_point);
  }
}

static __global__ void get_min_distance_cube_index(const int nthreads,
    const int n_cube, const float* group_cube_distance,
    int* min_distance_cube_index) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const float* distance = group_cube_distance + index * n_cube;
    float min_val = distance[0];
    int min_idx = 0;
    for (int i = 1; i < n_cube; ++i) {
      float d = distance[i];
      if (d < min_val) {
        min_val = d;
        min_idx = i;
      }
    }
    min_distance_cube_index[index] = min_idx;
  }
}

static __global__ void get_symmetry_loss(const int nthreads, const int n_cube,
    const int batch_size, const float* group_cube_distance,
    const int* min_distance_cube_index, float* loss_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float distance = group_cube_distance[index * n_cube +
        min_distance_cube_index[index]];
    CudaAtomicAdd(loss_ptr, distance / (batch_size * n_cube));
  }
}

static __global__ void fill_grad_group_cube_distance(const int nthreads,
    const int n_cube, const int batch_size, const float* loss, 
    const int* min_distance_cube_index, float* grad_group_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    grad_group_cube_distance[index * n_cube + min_distance_cube_index[index]] =
        (*loss) / (batch_size * n_cube);
  }
}

static __global__ void fill_grad_point_cube_distance(const int nthreads,
    const int n_cube, const int n_sample_point,
    const float* grad_group_cube_distance, float* grad_point_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point * n_cube);
    int src_cube_index = (index / (n_sample_point * n_cube)) % n_cube;
    int sample_point_index = (index / n_cube) % n_sample_point;
    int des_cube_index = index % n_cube;
    grad_point_cube_distance[index] = grad_group_cube_distance[
        ((batch_index * n_cube) + src_cube_index) * n_cube + des_cube_index] /
        n_sample_point;
  }
}

static __global__ void fill_grad_wrt_zqt_phase_two(const int nthreads,
    const int n_cube, const int n_point, const float* in_z, const float* in_q,
    const float* in_t, const float* in_pos,
    const float* grad_point_cube_distance, float* grad_z, float* grad_q,
    float* grad_t, float* grad_all_sample_points) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_point * n_cube);
    int point_index = (index / n_cube) % n_point;
    int cube_index = index % n_cube;
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
    float px = in_pos[(batch_index * 3 + 0) * n_point + point_index];
    float py = in_pos[(batch_index * 3 + 1) * n_point + point_index];
    float pz = in_pos[(batch_index * 3 + 2) * n_point + point_index];
    px -= t[0];  py -= t[1];  pz -= t[2];
    float tmp_px = px, tmp_py = py, tmp_pz = pz;
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    conjugate(&qw, &qx, &qy, &qz);
    float tmp_qw = qw, tmp_qx = qx, tmp_qy = qy, tmp_qz = qz;  // value before normalize
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    float dx = MAX(abs(px) - z[0], 0);
    float dy = MAX(abs(py) - z[1], 0);
    float dz = MAX(abs(pz) - z[2], 0);

    float* gz = grad_z + (batch_index * n_cube + cube_index) * 3;
    float* gq = grad_q + (batch_index * n_cube + cube_index) * 4;
    float* gt = grad_t + (batch_index * n_cube + cube_index) * 3;
    float grad_distance = grad_point_cube_distance[(batch_index * n_point +
        point_index) * n_cube + cube_index];
    float gdx = grad_distance * 2 * dx;
    float gdy = grad_distance * 2 * dy;
    float gdz = grad_distance * 2 * dz;
    // gradient w.r.t. z
    if (abs(px) - z[0] > 0) {
      CudaAtomicAdd(gz + 0, -gdx);
      gdx *= SIGN(px);
    }
    else {
      gdx = 0.0f;
    }
    if (abs(py) - z[1] > 0) {
      CudaAtomicAdd(gz + 1, -gdy);
      gdy *= SIGN(py);
    }
    else {
      gdy = 0.0f;
    }
    if (abs(pz) - z[2] > 0) {
      CudaAtomicAdd(gz + 2, -gdz);
      gdz *= SIGN(pz);
    }
    else {
      gdz = 0.0f;
    }
    // gradient w.r.t. q
    {
      float grad_rotation_matrix[9];
      grad_rotation_matrix[0] = gdx * tmp_px;
      grad_rotation_matrix[1] = gdx * tmp_py;
      grad_rotation_matrix[2] = gdx * tmp_pz;
      grad_rotation_matrix[3] = gdy * tmp_px;
      grad_rotation_matrix[4] = gdy * tmp_py;
      grad_rotation_matrix[5] = gdy * tmp_pz;
      grad_rotation_matrix[6] = gdz * tmp_px;
      grad_rotation_matrix[7] = gdz * tmp_py;
      grad_rotation_matrix[8] = gdz * tmp_pz;
      float gqw, gqx, gqy, gqz;
      grad_rotation_matrix_to_quaternion(grad_rotation_matrix, tmp_qw, tmp_qx,
          tmp_qy, tmp_qz, &gqw, &gqx, &gqy, &gqz);
      conjugate(&gqw, &gqx, &gqy, &gqz);
      CudaAtomicAdd(gq + 0, gqw);
      CudaAtomicAdd(gq + 1, gqx);
      CudaAtomicAdd(gq + 2, gqy);
      CudaAtomicAdd(gq + 3, gqz);
    }
    t_matvec_kernel(rotation_matrix, &gdx, &gdy, &gdz);
    // gradient w.r.t. t
    {
      CudaAtomicAdd(gt + 0, -gdx);
      CudaAtomicAdd(gt + 1, -gdy);
      CudaAtomicAdd(gt + 2, -gdz);
    }
    // gradient w.r.t. all_sample_points
    {
      CudaAtomicAdd(grad_all_sample_points + (batch_index * 3 + 0) * n_point +
          point_index, gdx);
      CudaAtomicAdd(grad_all_sample_points + (batch_index * 3 + 1) * n_point +
          point_index, gdy);
      CudaAtomicAdd(grad_all_sample_points + (batch_index * 3 + 2) * n_point +
          point_index, gdz);
    }
  }
}

static __global__ void flip_gradient_along_z_axis(const int nthreads,
    const int n_point, float* gradients) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / n_point;
    int point_index = index % n_point;
    gradients[(batch_index * 3 + 2) * n_point + point_index] *= -1.0f;
  }
}

static __global__ void fill_grad_wrt_zqt_phase_one(const int nthreads,
    const int n_cube, const int n_sample_point, const float* in_z,
    const float* in_q, const float* in_t, const float* in_sample_points,
    const float* grad_all_sample_points, float* grad_z, float* grad_q,
    float* grad_t) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;
    int sample_point_index = index % n_sample_point;
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    float px = in_sample_points[0 * n_sample_point + sample_point_index];
    float py = in_sample_points[1 * n_sample_point + sample_point_index];
    float pz = in_sample_points[2 * n_sample_point + sample_point_index];
    float raw_px = px, raw_py = py, raw_pz = pz;
    px *= z[0];  py *= z[1];  pz *= z[2];
    float tmp_px = px, tmp_py = py, tmp_pz = pz;
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    float tmp_qw = qw, tmp_qx = qx, tmp_qy = qy, tmp_qz = qz;
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);

    float* gz = grad_z + (batch_index * n_cube + cube_index) * 3;
    float* gq = grad_q + (batch_index * n_cube + cube_index) * 4;
    float* gt = grad_t + (batch_index * n_cube + cube_index) * 3;
    float gdx = grad_all_sample_points[((batch_index * 3 + 0) * n_cube +
        cube_index) * n_sample_point + sample_point_index];
    float gdy = grad_all_sample_points[((batch_index * 3 + 1) * n_cube +
        cube_index) * n_sample_point + sample_point_index];
    float gdz = grad_all_sample_points[((batch_index * 3 + 2) * n_cube +
        cube_index) * n_sample_point + sample_point_index];
    // gradient w.r.t. t
    {
      CudaAtomicAdd(gt + 0, gdx);
      CudaAtomicAdd(gt + 1, gdy);
      CudaAtomicAdd(gt + 2, gdz);
    }
    // gradient w.r.t. q
    {
      float grad_rotation_matrix[9];
      grad_rotation_matrix[0] = gdx * tmp_px;
      grad_rotation_matrix[1] = gdx * tmp_py;
      grad_rotation_matrix[2] = gdx * tmp_pz;
      grad_rotation_matrix[3] = gdy * tmp_px;
      grad_rotation_matrix[4] = gdy * tmp_py;
      grad_rotation_matrix[5] = gdy * tmp_pz;
      grad_rotation_matrix[6] = gdz * tmp_px;
      grad_rotation_matrix[7] = gdz * tmp_py;
      grad_rotation_matrix[8] = gdz * tmp_pz;
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
      CudaAtomicAdd(gz + 0, gdx * raw_px);
      CudaAtomicAdd(gz + 1, gdy * raw_py);
      CudaAtomicAdd(gz + 2, gdz * raw_pz);
    }
  }
}

static std::vector<float> cube_volume_points_host {
  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
  -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0, -1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  1.0,
  -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0, -1.0,  0.0,  1.0,
};

void compute_symmetry_loss_v3(OpKernelContext* context, const int n_cube,
    const int batch_size, const int depth, const float scale,
    const float* in_z, const float* in_q, const float* in_t, float* loss_ptr) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // sample points in cube volume
  Tensor cube_volume_points;
  int n_sample_point = cube_volume_points_host.size() / 3;
  const TensorShape cube_volume_points_shape({3, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              cube_volume_points_shape, &cube_volume_points));
  auto cube_volume_points_ptr = cube_volume_points.flat<float>().data();
  cudaMemcpy(cube_volume_points_ptr, cube_volume_points_host.data(),
      sizeof(float) * 3 * n_sample_point, cudaMemcpyHostToDevice);
  thrust::transform(thrust::device, cube_volume_points_ptr,
      cube_volume_points_ptr + 3 * n_sample_point,
      thrust::make_constant_iterator(scale), cube_volume_points_ptr,
      thrust::multiplies<float>());

  // get all sampled points location
  Tensor all_sample_points;
  const TensorShape all_sample_points_shape({
      batch_size, 3, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              all_sample_points_shape, &all_sample_points));
  auto all_sample_points_ptr = all_sample_points.flat<float>().data();
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_all_sample_points
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, cube_volume_points_ptr, in_z, in_q,
          in_t, all_sample_points_ptr);

  // flip points along z = 0.5 - 0.5/(2**depth) plane
  float symmetry_plane = static_cast<float>(0.5 * (1.0 - 1.0 / pow(2, depth)));
  const int n_point = n_cube * n_sample_point;
  nthreads = batch_size * n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  flip_points_along_z_axis
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_point, symmetry_plane, all_sample_points_ptr);

  // fill point to cube distance matrix, [batch_size, n_all_point, n_cube]
  Tensor point_cube_distance;
  const TensorShape point_cube_distance_shape({batch_size, n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              point_cube_distance_shape,
                              &point_cube_distance));
  auto point_cube_distance_ptr = point_cube_distance.flat<float>().data();
  nthreads = batch_size * n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, all_sample_points_ptr,
          point_cube_distance_ptr);

  // aggregate group points distance, [batch_size, n_cube, n_cube]
  Tensor group_cube_distance;
  const TensorShape group_cube_distance_shape({batch_size, n_cube, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              group_cube_distance_shape,
                              &group_cube_distance));
  auto group_cube_distance_ptr = group_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, group_cube_distance_ptr,
      group_cube_distance.NumElements());
  nthreads = batch_size * n_cube * n_sample_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_group_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, point_cube_distance_ptr,
          group_cube_distance_ptr);

  // get min distance cube index
  Tensor min_distance_cube_index;
  const TensorShape min_distance_cube_index_shape({batch_size, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              min_distance_cube_index_shape,
                              &min_distance_cube_index));
  auto min_distance_cube_index_ptr =
      min_distance_cube_index.flat<int>().data();
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_min_distance_cube_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, group_cube_distance_ptr,
          min_distance_cube_index_ptr);

  // get symmetry loss
  primitive::gpu_set_zero(context, loss_ptr, 1);
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_symmetry_loss
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, group_cube_distance_ptr,
          min_distance_cube_index_ptr, loss_ptr);
}


void compute_symmetry_loss_v3_grad(OpKernelContext* context, const int n_cube,
    const int batch_size, const int depth, const float scale,
    const float* loss, const float* in_z, const float* in_q, const float* in_t,
    float* grad_z, float* grad_q, float* grad_t) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  /// -- prepare forward medial data for gradient computation --
  // sample points in cube volume
  Tensor cube_volume_points;
  int n_sample_point = cube_volume_points_host.size() / 3;
  const TensorShape cube_volume_points_shape({3, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              cube_volume_points_shape, &cube_volume_points));
  auto cube_volume_points_ptr = cube_volume_points.flat<float>().data();
  cudaMemcpy(cube_volume_points_ptr, cube_volume_points_host.data(),
      sizeof(float) * 3 * n_sample_point, cudaMemcpyHostToDevice);
  thrust::transform(thrust::device, cube_volume_points_ptr,
      cube_volume_points_ptr + 3 * n_sample_point,
      thrust::make_constant_iterator(scale), cube_volume_points_ptr,
      thrust::multiplies<float>());

  // get all sampled points location
  Tensor all_sample_points;
  const TensorShape all_sample_points_shape({
      batch_size, 3, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              all_sample_points_shape, &all_sample_points));
  auto all_sample_points_ptr = all_sample_points.flat<float>().data();
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_all_sample_points
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, cube_volume_points_ptr, in_z, in_q,
          in_t, all_sample_points_ptr);

  // flip points along z = 0.5 - 0.5/(2**depth) plane
  float symmetry_plane = static_cast<float>(0.5 * (1.0 - 1.0 / pow(2, depth)));
  const int n_point = n_cube * n_sample_point;
  nthreads = batch_size * n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  flip_points_along_z_axis
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_point, symmetry_plane, all_sample_points_ptr);

  // fill point to cube distance matrix, [batch_size, n_all_point, n_cube]
  Tensor point_cube_distance;
  const TensorShape point_cube_distance_shape({batch_size, n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              point_cube_distance_shape,
                              &point_cube_distance));
  auto point_cube_distance_ptr = point_cube_distance.flat<float>().data();
  nthreads = batch_size * n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, all_sample_points_ptr,
          point_cube_distance_ptr);

  // aggregate group points distance, [batch_size, n_cube, n_cube]
  Tensor group_cube_distance;
  const TensorShape group_cube_distance_shape({batch_size, n_cube, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              group_cube_distance_shape,
                              &group_cube_distance));
  auto group_cube_distance_ptr = group_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, group_cube_distance_ptr,
      group_cube_distance.NumElements());
  nthreads = batch_size * n_cube * n_sample_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_group_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, point_cube_distance_ptr,
          group_cube_distance_ptr);

  // get min distance cube index
  Tensor min_distance_cube_index;
  const TensorShape min_distance_cube_index_shape({batch_size, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              min_distance_cube_index_shape,
                              &min_distance_cube_index));
  auto min_distance_cube_index_ptr =
      min_distance_cube_index.flat<int>().data();
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_min_distance_cube_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, group_cube_distance_ptr,
          min_distance_cube_index_ptr);
  /// ----------------------------------------------------------
  
  // splash gradient to point cube distance
  Tensor grad_group_cube_distance;
  const TensorShape ggcd_shape({batch_size, n_cube, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, ggcd_shape,
                              &grad_group_cube_distance));
  auto ggcd_ptr = grad_group_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, ggcd_ptr,
      grad_group_cube_distance.NumElements());
  nthreads = batch_size * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_group_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, batch_size, loss, min_distance_cube_index_ptr,
          ggcd_ptr);

  // gradient of point cube distance
  Tensor grad_point_cube_distance;
  const TensorShape gpcd_shape({batch_size, n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, gpcd_shape,
                              &grad_point_cube_distance));
  auto gpcd_ptr = grad_point_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, gpcd_ptr,
      grad_point_cube_distance.NumElements());
  nthreads = batch_size * n_point * n_cube;  // n_point == n_cube * n_sample_point
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, ggcd_ptr, gpcd_ptr);

  // init zero gradient
  primitive::gpu_set_zero(context, grad_z, batch_size * n_cube * 3);
  primitive::gpu_set_zero(context, grad_q, batch_size * n_cube * 4);
  primitive::gpu_set_zero(context, grad_t, batch_size * n_cube * 3);

  // gradient w.r.t. (z, q, t) for phase two: fill distance matrix
  Tensor grad_all_sample_points;
  const TensorShape grad_all_sample_points_shape({batch_size, 3, n_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              grad_all_sample_points_shape,
                              &grad_all_sample_points));
  auto gasp_ptr = grad_all_sample_points.flat<float>().data();
  primitive::gpu_set_zero(context, gasp_ptr,
      grad_all_sample_points.NumElements());
  nthreads = batch_size * n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_zqt_phase_two
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, all_sample_points_ptr,
          gpcd_ptr, grad_z, grad_q, grad_t, gasp_ptr);

  // flip gradient for z axis
  nthreads = batch_size * n_point;
  config = GetCudaLaunchConfig(nthreads, d);
  flip_gradient_along_z_axis
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_point, gasp_ptr);

  // gradient w.r.t. (z, q, t) for phase one: get all sampled points
  nthreads = batch_size * n_cube * n_sample_point;  // n_point == n_cube * n_sample_point
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_zqt_phase_one
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, in_z, in_q, in_t,
          cube_volume_points_ptr, gasp_ptr, grad_z, grad_q, grad_t);
}

}  // namespace tensorflow
