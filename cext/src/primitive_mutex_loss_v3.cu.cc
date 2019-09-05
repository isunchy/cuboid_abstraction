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

static __global__ void fill_transform_points(const int nthreads,
    const int n_cube, const int n_sample_point, const float* sample_points,
    const float* in_z, const float* in_q, const float* in_t,
    float* transformed_points) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;
    int sample_point_index = index % n_sample_point;
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    const float* t = in_t + (batch_index * n_cube + cube_index) * 3;
    float px = sample_points[0 * n_sample_point + sample_point_index];
    float py = sample_points[1 * n_sample_point + sample_point_index];
    float pz = sample_points[2 * n_sample_point + sample_point_index];
    px *= z[0];  py *= z[1];  pz *= z[2];
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    px += t[0];  py += t[1];  pz += t[2];
    transformed_points[((batch_index * n_cube + cube_index) * 3 + 0) *
        n_sample_point + sample_point_index] = px;
    transformed_points[((batch_index * n_cube + cube_index) * 3 + 1) *
        n_sample_point + sample_point_index] = py;
    transformed_points[((batch_index * n_cube + cube_index) * 3 + 2) *
        n_sample_point + sample_point_index] = pz;
  }
}

static __global__ void fill_points_cube_axis_distance(const int nthreads,
    const int n_cube, const int n_sample_point,
    const float* transformed_points, const float* in_z, const float* in_q,
    const float* in_t, float* distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_cube * n_sample_point);
    int s_cube_index = (index / (n_cube * n_sample_point)) % n_cube;
    int t_cube_index = (index / n_sample_point) % n_cube;
    int sample_point_index = index % n_sample_point;
    if (t_cube_index != s_cube_index) {
      const float* z = in_z + (batch_index * n_cube + t_cube_index) * 3;
      const float* q = in_q + (batch_index * n_cube + t_cube_index) * 4;
      const float* t = in_t + (batch_index * n_cube + t_cube_index) * 3;
      float px = transformed_points[((batch_index * n_cube + s_cube_index) * 3
          + 0) * n_sample_point + sample_point_index];
      float py = transformed_points[((batch_index * n_cube + s_cube_index) * 3
          + 1) * n_sample_point + sample_point_index];
      float pz = transformed_points[((batch_index * n_cube + s_cube_index) * 3
          + 2) * n_sample_point + sample_point_index];
      px -= t[0];  py -= t[1];  pz -= t[2];
      float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
      float rotation_matrix[9];
      conjugate(&qw, &qx, &qy, &qz);
      as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
      matvec_kernel(rotation_matrix, &px, &py, &pz);
      distance[index * 3 + 0] = MAX(z[0] - abs(px), 0);
      distance[index * 3 + 1] = MAX(z[1] - abs(py), 0);
      distance[index * 3 + 2] = MAX(z[2] - abs(pz), 0);
    }
  }
}

static __global__ void fill_points_cube_mutex_distance(const int nthreads,
    const float* axis_distance, float* mutex_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float dx = axis_distance[index * 3 + 0];
    float dy = axis_distance[index * 3 + 1];
    float dz = axis_distance[index * 3 + 2];
    mutex_distance[index] = dx;
    if (dy < mutex_distance[index]) {
      mutex_distance[index] = dy;
    }
    if (dz < mutex_distance[index]) {
      mutex_distance[index] = dz;
    }
  }
}

static __global__ void get_points_max_mutex_distance_index(const int nthreads,
    const int n_cube, const int n_sample_point, const float* mutex_distance,
    int* max_distance_cube_index) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;  // src cube
    int sample_point_index = index % n_sample_point;
    float max_val = -1.0f;
    int max_idx = 0;
    for (int i = 0; i < n_cube; ++i) {  // des cube
      if (i != cube_index) {
        float distance = mutex_distance[((batch_index * n_cube + cube_index) *
            n_cube + i) * n_sample_point + sample_point_index];
        if (distance > max_val) {
          max_val = distance;
          max_idx = i;
        }
      }
    }
    max_distance_cube_index[(batch_index * n_cube + cube_index) *
        n_sample_point + sample_point_index] = max_idx;
  }
}

static __global__ void get_mutex_loss(const int nthreads, const int n_cube,
    const int n_sample_point, const int batch_size,
    const float* mutex_distance, const int* max_distance_cube_index,
    float* loss_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;  // src cube
    int sample_point_index = index % n_sample_point;
    int max_cube_index = max_distance_cube_index[(batch_index * n_cube +
        cube_index) * n_sample_point + sample_point_index];
    float distance = mutex_distance[((batch_index * n_cube + cube_index) *
        n_cube + max_cube_index) * n_sample_point + sample_point_index];
    CudaAtomicAdd(loss_ptr, distance / (batch_size * n_cube * n_sample_point));
  }
}

static __global__ void fill_grad_points_cube_mutex_distance(const int nthreads,
    const int n_cube, const int n_sample_point, const int batch_size,
    const float* loss, const int* max_distance_cube_index,
    float* grad_points_cube_mutex_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;  // src cube
    int sample_point_index = index % n_sample_point;
    int max_cube_index = max_distance_cube_index[(batch_index * n_cube +
        cube_index) * n_sample_point + sample_point_index];
    grad_points_cube_mutex_distance[((batch_index * n_cube + cube_index) *
        n_cube + max_cube_index) * n_sample_point + sample_point_index] =
        *(loss) / (batch_size * n_cube * n_sample_point);
  }  
}

static __global__ void fill_grad_points_cube_axis_distance(const int nthreads,
    const float* points_cube_axis_distance,
    const float* grad_points_cube_mutex_distance,
    float* grad_points_cube_axis_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float dx = points_cube_axis_distance[index * 3 + 0];
    float dy = points_cube_axis_distance[index * 3 + 1];
    float dz = points_cube_axis_distance[index * 3 + 2];
    float min_axis_distance = dx;
    int min_axis_index = 0;
    if (dy < min_axis_distance) {
      min_axis_distance = dy;
      min_axis_index = 1;
    }
    if (dz < min_axis_distance) {
      min_axis_index = 2;
    }
    grad_points_cube_axis_distance[index * 3 + min_axis_index] =
        grad_points_cube_mutex_distance[index];
  }
}

static __global__ void fill_grad_transformed_points(const int nthreads,
    const int n_cube, const int n_sample_point,
    const float* transformed_points,
    const float* grad_points_cube_axis_distance, const float* in_z,
    const float* in_q, const float* in_t, float* grad_z, float* grad_q,
    float* grad_t, float* grad_transformed_points) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_cube * n_sample_point);
    int s_cube_index = (index / (n_cube * n_sample_point)) % n_cube;
    int t_cube_index = (index / n_sample_point) % n_cube;
    int sample_point_index = index % n_sample_point;
    if (t_cube_index != s_cube_index) {
      const float* z = in_z + (batch_index * n_cube + t_cube_index) * 3;
      const float* q = in_q + (batch_index * n_cube + t_cube_index) * 4;
      const float* t = in_t + (batch_index * n_cube + t_cube_index) * 3;
      float px = transformed_points[((batch_index * n_cube + s_cube_index) * 3
          + 0) * n_sample_point + sample_point_index];
      float py = transformed_points[((batch_index * n_cube + s_cube_index) * 3
          + 1) * n_sample_point + sample_point_index];
      float pz = transformed_points[((batch_index * n_cube + s_cube_index) * 3
          + 2) * n_sample_point + sample_point_index];
      px -= t[0];  py -= t[1];  pz -= t[2];
      float tmp_px = px, tmp_py = py, tmp_pz = pz;
      float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
      float rotation_matrix[9];
      conjugate(&qw, &qx, &qy, &qz);
      float tmp_qw = qw, tmp_qx = qx, tmp_qy = qy, tmp_qz = qz;
      as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
      matvec_kernel(rotation_matrix, &px, &py, &pz);

      float* gz = grad_z + (batch_index * n_cube + t_cube_index) * 3;
      float* gq = grad_q + (batch_index * n_cube + t_cube_index) * 4;
      float* gt = grad_t + (batch_index * n_cube + t_cube_index) * 3;
      float gdx = grad_points_cube_axis_distance[index * 3 + 0];
      float gdy = grad_points_cube_axis_distance[index * 3 + 1];
      float gdz = grad_points_cube_axis_distance[index * 3 + 2];
      // gradient w.r.t. z
      {
        if (z[0] - abs(px) > 0) {
          CudaAtomicAdd(gz + 0, gdx);
          gdx *= -SIGN(px);
        }
        else {
          gdx = 0.0f;
        }
        if (z[1] - abs(py) > 0) {
          CudaAtomicAdd(gz + 1, gdy);
          gdy *= -SIGN(py);
        }
        else {
          gdy = 0.0f;
        }
        if (z[2] - abs(pz) > 0) {
          CudaAtomicAdd(gz + 2, gdz);
          gdz *= -SIGN(pz);
        }
        else {
          gdz = 0.0f;
        }
      }
      // gradients w.r.t. q
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
        grad_rotation_matrix_to_quaternion(grad_rotation_matrix, tmp_qw,
            tmp_qx, tmp_qy, tmp_qz, &gqw, &gqx, &gqy, &gqz);
        conjugate(&gqw, &gqx, &gqy, &gqz);
        CudaAtomicAdd(gq + 0, gqw);
        CudaAtomicAdd(gq + 1, gqx);
        CudaAtomicAdd(gq + 2, gqy);
        CudaAtomicAdd(gq + 3, gqz);
      }
      t_matvec_kernel(rotation_matrix, &gdx, &gdy, &gdz);
      // gradients w.r.t. t
      {
        CudaAtomicAdd(gt + 0, -gdx);
        CudaAtomicAdd(gt + 1, -gdy);
        CudaAtomicAdd(gt + 2, -gdz);
      }
      // gradients w.r.t. transformed points
      {
        CudaAtomicAdd(grad_transformed_points + ((batch_index * n_cube +
            s_cube_index) * 3 + 0) * n_sample_point + sample_point_index, gdx);
        CudaAtomicAdd(grad_transformed_points + ((batch_index * n_cube +
            s_cube_index) * 3 + 1) * n_sample_point + sample_point_index, gdy);
        CudaAtomicAdd(grad_transformed_points + ((batch_index * n_cube +
            s_cube_index) * 3 + 2) * n_sample_point + sample_point_index, gdz);
      }
    }
  }
}

static __global__ void fill_grad_wrt_zqt(const int nthreads, const int n_cube,
    const int n_sample_point, const float* sample_points,
    const float* grad_transformed_points, const float* in_z, const float* in_q,
    const float* in_t, float* grad_z, float* grad_q, float* grad_t) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int batch_index = index / (n_cube * n_sample_point);
    int cube_index = (index / n_sample_point) % n_cube;
    int sample_point_index = index % n_sample_point;
    const float* z = in_z + (batch_index * n_cube + cube_index) * 3;
    const float* q = in_q + (batch_index * n_cube + cube_index) * 4;
    float px = sample_points[0 * n_sample_point + sample_point_index];
    float py = sample_points[1 * n_sample_point + sample_point_index];
    float pz = sample_points[2 * n_sample_point + sample_point_index];
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
    float gdx = grad_transformed_points[((batch_index * n_cube + cube_index) *
        3 + 0) * n_sample_point + sample_point_index];
    float gdy = grad_transformed_points[((batch_index * n_cube + cube_index) *
        3 + 1) * n_sample_point + sample_point_index];
    float gdz = grad_transformed_points[((batch_index * n_cube + cube_index) *
        3 + 2) * n_sample_point + sample_point_index];
    // gradients w.r.t t
    {
      CudaAtomicAdd(gt + 0, gdx);
      CudaAtomicAdd(gt + 1, gdy);
      CudaAtomicAdd(gt + 2, gdz);
    }
    // gradients w.r.t q
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
    // gradients w.r.t z
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

void compute_mutex_loss_v3(OpKernelContext* context, const int n_cube,
    const int batch_size, const float scale, const float* in_z,
    const float* in_q, const float* in_t, float* loss_ptr) {
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

  // fill transformed sampled points [batch_size, n_cube, 3, n_sample_point]
  Tensor transformed_points;
  const TensorShape transformed_points_shape({
      batch_size, n_cube, 3, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              transformed_points_shape, &transformed_points));
  auto transformed_points_ptr = transformed_points.flat<float>().data();
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_transform_points
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, cube_volume_points_ptr, in_z, in_q,
          in_t, transformed_points_ptr);

  // fill axis distance between transformed points with other cubes
  Tensor points_cube_axis_distance;
  const TensorShape pcad_shape({
      batch_size, n_cube, n_cube, n_sample_point, 3});  // Note the dim order!!
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, pcad_shape,
                              &points_cube_axis_distance));
  auto pcad_ptr = points_cube_axis_distance.flat<float>().data();
  primitive::gpu_set_zero(context, pcad_ptr,
      points_cube_axis_distance.NumElements());
  nthreads = batch_size * n_cube * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_points_cube_axis_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, transformed_points_ptr, in_z, in_q,
          in_t, pcad_ptr);

  // fill mutex distance between transform points with other cubes
  Tensor points_cube_mutex_distance;
  const TensorShape pcmd_shape({batch_size, n_cube, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, pcmd_shape,
                              &points_cube_mutex_distance));
  auto pcmd_ptr = points_cube_mutex_distance.flat<float>().data();
  nthreads = batch_size * n_cube * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_points_cube_mutex_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, pcad_ptr, pcmd_ptr);

  // get max mutex distance cube index for each transformed points
  Tensor max_mutex_distance_cube_index;
  const TensorShape mmdci_shape({batch_size, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, mmdci_shape,
                              &max_mutex_distance_cube_index));
  auto mmdci_ptr = max_mutex_distance_cube_index.flat<int>().data();
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  get_points_max_mutex_distance_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, pcmd_ptr, mmdci_ptr);

  // get mutex loss
  primitive::gpu_set_zero(context, loss_ptr, 1);
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  get_mutex_loss
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, batch_size, pcmd_ptr, mmdci_ptr,
          loss_ptr);
}


void compute_mutex_loss_grad_v3(OpKernelContext* context, const int n_cube,
    const int batch_size, const float scale, const float* loss,
    const float* in_z, const float* in_q, const float* in_t, float* grad_z,
    float* grad_q, float* grad_t) {
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

  // fill transformed sampled points [batch_size, n_cube, 3, n_sample_point]
  Tensor transformed_points;
  const TensorShape transformed_points_shape({
      batch_size, n_cube, 3, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              transformed_points_shape, &transformed_points));
  auto transformed_points_ptr = transformed_points.flat<float>().data();
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_transform_points
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, cube_volume_points_ptr, in_z, in_q,
          in_t, transformed_points_ptr);

  // fill axis distance between transformed points with other cubes
  Tensor points_cube_axis_distance;
  const TensorShape pcad_shape({
      batch_size, n_cube, n_cube, n_sample_point, 3});  // Note the dim order!!
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, pcad_shape,
                              &points_cube_axis_distance));
  auto pcad_ptr = points_cube_axis_distance.flat<float>().data();
  primitive::gpu_set_zero(context, pcad_ptr,
      points_cube_axis_distance.NumElements());
  nthreads = batch_size * n_cube * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_points_cube_axis_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, transformed_points_ptr, in_z, in_q,
          in_t, pcad_ptr);

  // fill mutex distance between transform points with other cubes
  Tensor points_cube_mutex_distance;
  const TensorShape pcmd_shape({batch_size, n_cube, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, pcmd_shape,
                              &points_cube_mutex_distance));
  auto pcmd_ptr = points_cube_mutex_distance.flat<float>().data();
  nthreads = batch_size * n_cube * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_points_cube_mutex_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, pcad_ptr, pcmd_ptr);

  // get max mutex distance cube index for each transformed points
  Tensor max_mutex_distance_cube_index;
  const TensorShape mmdci_shape({batch_size, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, mmdci_shape,
                              &max_mutex_distance_cube_index));
  auto mmdci_ptr = max_mutex_distance_cube_index.flat<int>().data();
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  get_points_max_mutex_distance_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, pcmd_ptr, mmdci_ptr);
  /// ----------------------------------------------------------
  
  // splash gradient to point to cube mutex distance
  Tensor grad_points_cube_mutex_distance;
  const TensorShape gpcmd_shape({batch_size, n_cube, n_cube, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, gpcmd_shape,
                              &grad_points_cube_mutex_distance));
  auto gpcmd_ptr = grad_points_cube_mutex_distance.flat<float>().data();
  primitive::gpu_set_zero(context, gpcmd_ptr,
      grad_points_cube_mutex_distance.NumElements());
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_points_cube_mutex_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, batch_size, loss, mmdci_ptr,
          gpcmd_ptr);

  // gradient for point to cube axis distance
  Tensor grad_points_cube_axis_distance;
  const TensorShape gpcad_shape({
      batch_size, n_cube, n_cube, n_sample_point, 3});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, gpcad_shape,
                              &grad_points_cube_axis_distance));
  auto gpcad_ptr = grad_points_cube_axis_distance.flat<float>().data();
  primitive::gpu_set_zero(context, gpcad_ptr,
      grad_points_cube_axis_distance.NumElements());
  nthreads = batch_size * n_cube * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_points_cube_axis_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, pcad_ptr, gpcmd_ptr, gpcad_ptr);

  // init zero gradient
  primitive::gpu_set_zero(context, grad_z, batch_size * n_cube * 3);
  primitive::gpu_set_zero(context, grad_q, batch_size * n_cube * 4);
  primitive::gpu_set_zero(context, grad_t, batch_size * n_cube * 3);

  // gradient for transformed sampled points
  Tensor grad_transformed_points;
  const TensorShape grad_transformed_points_shape({
      batch_size, n_cube, 3, n_sample_point});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              grad_transformed_points_shape,
                              &grad_transformed_points));
  auto grad_transformed_points_ptr =
      grad_transformed_points.flat<float>().data();
  primitive::gpu_set_zero(context, grad_transformed_points_ptr,
      grad_transformed_points.NumElements());
  nthreads = batch_size * n_cube * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_transformed_points
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, transformed_points_ptr, gpcad_ptr,
          in_z, in_q, in_t, grad_z, grad_q, grad_t,
          grad_transformed_points_ptr);

  // gradient w.r.t. (z, q, t)
  nthreads = batch_size * n_cube * n_sample_point;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_zqt
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_sample_point, cube_volume_points_ptr,
          grad_transformed_points_ptr, in_z, in_q, in_t, grad_z, grad_q,
          grad_t);
}

}  // namespace tensorflow
