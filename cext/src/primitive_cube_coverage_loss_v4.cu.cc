#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

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

static __global__ void fill_group_cube_distance(const int nthreads,
    const int n_cube, const int n_point, const float* point_cube_distance,
    const int* point_group_index, float* group_cube_distance,
    int* group_point_count) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int point_index = index / n_cube;
    int cube_index = index % n_cube;
    int group_index = point_group_index[point_index];
    CudaAtomicAdd(group_cube_distance + group_index * n_cube + cube_index,
        point_cube_distance[index]);
    if (cube_index == 0) {
      CudaAtomicAdd(group_point_count + group_index, 1);
    }
  }
}

static __global__ void get_mean_distance(const int nthreads, const int n_cube,
    const int* group_point_count, float* group_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int group_index = index / n_cube;
    int point_count = group_point_count[group_index];
    if (point_count != 0) {
      group_cube_distance[index] /= point_count;
    }
  }
}

static __global__ void get_min_distance_cube_index(const int nthreads,
    const int n_cube, const float* group_cube_distance,
    int* min_distance_cube_index) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const float* distance = group_cube_distance + index * n_cube;
    float min_val = distance[0];
    float min_idx = 0;
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

static __global__ void get_cube_coverage_loss(const int nthreads,
    const int n_cube, const int n_src_cube, const int batch_size,
    const float* group_cube_distance, const int* min_distance_cube_index,
    float* loss_ptr) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float distance = group_cube_distance[index * n_cube +
        min_distance_cube_index[index]];
    CudaAtomicAdd(loss_ptr, distance / (batch_size * n_src_cube));
  }
}

static __global__ void fill_grad_group_cube_distance(const int nthreads,
    const int n_cube, const int n_src_cube, const int batch_size,
    const float* loss, const int* min_distance_cube_index,
    float* grad_group_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    grad_group_cube_distance[index * n_cube + min_distance_cube_index[index]] =
        (*loss) / (batch_size * n_src_cube);
  }
}

static __global__ void fill_grad_point_cube_distance(const int nthreads,
    const int n_cube, const int n_point, const float* grad_group_cube_distance,
    const int* point_group_index, const int* group_point_count,
    float* grad_point_cube_distance) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int point_index = index / n_cube;
    int cube_index = index % n_cube;
    int group_index = point_group_index[point_index];
    int point_count = group_point_count[group_index];
    if (point_count != 0) {
      grad_point_cube_distance[point_index * n_cube + cube_index] =
          grad_group_cube_distance[group_index * n_cube + cube_index] /
          point_count;
    }
  }
}

static __global__ void fill_grad_wrt_zqt(const int nthreads, const int n_cube,
    const int n_point, const float* in_z, const float* in_q,
    const float* in_t, const float* in_pos,
    const float* grad_point_cube_distance, float* grad_z, float* grad_q,
    float* grad_t) {
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
    float tmp_px = px, tmp_py = py, tmp_pz = pz;
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float rotation_matrix[9];
    conjugate(&qw, &qx, &qy, &qz);
    float tmp_qw = qw, tmp_qx = qx, tmp_qy = qy, tmp_qz = qz;
    as_rotation_matrix(qw, qx, qy, qz, rotation_matrix);
    matvec_kernel(rotation_matrix, &px, &py, &pz);
    float dx = MAX(abs(px) - z[0], 0);
    float dy = MAX(abs(py) - z[1], 0);
    float dz = MAX(abs(pz) - z[2], 0);

    float* gz = grad_z + (batch_index * n_cube + cube_index) * 3;
    float* gq = grad_q + (batch_index * n_cube + cube_index) * 4;
    float* gt = grad_t + (batch_index * n_cube + cube_index) * 3;
    float grad_distance = grad_point_cube_distance[point_index * n_cube +
        cube_index];
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
  }
}

void compute_cube_coverage_loss_v4(OpKernelContext* context, const int n_cube,
    const int n_point, const int n_src_cube, const int batch_size,
    const float* in_z, const float* in_q, const float* in_t,
    const float* in_pos, const int* point_group_index, float* loss_ptr,
    int* relatoin_ptr) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  // fill point to cube distance matrix, [n_point, n_cube]
  Tensor point_cube_distance;
  const TensorShape point_cube_distance_shape({n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              point_cube_distance_shape,
                              &point_cube_distance));
  auto point_cube_distance_ptr = point_cube_distance.flat<float>().data();
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, in_pos,
          point_cube_distance_ptr);

  // aggregate group points distance, [n_src_cube, n_cube]
  Tensor group_cube_distance;
  const TensorShape group_cube_distance_shape({
      batch_size, n_src_cube, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              group_cube_distance_shape,
                              &group_cube_distance));
  auto group_cube_distance_ptr = group_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, group_cube_distance_ptr,
      group_cube_distance.NumElements());
  Tensor group_point_count;
  const TensorShape group_point_count_shape({batch_size, n_src_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              group_point_count_shape, &group_point_count));
  auto group_point_count_ptr = group_point_count.flat<int>().data();
  primitive::gpu_set_zero(context, group_point_count_ptr,
      group_point_count.NumElements());
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_group_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, point_cube_distance_ptr,
          point_group_index, group_cube_distance_ptr, group_point_count_ptr);

  // get mean group cube distance
  nthreads = batch_size * n_src_cube * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_mean_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, group_point_count_ptr, group_cube_distance_ptr);

  // get min distance cube index
  Tensor min_distance_cube_index;
  const TensorShape min_distance_cube_index_shape({batch_size, n_src_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              min_distance_cube_index_shape,
                              &min_distance_cube_index));
  auto min_distance_cube_index_ptr = min_distance_cube_index.flat<int>().data();
  nthreads = batch_size * n_src_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_min_distance_cube_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, group_cube_distance_ptr,
           min_distance_cube_index_ptr);

  // set cube relation
  cudaMemcpy(relatoin_ptr, min_distance_cube_index_ptr,
      batch_size * n_src_cube * sizeof(int), cudaMemcpyDeviceToDevice);

  // get cube coverage loss
  primitive::gpu_set_zero(context, loss_ptr, 1);
  nthreads = batch_size * n_src_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_cube_coverage_loss
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_src_cube, batch_size, group_cube_distance_ptr,
          min_distance_cube_index_ptr, loss_ptr);
}

void compute_cube_coverage_loss_v4_grad(OpKernelContext* context,
    const int n_cube, const int n_point, const int n_src_cube,
    const int batch_size, const float* loss, const float* in_z,
    const float* in_q, const float* in_t, const float* in_pos,
    const int* point_group_index, float* grad_z, float* grad_q,
    float* grad_t) {
  // get GPU device
  GPUDevice d = context->eigen_device<GPUDevice>();
  CudaLaunchConfig config;
  int nthreads;

  /// -- prepare forward medial data for gradient computation -- 
  // fill point to cube distance matrix, [n_point, n_cube]
  Tensor point_cube_distance;
  const TensorShape point_cube_distance_shape({n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              point_cube_distance_shape,
                              &point_cube_distance));
  auto point_cube_distance_ptr = point_cube_distance.flat<float>().data();
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, in_pos,
          point_cube_distance_ptr);

  // aggregate group points distance, [n_src_cube, n_cube]
  Tensor group_cube_distance;
  const TensorShape group_cube_distance_shape({
      batch_size, n_src_cube, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                              group_cube_distance_shape,
                              &group_cube_distance));
  auto group_cube_distance_ptr = group_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, group_cube_distance_ptr,
      group_cube_distance.NumElements());
  Tensor group_point_count;
  const TensorShape group_point_count_shape({batch_size, n_src_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              group_point_count_shape, &group_point_count));
  auto group_point_count_ptr = group_point_count.flat<int>().data();
  primitive::gpu_set_zero(context, group_point_count_ptr,
      group_point_count.NumElements());
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_group_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, point_cube_distance_ptr,
          point_group_index, group_cube_distance_ptr, group_point_count_ptr);

  // get mean group cube distance
  nthreads = batch_size * n_src_cube * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_mean_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, group_point_count_ptr, group_cube_distance_ptr);

  // get min distance cube index
  Tensor min_distance_cube_index;
  const TensorShape min_distance_cube_index_shape({batch_size, n_src_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32,
                              min_distance_cube_index_shape,
                              &min_distance_cube_index));
  auto min_distance_cube_index_ptr = min_distance_cube_index.flat<int>().data();
  nthreads = batch_size * n_src_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  get_min_distance_cube_index
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, group_cube_distance_ptr,
           min_distance_cube_index_ptr);
  /// ----------------------------------------------------------

  // splash gradient to group points distance
  Tensor grad_group_cube_distance;
  const TensorShape ggcd_shape({batch_size, n_src_cube, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, ggcd_shape,
                              &grad_group_cube_distance));
  auto ggcd_ptr = grad_group_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, ggcd_ptr,
      grad_group_cube_distance.NumElements());
  nthreads = batch_size * n_src_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_group_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_src_cube, batch_size, loss,
          min_distance_cube_index_ptr, ggcd_ptr);

  // gradient of point cube distance
  Tensor grad_point_cube_distance;
  const TensorShape gpcd_shape({n_point, n_cube});
  OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, gpcd_shape,
                              &grad_point_cube_distance));
  auto gpcd_ptr = grad_point_cube_distance.flat<float>().data();
  primitive::gpu_set_zero(context, gpcd_ptr,
      grad_point_cube_distance.NumElements());
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_point_cube_distance
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, ggcd_ptr, point_group_index,
          group_point_count_ptr, gpcd_ptr);

  // init zero gradient
  primitive::gpu_set_zero(context, grad_z, batch_size * n_cube * 3);
  primitive::gpu_set_zero(context, grad_q, batch_size * n_cube * 4);
  primitive::gpu_set_zero(context, grad_t, batch_size * n_cube * 3);

  // gradient w.r.t. (z, q, t)
  nthreads = n_point * n_cube;
  config = GetCudaLaunchConfig(nthreads, d);
  fill_grad_wrt_zqt
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          nthreads, n_cube, n_point, in_z, in_q, in_t, in_pos, gpcd_ptr,
          grad_z, grad_q, grad_t);
}

}  // namespace tensorflow
