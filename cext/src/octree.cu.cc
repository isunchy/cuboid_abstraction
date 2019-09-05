#define EIGEN_USE_THREADS

#include "octree.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {
perftools::gputools::DeviceMemory<float> AsDeviceMemory(
    const float* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(
      const_cast<float*>(cuda_memory));
  perftools::gputools::DeviceMemory<float> typed(wrapped);
  return typed;
}
}  // namespace

namespace octree {

__global__ void octree2col_kernel(float* data_col, const float* data_octree,
    int height, int kernel_dim, int stride, const int* neigh, const int* ni,
    int height_col, int n, int thread_num) {
  CUDA_1D_KERNEL_LOOP(i, thread_num) {
    int h = i % height_col;
    int h1 = h + n * height_col;
    if (h1 >= height) { data_col[i] = 0; continue; }
    int t = i / height_col;
    int k = t % kernel_dim;
    int c = t / kernel_dim;
    int octree_h = height << 3 * (stride - 1);

    int index = stride == 2 ? (h1 << 6) + ni[k] :
      (h1 >> 3 << 6) + ni[(h1 % 8) * kernel_dim + k];
    int p = neigh[index];
    data_col[i] = p == -1 ? float(0) : data_octree[c * octree_h + p];
  }
}

__global__ void col2octree_kernel(const float* data_col, float* data_octree,
    int height, int kernel_dim, int stride, const int* neigh, const int* ni,
    int height_col, int n, int thread_num) {
  CUDA_1D_KERNEL_LOOP(i, thread_num) {
    int h = i % height_col;
    int h1 = h + n * height_col;
    if (h1 >= height) continue;
    int t = i / height_col;
    int k = t % kernel_dim;
    int c = t / kernel_dim;
    int octree_h = height << 3 * (stride - 1);

    int index = stride == 2 ? (h1 << 6) + ni[k] :
      (h1 >> 3 << 6) + ni[(h1 % 8) * kernel_dim + k];
    int p = neigh[index];
    if (p != -1) CudaAtomicAdd(data_octree + c * octree_h + p, data_col[i]);
  }
}

void octree2col_gpu(OpKernelContext* ctx, float* data_col,
    const float* data_octree, int channel, int height, int kernel_size,
    int stride, const int* neigh, const int* ni, int height_col, int n) {
  const int kernel = kernel_size * kernel_size * kernel_size;
  const int thread_num = channel * kernel * height_col;
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(thread_num, d);
  octree2col_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          data_col, data_octree, height, kernel, stride, neigh, ni, height_col,
          n, thread_num);
}

void col2octree_gpu(OpKernelContext* ctx, const float* data_col,
    float* data_octree, int channel, int height, int kernel_size, int stride,
    const int* neigh, const int* ni, int height_col, int n) {
  const int kernel = kernel_size * kernel_size * kernel_size;
  const int thread_num = channel * kernel * height_col;
  int octree_h = height << 3 * (stride - 1);
  // set data_octree to zero ONCE when n ==0
  if (n == 0) tensorflow_gpu_set_zero(ctx, data_octree, channel * octree_h);
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(thread_num, d);
  col2octree_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          data_col, data_octree, height, kernel, stride, neigh, ni, height_col,
          n, thread_num);
}

__global__ void pad_forward_kernel(float* Y, const int Hy, const float* X,
    int Hx, const int* label, int thread_num) {
  CUDA_1D_KERNEL_LOOP(i, thread_num) {
    int h = i % Hy;
    int c = i / Hy;
    int idx = label[h];
    Y[i] = idx == -1 ? float(0) : X[c * Hx + idx];
  }
}

__global__ void pad_backward_kernel(float* X, int Hx, const float* Y, int Hy,
    const int* label, int thread_num) {
  CUDA_1D_KERNEL_LOOP(i, thread_num) {
    int h = i % Hy;
    int c = i / Hy;
    int idx = label[h];
    if (idx != -1) {
      X[c * Hx + idx] = Y[i];
    }
  }
}

void pad_forward_gpu(OpKernelContext* ctx, float* Y, int Hy, int Cy,
    const float* X, int Hx, const int* label) {
  int n = Hy * Cy; // Note: Cx == Cy
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  pad_forward_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          Y, Hy, X, Hx, label, n);
}

void pad_backward_gpu(OpKernelContext* ctx, float* X, int Hx, int Cx,
    const float* Y, int Hy, const int* label) {
  int n = Hy * Cx; // Note: Cx == Cy
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  pad_backward_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          X, Hx, Y, Hy, label, n);
}

__global__ void gen_key_kernel(int* key_split, const int* key,
    const int* children, int thread_num) {
  typedef unsigned char ubyte;
  CUDA_1D_KERNEL_LOOP(id, thread_num) {
    int i = id >> 3;
    int j = id % 8;
    int label = children[i];
    if (label != -1) {
      const ubyte* k0 = reinterpret_cast<const ubyte*>(key + i);
      ubyte* k1 = reinterpret_cast<ubyte*>(key_split + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] = k0[3];
    }
  }
}

__global__ void gen_full_key_kernel(int* key, int depth, int batch_size,
    int thread_num) {
  CUDA_1D_KERNEL_LOOP(i, thread_num) {
    unsigned node_num = 1 << 3 * depth;
    unsigned k = i % node_num;
    unsigned xyz = 0;
    unsigned char* ptr = reinterpret_cast<unsigned char*>(&xyz);
#pragma unroll
    for (int d = 0; d < depth; ++d) {
      ptr[0] |= (k & (1 << 3 * d + 2)) >> (2 * d + 2);
      ptr[1] |= (k & (1 << 3 * d + 1)) >> (2 * d + 1);
      ptr[2] |= (k & (1 << 3 * d + 0)) >> (2 * d + 0);
    }
    ptr[3] = i / node_num;
    key[i] = xyz;
  }
}

__global__ void calc_neigh_kernel(int* neigh_split, const int* neigh,
  const int* children, const int* parent, const int* dis, int thread_num) {
  CUDA_1D_KERNEL_LOOP(id, thread_num) {
    int i = id >> 6;
    int j = id % 64;

    int l0 = children[i];
    if (l0 != -1) {
      const int* ngh0 = neigh + (i >> 3 << 6);
      const int* pi0 = parent + (i % 8) * 64;
      int* ngh1 = neigh_split + (l0 << 6);
      int t = -1;
      int k = ngh0[pi0[j]];
      if (k != -1) {
        int l1 = children[k];
        if (l1 != -1) {
          t = (l1 << 3) + dis[j];
        }
      }
      ngh1[j] = t;
    }
  }
}

__global__ void calc_full_neigh_kernel(int* neigh, int depth, int batch_size,
    int thread_num) {
  CUDA_1D_KERNEL_LOOP(id, thread_num) {
    const unsigned  bound = 1 << depth;
    unsigned node_num = 1 << 3 * depth;
    unsigned num = node_num >> 3;

    unsigned tm = id;
    unsigned z = tm % 4; tm /= 4;
    unsigned y = tm % 4; tm /= 4;
    unsigned x = tm % 4; tm /= 4;
    unsigned i = (tm % num) * 8;
    unsigned n = tm / num;

    unsigned x0 = 0, y0 = 0, z0 = 0;
#pragma unroll
    for (unsigned d = 0; d < depth; ++d) {
      x0 |= (i & (1 << 3 * d + 2)) >> (2 * d + 2);
      y0 |= (i & (1 << 3 * d + 1)) >> (2 * d + 1);
      z0 |= (i & (1 << 3 * d + 0)) >> (2 * d + 0);
    }

    unsigned x1 = x0 + x - 1;
    unsigned y1 = y0 + y - 1;
    unsigned z1 = z0 + z - 1;

    int v = -1;
    if ((x1 & bound) == 0 && (y1 & bound) == 0 && (z1 & bound) == 0) {
      unsigned key1 = 0;
#pragma unroll
      for (int d = 0; d < depth; ++d) {
        unsigned mask = 1u << d;
        key1 |= ((x1 & mask) << (2 * d + 2)) |
                ((y1 & mask) << (2 * d + 1)) |
                ((z1 & mask) << (2 * d));
      }
      v = key1 + n * node_num;
    }

    neigh[id] = v;
  }
}

void generate_key_gpu(OpKernelContext* ctx, int* key_split, const int* key,
    const int* children, int node_num) {
  // use the information of parent layer to calculate the neigh_split
  // of the current layer
  int n = node_num << 3; // the node number of parent layer
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  gen_key_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          key_split, key, children, n);
}

void generate_key_gpu(OpKernelContext* ctx, int* key, int depth,
    int batch_size) {
  int n = batch_size * (1 << 3 * depth);
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  gen_full_key_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          key, depth, batch_size, n);
}

void calc_neigh_gpu(OpKernelContext* ctx, int* neigh_split, const int* neigh,
    const int* children, const int* parent, const int* dis, int node_num) {
  int n = node_num << 6;
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  calc_neigh_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          neigh_split, neigh, children, parent, dis, n);
}

void calc_neigh_gpu(OpKernelContext* ctx, int* neigh, int depth,
    int batch_size) {
  int n = batch_size * (1 << (3 * depth + 3));
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  calc_full_neigh_kernel
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          neigh, depth, batch_size, n);
}

void tensorflow_gpu_gemm(OpKernelContext* ctx, bool transa, bool transb,
    uint64 m, uint64 n, uint64 k, float alpha, const float* a, const float* b,
    float beta, float* c) {
  perftools::gputools::blas::Transpose trans[] = {
      perftools::gputools::blas::Transpose::kNoTranspose,
      perftools::gputools::blas::Transpose::kTranspose };

  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  auto a_ptr = AsDeviceMemory(a);
  auto b_ptr = AsDeviceMemory(b);
  auto c_ptr = AsDeviceMemory(c);

  bool blas_launch_status =
    stream
    ->ThenBlasGemm(trans[transb], trans[transa], n, m, k, alpha, b_ptr,
      transb ? k : n, a_ptr, transa ? m : k, beta, &c_ptr, n)
    .ok();

  OP_REQUIRES(ctx, blas_launch_status, errors::Aborted("CuBlasGemm failed!"));
}

template <typename T>
void tensorflow_gpu_set_zero(OpKernelContext* ctx, T* Y, int N) {
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
  perftools::gputools::DeviceMemoryBase output_ptr(
      Y, N * sizeof(T));
  stream->ThenMemZero(&output_ptr, N * sizeof(T));
}

// Explicit instantiation
template void tensorflow_gpu_set_zero<float>(OpKernelContext* ctx, float* Y,
    int N);
template void tensorflow_gpu_set_zero<int>(OpKernelContext* ctx, int* Y,
    int N);

}  // namespace octree

}  // namespace tensorflow
