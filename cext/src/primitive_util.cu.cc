#define EIGEN_USE_THREADS

#include "primitive_util.h"

#include "cuda.h"
#include "device_launch_parameters.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

namespace primitive {

template <typename T>
void gpu_set_zero(OpKernelContext* ctx, T* Y, int N) {
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
  perftools::gputools::DeviceMemoryBase output_ptr(Y, N * sizeof(T));
  stream->ThenMemZero(&output_ptr, N * sizeof(T));
}

// Explicit instantiation
template void gpu_set_zero<float>(OpKernelContext* ctx, float* Y, int N);
template void gpu_set_zero<int>(OpKernelContext* ctx, int* Y, int N);

}  // namespace primitive

}  // namespace tensorflow
