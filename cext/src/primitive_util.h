#ifndef TENSORFLOW_USER_OPS_PRIMITIVE_UTIL_H_
#define TENSORFLOW_USER_OPS_PRIMITIVE_UTIL_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace primitive {

template <typename T>
void gpu_set_zero(OpKernelContext* ctx, T* Y, const int N);

}  // namespace primitive

}  // namespace tensorflow

#endif  // !TENSORFLOW_USER_OPS_PRIMITIVE_UTIL_H_
