#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void group_points(OpKernelContext* context, const int n_point, const int n_cube,
    const float* in_z, const float* in_q, const float* in_t,
    const float* in_pos, int* index);

REGISTER_OP("PrimitiveGroupPoints")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Input("in_pos: float")
.Output("out_index: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({c->Dim(c->input(3), 1)}));
  return Status::OK();
})
.Doc(R"doc(
Group points by the nearest cube.
)doc");

class PrimitiveGroupPointsOp : public OpKernel {
 public:
  explicit PrimitiveGroupPointsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in_z [bs, n_cube * 3]
    const Tensor& in_z = context->input(0);
    auto in_z_ptr = in_z.flat<float>().data();
    batch_size_ = in_z.dim_size(0);
    n_cube_ = in_z.dim_size(1) / 3;

    // in_q [bs, n_cube * 4]
    const Tensor& in_q = context->input(1);
    auto in_q_ptr = in_q.flat<float>().data();
    CHECK_EQ(in_q.dim_size(0), batch_size_);
    CHECK_EQ(in_q.dim_size(1), n_cube_ * 4);

    // in_t [bs, n_cube * 3]
    const Tensor& in_t = context->input(2);
    auto in_t_ptr = in_t.flat<float>().data();
    CHECK_EQ(in_t.dim_size(0), batch_size_);
    CHECK_EQ(in_t.dim_size(1), n_cube_ * 3);

    // in_pos [4, n_point]
    const Tensor& in_pos = context->input(3);
    auto in_pos_ptr = in_pos.flat<float>().data();
    CHECK_EQ(in_pos.dim_size(0), 4);
    n_point_ = in_pos.dim_size(1);

    // out index
    /// point group index is accumulated with batch size
    /// [0, 1, ..., n_cube - 1, n_cube, n_cube + 1, ..., 2*n_cube - 1, ...,
    ///  (batch_size - 1)*n_cube, ..., batch_size*n_cube - 1]
    Tensor* index_output_tensor = nullptr;
    TensorShape index_output_shape({n_point_});
    OP_REQUIRES_OK(context, context->allocate_output("out_index",
                                index_output_shape, &index_output_tensor));
    auto index_output_ptr = index_output_tensor->flat<int>().data();

    // split points to group
    group_points(context, n_point_, n_cube_, in_z_ptr, in_q_ptr, in_t_ptr,
        in_pos_ptr, index_output_ptr);
  }

 private:
  int n_cube_;
  int n_point_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveGroupPoints").Device(DEVICE_GPU),
    PrimitiveGroupPointsOp);

}  // namespace tensorflow
