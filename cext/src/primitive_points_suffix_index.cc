#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void suffix_index_to_points(OpKernelContext* context, const int batch_size,
    const int n_point, const float* in_points, float* out_pos);


REGISTER_OP("PrimitivePointsSuffixIndex")
.Input("in_pos: float")
.Output("out_pos: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({4, c->UnknownDim()}));
  return Status::OK();
})
.Doc(R"doc(
Add batch inner index to each points as the 4th dimension.
)doc");

class PrimitivePointsSuffixIndexOp : public OpKernel {
 public:
  explicit PrimitivePointsSuffixIndexOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in points [batch_size, 3*n_point]
    const Tensor& in_points = context->input(0);
    auto in_points_ptr = in_points.flat<float>().data();
    batch_size_ = in_points.shape().dim_size(0);
    n_point_ = in_points.shape().dim_size(1) / 3;

    // out position
    Tensor* pos_output_tensor = nullptr;
    TensorShape pos_output_shape({4, batch_size_ * n_point_});
    OP_REQUIRES_OK(context, context->allocate_output("out_pos",
                                pos_output_shape, &pos_output_tensor));
    auto pos_output_ptr = pos_output_tensor->flat<float>().data();

    // add index to position
    suffix_index_to_points(context, batch_size_, n_point_, in_points_ptr,
        pos_output_ptr);
  }

 private:
  int batch_size_;
  int n_point_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitivePointsSuffixIndex").Device(DEVICE_GPU),
    PrimitivePointsSuffixIndexOp);

}  // namespace tensorflow
