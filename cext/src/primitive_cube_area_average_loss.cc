#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void compute_cube_area_average_loss(OpKernelContext* context,
    const int n_cube, const int batch_size, const float* in_z,
    float* loss_ptr);

void compute_cube_area_average_loss_grad(OpKernelContext* context,
    const int n_cube, const int batch_size, const float* loss,
    const float* in_z, float* grad_z);

REGISTER_OP("PrimitiveCubeAreaAverageLoss")
.Input("in_z: float")
.Output("out_loss: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({1}));
  return Status::OK();
})
.Doc(R"doc(
Get the all cubes surface area. Compute the total difference of each cube'
surface area with the mean area.
)doc");

class PrimitiveCubeAreaAverageLossOp : public OpKernel {
 public:
  explicit PrimitiveCubeAreaAverageLossOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in_z [bs, n_cube * 3]
    const Tensor& in_z = context->input(0);
    auto in_z_ptr = in_z.flat<float>().data();
    batch_size_ = in_z.dim_size(0);
    n_cube_ = in_z.dim_size(1) / 3;

    // out area
    Tensor* out_loss = nullptr;
    TensorShape out_loss_shape({1});
    OP_REQUIRES_OK(context, context->allocate_output("out_loss",
                                out_loss_shape, &out_loss));
    auto out_loss_ptr = out_loss->flat<float>().data();

    // compute cube area
    compute_cube_area_average_loss(context, n_cube_, batch_size_, in_z_ptr,
        out_loss_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(
    Name("PrimitiveCubeAreaAverageLoss").Device(DEVICE_GPU),
    PrimitiveCubeAreaAverageLossOp);


REGISTER_OP("PrimitiveCubeAreaAverageLossGrad")
.Input("gradient: float")
.Input("in_z: float")
.Output("grad_z: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  return Status::OK();
})
.Doc(R"doc(
Gradient for cube area average loss.
)doc");

class PrimitiveCubeAreaAverageLossGradOp : public OpKernel {
 public:
  explicit PrimitiveCubeAreaAverageLossGradOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in gradients
    const Tensor& gradients = context->input(0);
    auto gradients_ptr = gradients.flat<float>().data();

    // in_z [bs, n_cube * 3]
    const Tensor& in_z = context->input(1);
    auto in_z_ptr = in_z.flat<float>().data();
    batch_size_ = in_z.dim_size(0);
    n_cube_ = in_z.dim_size(1) / 3;

    // grad_z
    Tensor* grad_z = nullptr;
    TensorShape grad_z_shape = in_z.shape();
    OP_REQUIRES_OK(context, context->allocate_output("grad_z",
                                grad_z_shape, &grad_z));
    auto grad_z_ptr = grad_z->flat<float>().data();

    // compute coverage loss gradient
    compute_cube_area_average_loss_grad(context, n_cube_, batch_size_,
        gradients_ptr, in_z_ptr, grad_z_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(
    Name("PrimitiveCubeAreaAverageLossGrad").Device(DEVICE_GPU),
    PrimitiveCubeAreaAverageLossGradOp);

}  // namespace tensorflow
