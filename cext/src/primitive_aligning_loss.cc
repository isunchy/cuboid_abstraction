#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void compute_aligning_loss(OpKernelContext* context, const int n_cube,
    const int batch_size, const float* in_q, const float* in_dir,
    float* loss_ptr);

void compute_aligning_loss_grad(OpKernelContext* context, const int n_cube,
    const int batch_size, const float* loss, const float* in_q,
    const float* in_dir, float* grad_q);

REGISTER_OP("PrimitiveAligningLoss")
.Input("in_q: float")
.Input("in_dir: float")
.Output("out_loss: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({1}));
  return Status::OK();
})
.Doc(R"doc(
Compute the consine distance between the input direction and the rotated
direction according to the input quaternion. The input direction should be a
unit direction.
)doc");

class PrimitiveAligningLossOp : public OpKernel {
 public:
  explicit PrimitiveAligningLossOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in_q [bs, n_cube * 4]
    const Tensor& in_q = context->input(0);
    auto in_q_ptr = in_q.flat<float>().data();
    batch_size_ = in_q.dim_size(0);
    n_cube_ = in_q.dim_size(1) / 4;

    // in_dir [3]
    const Tensor& in_dir = context->input(1);
    auto in_dir_ptr = in_dir.flat<float>().data();
    CHECK_EQ(in_dir.NumElements(), 3);

    // out loss
    Tensor* out_loss = nullptr;
    TensorShape out_loss_shape({1});
    OP_REQUIRES_OK(context, context->allocate_output("out_loss",
                                out_loss_shape, &out_loss));
    auto out_loss_ptr = out_loss->flat<float>().data();

    // compute aligning loss
    compute_aligning_loss(context, n_cube_, batch_size_, in_q_ptr,
        in_dir_ptr, out_loss_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveAligningLoss").Device(DEVICE_GPU),
    PrimitiveAligningLossOp);

REGISTER_OP("PrimitiveAligningLossGrad")
.Input("gradient: float")
.Input("in_q: float")
.Input("in_dir: float")
.Output("grad_q: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  return Status::OK();
})
.Doc(R"doc(
Gradient for the primitive aligning loss;
)doc");

class PrimitiveAligningLossGradOp : public OpKernel {
 public:
  explicit PrimitiveAligningLossGradOp(OpKernelConstruction* context)
    : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in gradients
    const Tensor& gradients = context->input(0);
    auto gradients_ptr = gradients.flat<float>().data();

    // in_q [n_cube * 4, 1]
    const Tensor& in_q = context->input(1);
    auto in_q_ptr = in_q.flat<float>().data();
    batch_size_ = in_q.dim_size(0);
    n_cube_ = in_q.dim_size(1) / 4;

    // in_dir [3]
    const Tensor& in_dir = context->input(2);
    auto in_dir_ptr = in_dir.flat<float>().data();
    CHECK_EQ(in_dir.NumElements(), 3);

    // grad_q
    Tensor* grad_q = nullptr;
    TensorShape grad_q_shape = in_q.shape();
    OP_REQUIRES_OK(context, context->allocate_output("grad_q",
                                grad_q_shape, &grad_q));
    auto grad_q_ptr = grad_q->flat<float>().data();

    // compute aligning loss gradient
    compute_aligning_loss_grad(context, n_cube_, batch_size_, gradients_ptr,
        in_q_ptr, in_dir_ptr, grad_q_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveAligningLossGrad").Device(DEVICE_GPU),
    PrimitiveAligningLossGradOp);

}  // namespace tensorflow
