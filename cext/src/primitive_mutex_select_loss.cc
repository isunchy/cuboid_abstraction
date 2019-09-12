#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void compute_mutex_select_loss(OpKernelContext* context, const int n_cube,
    const int batch_size, const float scale, const float* in_z,
    const float* in_q, const float* in_t, const int* in_mask, float* loss_ptr);

void compute_mutex_select_loss_grad(OpKernelContext* context, const int n_cube,
    const int batch_size, const float scale, const float* loss,
    const float* in_z, const float* in_q, const float* in_t, const int* in_mask,
    float* grad_z, float* grad_q, float* grad_t);

REGISTER_OP("PrimitiveMutexSelectLoss")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Input("in_mask: int32")
.Attr("scale: float = 0.9")
.Output("out_loss: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({1}));
  return Status::OK();
})
.Doc(R"doc(
Among the selected cube set, sample points in cube volume, and compute the
distance of each point invades the other cubes.
)doc");

class PrimitiveMutexSelectLossOp : public OpKernel {
 public:
  explicit PrimitiveMutexSelectLossOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

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

    // in_mask [bs, n_cube]
    const Tensor& in_mask = context->input(3);
    auto in_mask_ptr = in_mask.flat<int>().data();
    CHECK_EQ(in_mask.dim_size(0), batch_size_);
    CHECK_EQ(in_mask.dim_size(1), n_cube_);

    // out loss
    Tensor* out_loss = nullptr;
    TensorShape out_loss_shape({1});
    OP_REQUIRES_OK(context, context->allocate_output("out_loss",
                                out_loss_shape, &out_loss));
    auto out_loss_ptr = out_loss->flat<float>().data();
    
    // compute mutex loss
    compute_mutex_select_loss(context, n_cube_, batch_size_, scale_, in_z_ptr,
        in_q_ptr, in_t_ptr, in_mask_ptr, out_loss_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
  float scale_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveMutexSelectLoss").Device(DEVICE_GPU),
    PrimitiveMutexSelectLossOp);


REGISTER_OP("PrimitiveMutexSelectLossGrad")
.Input("gradient: float")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Input("in_mask: int32")
.Attr("scale: float")
.Output("grad_z: float")
.Output("grad_q: float")
.Output("grad_t: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  c->set_output(1, c->input(2));
  c->set_output(2, c->input(3));
  return Status::OK();
})
.Doc(R"doc(
Gradient for primitive mutes loss.
)doc");

class PrimitiveMutexSelectLossGradOp : public OpKernel {
 public:
  explicit PrimitiveMutexSelectLossGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(OpKernelContext* context) override {
    // in gradients
    const Tensor& gradients = context->input(0);
    auto gradients_ptr = gradients.flat<float>().data();

    // in_z [bs, n_cube * 3]
    const Tensor& in_z = context->input(1);
    auto in_z_ptr = in_z.flat<float>().data();
    batch_size_ = in_z.dim_size(0);
    n_cube_ = in_z.dim_size(1) / 3;

    // in_q [bs, n_cube * 4]
    const Tensor& in_q = context->input(2);
    auto in_q_ptr = in_q.flat<float>().data();
    CHECK_EQ(in_q.dim_size(0), batch_size_);
    CHECK_EQ(in_q.dim_size(1), n_cube_ * 4);

    // in_t [bs, n_cube * 3]
    const Tensor& in_t = context->input(3);
    auto in_t_ptr = in_t.flat<float>().data();
    CHECK_EQ(in_t.dim_size(0), batch_size_);
    CHECK_EQ(in_t.dim_size(1), n_cube_ * 3);

    // in_mask [bs, n_cube]
    const Tensor& in_mask = context->input(4);
    auto in_mask_ptr = in_mask.flat<int>().data();
    CHECK_EQ(in_mask.dim_size(0), batch_size_);
    CHECK_EQ(in_mask.dim_size(1), n_cube_);

    // grad_z
    Tensor* grad_z = nullptr;
    TensorShape grad_z_shape = in_z.shape();
    OP_REQUIRES_OK(context, context->allocate_output("grad_z",
                                grad_z_shape, &grad_z));
    auto grad_z_ptr = grad_z->flat<float>().data();

    // grad_q
    Tensor* grad_q = nullptr;
    TensorShape grad_q_shape = in_q.shape();
    OP_REQUIRES_OK(context, context->allocate_output("grad_q",
                                grad_q_shape, &grad_q));
    auto grad_q_ptr = grad_q->flat<float>().data();

    // grad_t
    Tensor* grad_t = nullptr;
    TensorShape grad_t_shape = in_t.shape();
    OP_REQUIRES_OK(context, context->allocate_output("grad_t",
                                grad_t_shape, &grad_t));
    auto grad_t_ptr = grad_t->flat<float>().data();

    // compute mutex loss gradient
    compute_mutex_select_loss_grad(context, n_cube_, batch_size_, scale_,
        gradients_ptr, in_z_ptr, in_q_ptr, in_t_ptr, in_mask_ptr, grad_z_ptr,
        grad_q_ptr, grad_t_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
  float scale_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveMutexSelectLossGrad").Device(DEVICE_GPU),
    PrimitiveMutexSelectLossGradOp);

}  //namespace tensorflow
