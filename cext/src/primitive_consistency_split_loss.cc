#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void compute_consistency_split_loss(OpKernelContext* context, const int n_cube,
    const int n_point, const int batch_size, const float num_sample,
    const float scale, const float* in_z, const float* in_q, const float* in_t,
    const float* in_pos, float* loss_ptr);

void compute_consistency_split_loss_grad(OpKernelContext* context,
    const int n_cube, const int n_point, const int batch_size,
    const float num_sample, const float scale, const float* loss,
    const float* in_z, const float* in_q, const float* in_t,
    const float* in_pos, float* grad_z, float* grad_q, float* grad_t);

REGISTER_OP("PrimitiveConsistencySplitLoss")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Input("in_pos: float")
.Attr("scale: float = 0.9")
.Attr("num_sample: int = 26")
.Output("out_loss: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0), c->UnknownDim()}));
  return Status::OK();
})
.Doc(R"doc(
Compute the distance of point sampled on cube with their nearest point in point
cloud.
)doc");

class PrimitiveConsistencySplitLossOp : public OpKernel {
 public:
  explicit PrimitiveConsistencySplitLossOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("num_sample", &num_sample_));
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

    // in_pos [4, n_point]
    const Tensor& in_pos = context->input(3);
    auto in_pos_ptr = in_pos.flat<float>().data();
    CHECK_EQ(in_pos.dim_size(0), 4);
    n_point_ = in_pos.dim_size(1);

    // out split loss [bs, n_cube]
    Tensor* out_loss = nullptr;
    TensorShape out_loss_shape({batch_size_, n_cube_});
    OP_REQUIRES_OK(context, context->allocate_output("out_loss",
                                out_loss_shape, &out_loss));
    auto out_loss_ptr = out_loss->flat<float>().data();

    CHECK(num_sample_ == 8 || num_sample_ == 26 || num_sample_ == 96);

    // compute consistency loss
    compute_consistency_split_loss(context, n_cube_, n_point_, batch_size_,
        num_sample_, scale_, in_z_ptr, in_q_ptr, in_t_ptr, in_pos_ptr,
        out_loss_ptr);
  }

 private:
  int n_cube_;
  int n_point_;
  int batch_size_;
  int num_sample_;
  float scale_;  // scale of sampled points inside cube
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveConsistencySplitLoss").Device(DEVICE_GPU),
    PrimitiveConsistencySplitLossOp);


REGISTER_OP("PrimitiveConsistencySplitLossGrad")
.Input("gradient: float")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Input("in_pos: float")
.Attr("scale: float")
.Attr("num_sample: int")
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
Gradient for primitive consistency loss;
)doc");

class PrimitiveConsistencySplitLossGradOp : public OpKernel {
 public:
  explicit PrimitiveConsistencySplitLossGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("num_sample", &num_sample_));
  }

  void Compute(OpKernelContext* context) override {
    // in gradients [bs, n_cube]
    const Tensor& gradients = context->input(0);
    auto gradients_ptr = gradients.flat<float>().data();
    batch_size_ = gradients.dim_size(0);
    n_cube_ = gradients.dim_size(1);

    // in_z [bs, n_cube * 3]
    const Tensor& in_z = context->input(1);
    auto in_z_ptr = in_z.flat<float>().data();
    CHECK_EQ(in_z.dim_size(0), batch_size_);
    CHECK_EQ(in_z.dim_size(1), n_cube_ * 3);

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

    // in_pos [4, n_point]
    const Tensor& in_pos = context->input(4);
    auto in_pos_ptr = in_pos.flat<float>().data();
    CHECK_EQ(in_pos.dim_size(0), 4);
    n_point_ = in_pos.dim_size(1);

    CHECK(num_sample_ == 8 || num_sample_ == 26 || num_sample_ == 96);

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

    // compute consistency loss gradient
    compute_consistency_split_loss_grad(context, n_cube_, n_point_, batch_size_,
        num_sample_, scale_, gradients_ptr, in_z_ptr, in_q_ptr, in_t_ptr,
        in_pos_ptr, grad_z_ptr, grad_q_ptr, grad_t_ptr);    
  }

 private:
  int n_cube_;
  int n_point_;
  int batch_size_;
  int num_sample_;
  float scale_;  // scale of sampled points inside cube
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveConsistencySplitLossGrad").Device(DEVICE_GPU),
    PrimitiveConsistencySplitLossGradOp);

}  // namespace tensorflow
