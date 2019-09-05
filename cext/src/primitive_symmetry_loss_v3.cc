#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void compute_symmetry_loss_v3(OpKernelContext* context, const int n_cube,
    const int batch_size, const int depth, const float scale,
    const float* in_z, const float* in_q, const float* in_t, float* loss_ptr);

void compute_symmetry_loss_v3_grad(OpKernelContext* context, const int n_cube,
    const int batch_size, const int depth, const float scale,
    const float* loss, const float* in_z, const float* in_q, const float* in_t,
    float* grad_z, float* grad_q, float* grad_t);

REGISTER_OP("PrimitiveSymmetryLossV3")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Attr("scale: float = 0.9")
.Attr("depth: int = 5")
.Output("out_loss: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({1}));
  return Status::OK();
})
.Doc(R"doc(
Sample points in cube volume, flip them along symmetry plane. The group of 
point cloud sampled on one cube should be covered by one cube (maybe itself).
)doc");

class PrimitiveSymmetryLossV3Op : public OpKernel {
 public:
  explicit PrimitiveSymmetryLossV3Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
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

    // out loss
    Tensor* out_loss = nullptr;
    TensorShape out_loss_shape({1});
    OP_REQUIRES_OK(context, context->allocate_output("out_loss",
                                out_loss_shape, &out_loss));
    auto out_loss_ptr = out_loss->flat<float>().data();
  
    // compute symmetry loss
    compute_symmetry_loss_v3(context, n_cube_, batch_size_, depth_, scale_,
        in_z_ptr, in_q_ptr, in_t_ptr, out_loss_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
  int depth_;  // octree node depth, for computing symmetry plane location
  float scale_;  // scale of sampled points inside cube
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveSymmetryLossV3").Device(DEVICE_GPU),
    PrimitiveSymmetryLossV3Op);


REGISTER_OP("PrimitiveSymmetryLossV3Grad")
.Input("gradient: float")
.Input("in_z: float")
.Input("in_q: float")
.Input("in_t: float")
.Attr("scale: float")
.Attr("depth: int")
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
Gradient for primitive symmetry loss;
)doc");

class PrimitiveSymmetryLossV3GradOp : public OpKernel {
 public:
  explicit PrimitiveSymmetryLossV3GradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("depth", &depth_));
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

    // compute symmetry loss gradient
    compute_symmetry_loss_v3_grad(context, n_cube_, batch_size_, depth_,
        scale_, gradients_ptr, in_z_ptr, in_q_ptr, in_t_ptr, grad_z_ptr,
        grad_q_ptr, grad_t_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
  int depth_;
  float scale_;  // scale of sampled points inside cube
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveSymmetryLossV3Grad").Device(DEVICE_GPU),
    PrimitiveSymmetryLossV3GradOp);

}  // namespace tensorflow
