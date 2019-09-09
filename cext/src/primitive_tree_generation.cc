#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void correct_tree_mask(OpKernelContext* context, const int batch_size,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_1, const int* in_relation_2, const int* in_mask,
    int* mask);

void lift_up_tree_mask(OpKernelContext* context, const int batch_size,
    const int n_part_1, const int n_part_2, const int n_part_3,
    const int* in_relation_1, const int* in_relation_2, const int* tree_mask_1,
    int* tree_mask_2, int* tree_mask_3);

REGISTER_OP("PrimitiveTreeGeneration")
.Input("in_mask: int32")
.Input("in_relation_1: int32")
.Input("in_relation_2: int32")
.Attr("n_part_1: int")
.Attr("n_part_2: int")
.Attr("n_part_3: int")
.Output("tree_mask_1: int32")
.Output("tree_mask_2: int32")
.Output("tree_mask_3: int32")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  c->set_output(1, c->input(0));
  c->set_output(2, c->input(0));
  return Status::OK();
})
.Doc(R"doc(
Correct the input tree mask w.r.t. tree completeness.
Output three level selected trees.
)doc");


class PrimitiveTreeGenerationOp : public OpKernel {
 public:
  explicit PrimitiveTreeGenerationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("n_part_1", &n_part_1_));
    OP_REQUIRES_OK(context, context->GetAttr("n_part_2", &n_part_2_));
    OP_REQUIRES_OK(context, context->GetAttr("n_part_3", &n_part_3_));
    n_part_sum_ = n_part_1_ + n_part_2_ + n_part_3_;
  }

  void Compute(OpKernelContext* context) override {
    // in mask [bs, n1+n2+n3]
    const Tensor& in_mask = context->input(0);
    auto in_mask_ptr = in_mask.flat<int>().data();
    const TensorShape tree_mask_shape = in_mask.shape();
    batch_size_ = in_mask.dim_size(0);
    CHECK_EQ(in_mask.dim_size(1), n_part_sum_);

    // in_relation_1 [bs, n1]
    const Tensor& in_relation_1 = context->input(1);
    auto in_relation_1_ptr = in_relation_1.flat<int>().data();
    CHECK_EQ(in_relation_1.dim_size(0), batch_size_);
    CHECK_EQ(in_relation_1.dim_size(1), n_part_1_);

    // in_relation_2 [bs, n2]
    const Tensor& in_relation_2 = context->input(2);
    auto in_relation_2_ptr = in_relation_2.flat<int>().data();
    CHECK_EQ(in_relation_2.dim_size(0), batch_size_);
    CHECK_EQ(in_relation_2.dim_size(1), n_part_2_);

    // out tree_mask [bs, n1+n2+n3]
    Tensor* out_tree_mask_1 = nullptr;
    Tensor* out_tree_mask_2 = nullptr;
    Tensor* out_tree_mask_3 = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("tree_mask_1",
                                tree_mask_shape, &out_tree_mask_1));
    OP_REQUIRES_OK(context, context->allocate_output("tree_mask_2",
                                tree_mask_shape, &out_tree_mask_2));
    OP_REQUIRES_OK(context, context->allocate_output("tree_mask_3",
                                tree_mask_shape, &out_tree_mask_3));
    auto out_tree_mask_1_ptr = out_tree_mask_1->flat<int>().data();
    auto out_tree_mask_2_ptr = out_tree_mask_2->flat<int>().data();
    auto out_tree_mask_3_ptr = out_tree_mask_3->flat<int>().data();

    // correct input tree mask, get the tree_mask_1
    correct_tree_mask(context, batch_size_, n_part_1_, n_part_2_, n_part_3_,
        in_relation_1_ptr, in_relation_2_ptr, in_mask_ptr, out_tree_mask_1_ptr);

    // lift up tree mask to get tree_mask_2 and tree_mask_3
    lift_up_tree_mask(context, batch_size_, n_part_1_, n_part_2_, n_part_3_,
        in_relation_1_ptr, in_relation_2_ptr, out_tree_mask_1_ptr,
        out_tree_mask_2_ptr, out_tree_mask_3_ptr);
  }

 private:
  int n_part_1_; // 16
  int n_part_2_; // 8
  int n_part_3_; // 4
  int n_part_sum_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveTreeGeneration").Device(DEVICE_GPU),
    PrimitiveTreeGenerationOp);

}  // namespace tensorflow
