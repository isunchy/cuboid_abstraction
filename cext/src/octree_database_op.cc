#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/variable.pb.h"

#include "octree.h"

namespace tensorflow {

REGISTER_OP("OctreeDatabase")
.Input("in_batch_data: string")
.Attr("segmentation: bool = false")
.Attr("content_flags: int = 7")
.Output("out_data: float")
.Output("out_octree: int32")
.Output("out_seglabel: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({ 1, 3, c->UnknownDim(), 1 }));
  c->set_output(1, c->UnknownShapeOfRank(1));
  c->set_output(2, c->UnknownShapeOfRank(1));
  return Status::OK();
})
.Doc(R"doc(
Decode octree batch info.
)doc");

class OctreeDatabaseOp : public OpKernel {
 public:
  explicit OctreeDatabaseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("segmentation",
                                             &this->segmentation_));
    OP_REQUIRES_OK(context, context->GetAttr("content_flags",
                                             &this->content_flags_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& in_data = context->input(0);

    // parse octree batch
    OctreeBatch oct_batch;
    auto dc = OctreeBatch::CLASSIFICATION;
    if (segmentation_) dc = OctreeBatch::SEGMENTATION;
    oct_batch.set_octreebatch(context, in_data, content_flags_, dc);

    // out data
    Tensor* data_output_tensor = nullptr;
    TensorShape data_output_shape = oct_batch.data_.shape();
    OP_REQUIRES_OK(context, context->allocate_output("out_data",
                                data_output_shape, &data_output_tensor));
    CHECK((*data_output_tensor).CopyFrom(oct_batch.data_, data_output_shape));

    // out octree
    Tensor* octree_output_tensor = nullptr;
    TensorShape octree_output_shape = oct_batch.octree_.shape();
    OP_REQUIRES_OK(context, context->allocate_output("out_octree",
                                octree_output_shape, &octree_output_tensor));
    CHECK((*octree_output_tensor).CopyFrom(oct_batch.octree_,
                                           octree_output_shape));

    // out label
    Tensor* seglabel_output_tensor = nullptr;
    TensorShape seglabel_output_shape = oct_batch.label_.shape();
    OP_REQUIRES_OK(context, context->allocate_output("out_seglabel",
                                seglabel_output_shape,
                                &seglabel_output_tensor));
    if (dc = OctreeBatch::SEGMENTATION) {
      CHECK((*seglabel_output_tensor).CopyFrom(oct_batch.label_,
                                               seglabel_output_shape));
    }
  }

 private:
  int content_flags_;
  bool segmentation_;
};
REGISTER_KERNEL_BUILDER(
    Name("OctreeDatabase").Device(DEVICE_CPU), OctreeDatabaseOp);

}  // namespace tensorflow
