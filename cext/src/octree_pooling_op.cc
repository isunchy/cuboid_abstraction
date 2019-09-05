#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "cuda_runtime.h"

#include "octree.h"

namespace tensorflow {

namespace octree {

void max_pooling_forward(OpKernelContext* ctx, float* top_data, int top_h,
    int* bottom_mask, const float* bottom_data, int bottom_h, int nthreads);
void max_pooling_backward(OpKernelContext* ctx, float* bottom_diff,
    int bottom_h, const int* bottom_mask, const float* top_diff, int top_h,
    int nthreads);

} // namespace octree

REGISTER_OP("OctreePooling")
.Input("in_data: float")
.Input("in_octree: int32")
.Attr("curr_depth: int")
.Output("out_data: float")
.Output("out_mask: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  ::tensorflow::shape_inference::ShapeHandle output_shape = c->input(0);
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(output_shape, 2, c->UnknownDim(), &output_shape));
  c->set_output(0, output_shape);
  c->set_output(1, output_shape);
  return Status::OK();
})
.Doc(R"doc(
Octree pooling operator.
)doc");

class OctreePoolingOp : public OpKernel {
 public:
  explicit OctreePoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("curr_depth",
                                             &this->curr_depth_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    const Tensor& in_data = context->input(0);
    const TensorShape& in_data_shape = in_data.shape();
    auto in_data_ptr = in_data.flat<float>().data();

    // in octree
    const Tensor& in_octree = context->input(1);
    auto in_octree_ptr = in_octree.flat<int>().data();

    // parse octree info
    this->oct_batch_.set_gpu(in_octree_ptr);
    CHECK_EQ(oct_batch_.node_num(curr_depth_), in_data_shape.dim_size(2));

    // get top_buffer_ tensor
    TensorShape top_buffer_shape = in_data_shape;
    top_buffer_shape.set_dim(2, in_data_shape.dim_size(2) >> 3);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, top_buffer_shape,
                                &top_buffer_));
    auto top_buffer_ptr = top_buffer_.flat<float>().data();

    // out data
    Tensor* data_output_tensor = nullptr;
    TensorShape data_output_shape = in_data_shape;
    data_output_shape.set_dim(2, oct_batch_.node_num(curr_depth_ - 1));
    OP_REQUIRES_OK(context, context->allocate_output("out_data",
                                data_output_shape, &data_output_tensor));
    auto data_output_ptr = data_output_tensor->flat<float>().data();

    // out mask
    Tensor* mask_output_tensor = nullptr;
    TensorShape mask_output_shape = in_data_shape;
    mask_output_shape.set_dim(2, in_data_shape.dim_size(2) >> 3);
    OP_REQUIRES_OK(context, context->allocate_output("out_mask",
                                mask_output_shape, &mask_output_tensor));
    auto mask_output_ptr = mask_output_tensor->flat<int>().data();

    // fill in mask
    int channel = in_data_shape.dim_size(1);
    int bottom_h = in_data_shape.dim_size(2);
    int top_h = bottom_h / 8;
    int nthreads = top_h * channel;
    octree::max_pooling_forward(context, top_buffer_ptr, top_h,
        mask_output_ptr, in_data_ptr, bottom_h, nthreads);

    octree::pad_forward_gpu(context, data_output_ptr,
        data_output_shape.dim_size(2), data_output_shape.dim_size(1),
        top_buffer_ptr, top_buffer_shape.dim_size(2),
        oct_batch_.children_gpu(curr_depth_ - 1));
  }

 private:
  int curr_depth_;
  OctreeBatchParser oct_batch_;
  Tensor top_buffer_;
};
REGISTER_KERNEL_BUILDER(
    Name("OctreePooling").Device(DEVICE_GPU), OctreePoolingOp);


REGISTER_OP("OctreePoolingGrad")
.Input("gradients: float")
.Input("in_data: float")
.Input("in_mask: int32")
.Input("in_octree: int32")
.Attr("curr_depth: int")
.Output("grad_data: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  return Status::OK();
})
.Doc(R"doc(
Gradient for octree pooling operator.
)doc");

class OctreePoolingGradOP : public OpKernel {
 public:
  explicit OctreePoolingGradOP(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("curr_depth",
                                             &this->curr_depth_));
  }

  void Compute(OpKernelContext* context) override {
    // in gradients
    const Tensor& gradients = context->input(0);
    const TensorShape& gradients_shape = gradients.shape();
    auto gradients_ptr = gradients.flat<float>().data();

    // in data
    const Tensor& in_data = context->input(1);
    const TensorShape& in_data_shape = in_data.shape();

    // in mask
    const Tensor& in_mask = context->input(2);
    const TensorShape& in_mask_shape = in_mask.shape();
    auto in_mask_ptr = in_mask.flat<int>().data();
    CHECK_EQ(in_mask_shape.dim_size(2), in_data_shape.dim_size(2) >> 3);

    // in octree
    const Tensor& in_octree = context->input(3);
    auto in_octree_ptr = in_octree.flat<int>().data();

    // out grad data
    Tensor* grad_data_tensor = nullptr;
    TensorShape grad_data_shape = in_data_shape;
    OP_REQUIRES_OK(context, context->allocate_output("grad_data",
                                grad_data_shape, &grad_data_tensor));
    auto grad_data_ptr = grad_data_tensor->flat<float>().data();

    // parse octree info
    this->oct_batch_.set_gpu(in_octree_ptr);
    CHECK_EQ(oct_batch_.node_num(curr_depth_), in_data_shape.dim_size(2));

    // get top_buffer_ tensor
    TensorShape top_buffer_shape = in_data_shape;
    top_buffer_shape.set_dim(2, in_data_shape.dim_size(2) >> 3);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, top_buffer_shape,
                                &top_buffer_));
    auto top_buffer_ptr = top_buffer_.flat<float>().data();

    octree::pad_backward_gpu(context, top_buffer_ptr,
        top_buffer_shape.dim_size(2), top_buffer_shape.dim_size(1),
        gradients_ptr, gradients_shape.dim_size(2),
        oct_batch_.children_gpu(curr_depth_ - 1));

    int channel = grad_data_shape.dim_size(1);
    int bottom_h = grad_data_shape.dim_size(2);
    int top_h = bottom_h / 8;
    octree::tensorflow_gpu_set_zero(context, grad_data_ptr,
                                    grad_data_shape.num_elements());
    int nthreads = top_h * channel;
    octree::max_pooling_backward(context, grad_data_ptr, bottom_h, in_mask_ptr,
      top_buffer_ptr, top_h, nthreads);
  }

 private:
  int curr_depth_;
  OctreeBatchParser oct_batch_;
  Tensor top_buffer_;
};
REGISTER_KERNEL_BUILDER(
    Name("OctreePoolingGrad").Device(DEVICE_GPU), OctreePoolingGradOP);

}  // namespace tensorflow
