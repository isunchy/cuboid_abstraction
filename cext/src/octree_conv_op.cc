#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "cuda_runtime.h"

#include "octree.h"

namespace tensorflow {

void init_neigh_index(OpKernelContext* ctx, Tensor& ni);

REGISTER_OP("OctreeConv")
.Input("in_data: float")
.Input("in_filter: float")
.Input("in_octree: int32")
.Attr("curr_depth: int")
.Attr("num_output: int")
.Attr("kernel_size: int")
.Attr("stride: int")
.Output("out_data: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  int num_output;
  TF_RETURN_IF_ERROR(c->GetAttr("num_output", &num_output));
  c->set_output(0, c->MakeShape({ 1, num_output, c->UnknownDim(), 1 }));
  return Status::OK();
})
.Doc(R"doc(
Octree convolution operator.
)doc");

class OctreeConvOP : public OpKernel {
 public:
  explicit OctreeConvOP(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("curr_depth",
                                             &this->curr_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("num_output",
                                             &this->num_output_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size",
                                             &this->kernel_size_));
    OP_REQUIRES_OK(context, context->GetAttr("stride", &this->stride_));
  }

  void Compute(OpKernelContext* context) override {
    // in data
    // data format: [1, channels, H, 1]
    const Tensor& in_data = context->input(0);
    const TensorShape& in_data_shape = in_data.shape();
    auto in_data_ptr = in_data.flat<float>().data();
    channels_ = in_data_shape.dim_size(1);

    // in filter
    // filter format: [out_channels, in_channels, kernel_size ** 3]
    const Tensor& in_filter = context->input(1);
    const TensorShape& in_filter_shape = in_filter.shape();
    auto in_filter_ptr = in_filter.flat<float>().data();
    CHECK_EQ(in_filter_shape.dim_size(0), num_output_);
    CHECK_EQ(in_filter_shape.dim_size(1), channels_);
    CHECK_EQ(in_filter_shape.dim_size(2),
             kernel_size_ * kernel_size_ * kernel_size_);

    // in octree
    const Tensor& in_octree = context->input(2);
    auto in_octree_ptr = in_octree.flat<int>().data();

    // parse octree info
    oct_batch_.set_gpu(in_octree_ptr);
    CHECK_EQ(oct_batch_.node_num(curr_depth_), in_data_shape.dim_size(2));

    // get workspace tensor
    kernel_dim_ = in_filter_shape.dim_size(1) * in_filter_shape.dim_size(2);
    workspace_h_ = in_data_shape.dim_size(2);
    workspace_ha_ = workspace_h_;
    workspace_n_ = 1;
    workspace_depth_ = curr_depth_;
    const int MAX_SIZE = octree::get_workspace_maxsize();
    int ideal_size = workspace_h_ * kernel_dim_;
    if (ideal_size > MAX_SIZE) {
      workspace_n_ = (ideal_size + MAX_SIZE - 1) / MAX_SIZE;
      workspace_ha_ = (workspace_h_ + workspace_n_ - 1) / workspace_n_;
    }
    const TensorShape workspace_shape({ kernel_dim_, workspace_ha_ });
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, workspace_shape,
                                &workspace_));
    auto workspace_ptr = workspace_.flat<float>().data();

    // get result_buffer tensor if needed
    if (workspace_n_ > 1) {
      const TensorShape result_shape({ num_output_, workspace_ha_ });
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, result_shape,
                                  &result_buffer_));
    }
    auto result_buffer_ptr = result_buffer_.flat<float>().data();

    // init neighbor info
    init_neigh_index(context, ni_);
    auto ni_ptr = ni_.flat<int32>().data();

    // out data
    Tensor* data_output_tensor = nullptr;
    TensorShape data_output_shape({ 1, num_output_, workspace_h_, 1 });
    OP_REQUIRES_OK(context, context->allocate_output("out_data",
                                data_output_shape, &data_output_tensor));
    auto data_output_ptr = data_output_tensor->flat<float>().data();

    float* result_data =
        workspace_n_ == 1 ? data_output_ptr : result_buffer_ptr;
    for (int n = 0; n < workspace_n_; ++n) {
      // set workspace data
      octree::octree2col_gpu(context, workspace_ptr, in_data_ptr, channels_,
          workspace_h_, kernel_size_, stride_,
          oct_batch_.neighbor_gpu(workspace_depth_), ni_ptr, workspace_ha_, n);
      // gemm
      octree::tensorflow_gpu_gemm(context, false, false,
          static_cast<uint64>(num_output_), static_cast<uint64>(workspace_ha_),
          static_cast<uint64>(kernel_dim_), static_cast<float>(1),
          in_filter_ptr, workspace_ptr, static_cast<float>(0), result_data);

      if (workspace_n_ == 1) break;
      int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
      for (int c = 0; c < num_output_; ++c) {
        cudaMemcpy(data_output_ptr + c * workspace_h_ + n * workspace_ha_,
            result_data + c * workspace_ha_, num * sizeof(float),
            cudaMemcpyDeviceToDevice);
      }
    }
  }

 private:
  int kernel_size_;
  int kernel_dim_;
  int stride_;

  int channels_;
  int num_output_;

  int curr_depth_;
  OctreeBatchParser oct_batch_;

  /////////////////////////////////////////////////////////////////////////////
  //  workspace shape:
  //              h
  //         ------------
  //         |          |
  // c * k^3 |          |
  //         |          |
  //         ------------
  /////////////////////////////////////////////////////////////////////////////
  int workspace_n_;   // times of workspace usage
  int workspace_ha_;  // actual workspace h
  int workspace_h_;   // ideal workspace h
  int workspace_depth_;
  Tensor workspace_;
  Tensor result_buffer_;

  Tensor ni_;
};
REGISTER_KERNEL_BUILDER(Name("OctreeConv").Device(DEVICE_GPU), OctreeConvOP);


REGISTER_OP("OctreeConvGrad")
.Input("gradients: float")
.Input("in_data: float")
.Input("in_filter: float")
.Input("in_octree: int32")
.Attr("curr_depth: int")
.Attr("num_output: int")
.Attr("kernel_size: int")
.Attr("stride: int")
.Output("grad_data: float")
.Output("grad_filter: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  c->set_output(1, c->input(2));
  return Status::OK();
})
.Doc(R"doc(
Gradient for octree convolution operator.
)doc");

class OctreeConvGradOP : public OpKernel {
 public:
  explicit OctreeConvGradOP(OpKernelConstruction* context):OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("curr_depth",
                                             &this->curr_depth_));
    OP_REQUIRES_OK(context, context->GetAttr("num_output",
                                             &this->num_output_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size",
                                             &this->kernel_size_));
    OP_REQUIRES_OK(context, context->GetAttr("stride", &this->stride_));
  }

  void Compute(OpKernelContext* context) override {
    // in gradients
    const Tensor& gradients = context->input(0);
    auto gradients_ptr = gradients.flat<float>().data();
    const TensorShape& gradients_shape = gradients.shape();
    CHECK_EQ(gradients_shape.dim_size(1), num_output_);

    // in data
    const Tensor& in_data = context->input(1);
    auto in_data_ptr = in_data.flat<float>().data();
    const TensorShape& in_data_shape = in_data.shape();
    channels_ = in_data_shape.dim_size(1);
    CHECK_EQ(in_data_shape.dim_size(0), gradients_shape.dim_size(0));
    CHECK_EQ(in_data_shape.dim_size(2), gradients_shape.dim_size(2));
    CHECK_EQ(in_data_shape.dim_size(3), gradients_shape.dim_size(3));

    // in filter
    const Tensor& in_filter = context->input(2);
    auto in_filter_ptr = in_filter.flat<float>().data();
    const TensorShape& in_filter_shape = in_filter.shape();
    CHECK_EQ(in_filter_shape.dim_size(0), num_output_);
    CHECK_EQ(in_filter_shape.dim_size(1), channels_);
    CHECK_EQ(in_filter_shape.dim_size(2),
             kernel_size_ * kernel_size_ * kernel_size_);

    // in octree
    const Tensor& in_octree = context->input(3);
    auto in_octree_ptr = in_octree.flat<int>().data();

    // out grad data
    Tensor* grad_data_tensor = nullptr;
    TensorShape grad_data_shape = in_data_shape;
    OP_REQUIRES_OK(context, context->allocate_output("grad_data",
                                grad_data_shape, &grad_data_tensor));
    auto grad_data_ptr = grad_data_tensor->flat<float>().data();

    // out grad filter
    Tensor* grad_filter_tensor = nullptr;
    TensorShape grad_filter_shape = in_filter_shape;
    OP_REQUIRES_OK(context, context->allocate_output("grad_filter",
                                grad_filter_shape, &grad_filter_tensor));
    auto grad_filter_ptr = grad_filter_tensor->flat<float>().data();
    octree::tensorflow_gpu_set_zero(context, grad_filter_ptr,
        grad_filter_tensor->NumElements());

    // parse octree info
    oct_batch_.set_gpu(in_octree_ptr);
    CHECK_EQ(oct_batch_.node_num(curr_depth_), in_data_shape.dim_size(2));

    // get workspace tensor
    kernel_dim_ = in_filter_shape.dim_size(1) * in_filter_shape.dim_size(2);
    workspace_h_ = in_data_shape.dim_size(2);
    workspace_ha_ = workspace_h_;
    workspace_n_ = 1;
    workspace_depth_ = curr_depth_;
    const int MAX_SIZE = octree::get_workspace_maxsize();
    int ideal_size = workspace_h_ * kernel_dim_;
    if (ideal_size > MAX_SIZE) {
      workspace_n_ = (ideal_size + MAX_SIZE - 1) / MAX_SIZE;
      workspace_ha_ = (workspace_h_ + workspace_n_ - 1) / workspace_n_;
    }
    const TensorShape workspace_shape({ kernel_dim_, workspace_ha_ });
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, workspace_shape,
                                &workspace_));
    auto workspace_ptr = workspace_.flat<float>().data();

    // get result_buffer tensor if needed
    if (workspace_n_ > 1) {
      const TensorShape result_shape({ num_output_, workspace_ha_ });
      OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, result_shape,
                                  &result_buffer_));
    }
    auto result_buffer_ptr = result_buffer_.flat<float>().data();

    // init neighbor info
    init_neigh_index(context, ni_);
    auto ni_ptr = ni_.flat<int32>().data();

    /// get weight gradient
    for (int n = 0; n < workspace_n_; ++n) {
      const float* result_buffer = gradients_ptr;
      octree::octree2col_gpu(context, workspace_ptr, in_data_ptr, channels_,
          workspace_h_, kernel_size_, stride_,
          oct_batch_.neighbor_gpu(workspace_depth_), ni_ptr, workspace_ha_, n);
      int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
      if (workspace_n_ > 1) {
        for (int c = 0; c < num_output_; ++c) {
          cudaMemcpy(result_buffer_ptr + c * workspace_ha_,
              gradients_ptr + c * workspace_h_ + n * workspace_ha_,
              num * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        result_buffer = result_buffer_ptr;
      }
      // C = alpha * A * B + beta * C
      octree::tensorflow_gpu_gemm(context, false, true,
          static_cast<int64>(num_output_), static_cast<int64>(kernel_dim_),
          static_cast<int64>(workspace_ha_), static_cast<float>(1),
          result_buffer, workspace_ptr, static_cast<float>(1),
          grad_filter_ptr);
    }

    /// get data gradient
    for (int n = 0; n < workspace_n_; ++n) {
      const float* result_buffer = gradients_ptr;
      if (workspace_n_ > 1) {
        int num = std::min(workspace_ha_, workspace_h_ - n * workspace_ha_);
        for (int c = 0; c < num_output_; ++c) {
          cudaMemcpy(result_buffer_ptr + c * workspace_ha_,
              gradients_ptr + c * workspace_h_ - n * workspace_ha_,
              num * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        result_buffer = result_buffer_ptr;
      }

      // gemm
      octree::tensorflow_gpu_gemm(context, true, false,
          static_cast<uint64>(kernel_dim_), static_cast<uint64>(workspace_ha_),
          static_cast<uint64>(num_output_), static_cast<float>(1),
          in_filter_ptr, result_buffer, static_cast<float>(0), workspace_ptr);
      // col2octree
      octree::col2octree_gpu(context, workspace_ptr, grad_data_ptr, channels_,
          workspace_h_, kernel_size_, stride_,
          oct_batch_.neighbor_gpu(workspace_depth_), ni_ptr, workspace_ha_, n);
    }
  }

 private:
  int kernel_size_;
  int kernel_dim_;
  int stride_;

  int channels_;
  int num_output_;

  int curr_depth_;
  OctreeBatchParser oct_batch_;

  int workspace_n_;
  int workspace_ha_;
  int workspace_h_;
  int workspace_depth_;
  Tensor workspace_;
  Tensor result_buffer_;

  Tensor ni_;
};
REGISTER_KERNEL_BUILDER(
    Name("OctreeConvGrad").Device(DEVICE_GPU), OctreeConvGradOP);


void init_neigh_index(OpKernelContext* ctx, Tensor& ni) {
  const TensorShape shape({ 216 });
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, shape, &ni));
  auto ni_ptr = ni.flat<int32>().data();

  // ni for kernel_size=3
  std::vector<int32> ni_temp(ni.NumElements());
  int id = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        for (int x = 0; x < 3; ++x) {
          for (int y = 0; y < 3; ++y) {
            for (int z = 0; z < 3; ++z) {
              ni_temp[id++] = (x + i << 4) | (y + j << 2) | z + k;
            }
          }
        }
      }
    }
  }
  cudaMemcpy(ni_ptr, ni_temp.data(), ni.NumElements() * sizeof(int32),
      cudaMemcpyHostToDevice);
}

}  // namespace tensorflow
