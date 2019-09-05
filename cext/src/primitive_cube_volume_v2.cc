#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

void compute_cube_volume(OpKernelContext* context, const int n_cube,
    const int batch_size, const float* in_z, float* out_volume);

REGISTER_OP("PrimitiveCubeVolumeV2")
.Input("in_z: float")
.Output("out_volume: float")
.SetShapeFn([](shape_inference::InferenceContext* c) {
  c->set_output(0, c->MakeShape({1}));
  return Status::OK();
})
.Doc(R"doc(
Compute primitive cube volume.
)doc");

class PrimitiveCubeVolumeV2Op : public OpKernel {
public:
  explicit PrimitiveCubeVolumeV2Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // in_z [bs, n_cube * 3]
    const Tensor& in_z = context->input(0);
    auto in_z_ptr = in_z.flat<float>().data();
    batch_size_ = in_z.dim_size(0);
    n_cube_ = in_z.dim_size(1) / 3;

    // out volume
    Tensor* out_volume = nullptr;
    TensorShape out_volume_shape({1});
    OP_REQUIRES_OK(context, context->allocate_output("out_volume",
                                out_volume_shape, &out_volume));
    auto out_volume_ptr = out_volume->flat<float>().data();

    // compute cube volume
    compute_cube_volume(context, n_cube_, batch_size_, in_z_ptr,
        out_volume_ptr);
  }

 private:
  int n_cube_;
  int batch_size_;
};
REGISTER_KERNEL_BUILDER(Name("PrimitiveCubeVolumeV2").Device(DEVICE_GPU),
    PrimitiveCubeVolumeV2Op);

}  // namespace tensorflow
