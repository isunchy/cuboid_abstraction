import tensorflow as tf
import os
from tensorflow.python.framework import ops

_current_path = os.path.dirname(os.path.realpath(__file__))
_primitive_gen_module = tf.load_op_library(os.path.join(_current_path, 'libprimitive_gen.so'))

# octree ops
octree_database = _primitive_gen_module.octree_database
octree_conv = _primitive_gen_module.octree_conv
octree_pooling = _primitive_gen_module.octree_pooling

octree_conv_grad = _primitive_gen_module.octree_conv_grad
octree_pooling_grad = _primitive_gen_module.octree_pooling_grad

# primitive ops
primitive_mutex_loss_v3 = _primitive_gen_module.primitive_mutex_loss_v3
primitive_group_points_v3 = _primitive_gen_module.primitive_group_points_v3
primitive_cube_coverage_loss_v3 = _primitive_gen_module.primitive_cube_coverage_loss_v3
primitive_coverage_loss_v2 = _primitive_gen_module.primitive_coverage_loss_v2
primitive_consistency_loss_v2 = _primitive_gen_module.primitive_consistency_loss_v2
primitive_symmetry_loss_v3 = _primitive_gen_module.primitive_symmetry_loss_v3
primitive_aligning_loss_v2 = _primitive_gen_module.primitive_aligning_loss_v2
primitive_cube_volume_v2 = _primitive_gen_module.primitive_cube_volume_v2
primitive_cube_area_average_loss = _primitive_gen_module.primitive_cube_area_average_loss
primitive_points_suffix_index = _primitive_gen_module.primitive_points_suffix_index

primitive_mutex_loss_v3_grad = _primitive_gen_module.primitive_mutex_loss_v3_grad
primitive_cube_coverage_loss_v3_grad = _primitive_gen_module.primitive_cube_coverage_loss_v3_grad
primitive_coverage_loss_v2_grad = _primitive_gen_module.primitive_coverage_loss_v2_grad
primitive_consistency_loss_v2_grad = _primitive_gen_module.primitive_consistency_loss_v2_grad
primitive_symmetry_loss_v3_grad = _primitive_gen_module.primitive_symmetry_loss_v3_grad
primitive_aligning_loss_v2_grad = _primitive_gen_module.primitive_aligning_loss_v2_grad
primitive_cube_area_average_loss_grad = _primitive_gen_module.primitive_cube_area_average_loss_grad

# mask prediction
primitive_coverage_split_loss_v3 = _primitive_gen_module.primitive_coverage_split_loss_v3
primitive_consistency_split_loss = _primitive_gen_module.primitive_consistency_split_loss
primitive_tree_generation = _primitive_gen_module.primitive_tree_generation
primitive_cube_coverage_loss_v4 = _primitive_gen_module.primitive_cube_coverage_loss_v4

primitive_coverage_split_loss_v3_grad = _primitive_gen_module.primitive_coverage_split_loss_v3_grad
primitive_consistency_split_loss_grad = _primitive_gen_module.primitive_consistency_split_loss_grad
primitive_cube_coverage_loss_v4_grad = _primitive_gen_module.primitive_cube_coverage_loss_v4_grad


ops.NotDifferentiable('OctreeDatabase')
ops.NotDifferentiable('PtimitiveGroupPointsV3')
ops.NotDifferentiable('PrimitiveCubeVolumeV2')
ops.NotDifferentiable('PrimitivePointsSuffixIndex')
ops.NotDifferentiable('PrimitiveTreeGeneration')


@ops.RegisterGradient('OctreeConv')
def _OctreeConvGrad(op, grad):
  return octree_conv_grad(grad,
                          op.inputs[0],
                          op.inputs[1],
                          op.inputs[2],
                          op.get_attr('curr_depth'),
                          op.get_attr('num_output'),
                          op.get_attr('kernel_size'),
                          op.get_attr('stride')) + \
         (None,)


@ops.RegisterGradient('OctreePooling')
def _OctreePoolingGrad(op, *grad):
  return [octree_pooling_grad(grad[0],
                              op.inputs[0],
                              op.outputs[1],
                              op.inputs[1],
                              op.get_attr('curr_depth')),
          None]


@ops.RegisterGradient('PrimitiveMutexLossV3')
def _PrimitiveMutexLossV3Grad(op, grad):
  return primitive_mutex_loss_v3_grad(grad,
                                      op.inputs[0],
                                      op.inputs[1],
                                      op.inputs[2],
                                      op.get_attr('scale'))


@ops.RegisterGradient('PrimitiveCubeCoverageLossV3')
def _PrimitiveCubeCoverageLossV3Grad(op, grad):
  return primitive_cube_coverage_loss_v3_grad(grad,
                                              op.inputs[0],
                                              op.inputs[1],
                                              op.inputs[2],
                                              op.inputs[3],
                                              op.inputs[4],
                                              op.get_attr('n_src_cube')) + \
         (None, None)


@ops.RegisterGradient('PrimitiveCoverageLossV2')
def _PrimitiveCoverageLossV2Grad(op, grad):
  return primitive_coverage_loss_v2_grad(grad,
                                         op.inputs[0],
                                         op.inputs[1],
                                         op.inputs[2],
                                         op.inputs[3]) + \
         (None,)


@ops.RegisterGradient('PrimitiveConsistencyLossV2')
def _PrimitiveConsistencyV2LossGrad(op, grad):
  return primitive_consistency_loss_v2_grad(grad,
                                            op.inputs[0],
                                            op.inputs[1],
                                            op.inputs[2],
                                            op.inputs[3],
                                            op.get_attr('scale'),
                                            op.get_attr('num_sample')) + \
         (None,)


@ops.RegisterGradient('PrimitiveSymmetryLossV3')
def _PrimitiveSymmetryLossV3Grad(op, grad):
  return primitive_symmetry_loss_v3_grad(grad,
                                         op.inputs[0],
                                         op.inputs[1],
                                         op.inputs[2],
                                         op.get_attr('scale'),
                                         op.get_attr('depth'))


@ops.RegisterGradient('PrimitiveAligningLossV2')
def _PrimitiveligningLossV2Grad(op, grad):
  return [primitive_aligning_loss_v2_grad(grad,
                                          op.inputs[0],
                                          op.inputs[1]),
          None]


@ops.RegisterGradient('PrimitiveCubeAreaAverageLoss')
def _PrimitiveCubeAreaAverageLossGrad(op, grad):
  return primitive_cube_area_average_loss_grad(grad,
                                               op.inputs[0])

@ops.RegisterGradient('PrimitiveCoverageSplitLossV3')
def _PrimitiveCoverageSplitLossV3Grad(op, *grad):
  return primitive_coverage_split_loss_v3_grad(grad[0],
                                            op.inputs[0],
                                            op.inputs[1],
                                            op.inputs[2],
                                            op.inputs[3]) + \
         (None,)

@ops.RegisterGradient('PrimitiveConsistencySplitLoss')
def _PrimitiveConsistencySplitLossGrad(op, grad):
  return primitive_consistency_split_loss_grad(grad,
                                               op.inputs[0],
                                               op.inputs[1],
                                               op.inputs[2],
                                               op.inputs[3],
                                               op.get_attr('scale'),
                                               op.get_attr('num_sample')) + \
         (None,)

@ops.RegisterGradient('PrimitiveCubeCoverageLossV4')
def _PrimitiveCubeCoverageLossV4Grad(op, *grad):
  return primitive_cube_coverage_loss_v4_grad(grad[0],
                                              op.inputs[0],
                                              op.inputs[1],
                                              op.inputs[2],
                                              op.inputs[3],
                                              op.inputs[4],
                                              op.get_attr('n_src_cube')) + \
         (None, None)
