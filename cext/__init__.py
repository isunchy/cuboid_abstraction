import os
import tensorflow as tf
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
primitive_mutex_loss = _primitive_gen_module.primitive_mutex_loss
primitive_group_points = _primitive_gen_module.primitive_group_points
primitive_cube_coverage_loss = _primitive_gen_module.primitive_cube_coverage_loss
primitive_coverage_loss = _primitive_gen_module.primitive_coverage_loss
primitive_consistency_loss = _primitive_gen_module.primitive_consistency_loss
primitive_symmetry_loss = _primitive_gen_module.primitive_symmetry_loss
primitive_aligning_loss = _primitive_gen_module.primitive_aligning_loss
primitive_cube_volume = _primitive_gen_module.primitive_cube_volume
primitive_cube_area_average_loss = _primitive_gen_module.primitive_cube_area_average_loss
primitive_points_suffix_index = _primitive_gen_module.primitive_points_suffix_index

primitive_mutex_loss_grad = _primitive_gen_module.primitive_mutex_loss_grad
primitive_cube_coverage_loss_grad = _primitive_gen_module.primitive_cube_coverage_loss_grad
primitive_coverage_loss_grad = _primitive_gen_module.primitive_coverage_loss_grad
primitive_consistency_loss_grad = _primitive_gen_module.primitive_consistency_loss_grad
primitive_symmetry_loss_grad = _primitive_gen_module.primitive_symmetry_loss_grad
primitive_aligning_loss_grad = _primitive_gen_module.primitive_aligning_loss_grad
primitive_cube_area_average_loss_grad = _primitive_gen_module.primitive_cube_area_average_loss_grad

# mask prediction
primitive_coverage_split_loss = _primitive_gen_module.primitive_coverage_split_loss
primitive_consistency_split_loss = _primitive_gen_module.primitive_consistency_split_loss
primitive_tree_generation = _primitive_gen_module.primitive_tree_generation

primitive_coverage_split_loss_grad = _primitive_gen_module.primitive_coverage_split_loss_grad
primitive_consistency_split_loss_grad = _primitive_gen_module.primitive_consistency_split_loss_grad

# cube update
primitive_coverage_select_loss = _primitive_gen_module.primitive_coverage_select_loss
primitive_consistency_select_loss = _primitive_gen_module.primitive_consistency_select_loss
primitive_mutex_select_loss = _primitive_gen_module.primitive_mutex_select_loss

primitive_coverage_select_loss_grad = _primitive_gen_module.primitive_coverage_select_loss_grad
primitive_consistency_select_loss_grad = _primitive_gen_module.primitive_consistency_select_loss_grad
primitive_mutex_select_loss_grad = _primitive_gen_module.primitive_mutex_select_loss_grad


ops.NotDifferentiable('OctreeDatabase')
ops.NotDifferentiable('PtimitiveGroupPoints')
ops.NotDifferentiable('PrimitiveCubeVolume')
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


@ops.RegisterGradient('PrimitiveMutexLoss')
def _PrimitiveMutexLossGrad(op, grad):
  return primitive_mutex_loss_grad(grad,
                                   op.inputs[0],
                                   op.inputs[1],
                                   op.inputs[2],
                                   op.get_attr('scale'))


@ops.RegisterGradient('PrimitiveCoverageLoss')
def _PrimitiveCoverageLossGrad(op, grad):
  return primitive_coverage_loss_grad(grad,
                                         op.inputs[0],
                                         op.inputs[1],
                                         op.inputs[2],
                                         op.inputs[3]) + \
         (None,)


@ops.RegisterGradient('PrimitiveConsistencyLoss')
def _PrimitiveConsistencyLossGrad(op, grad):
  return primitive_consistency_loss_grad(grad,
                                         op.inputs[0],
                                         op.inputs[1],
                                         op.inputs[2],
                                         op.inputs[3],
                                         op.get_attr('scale'),
                                         op.get_attr('num_sample')) + \
         (None,)


@ops.RegisterGradient('PrimitiveSymmetryLoss')
def _PrimitiveSymmetryLossGrad(op, grad):
  return primitive_symmetry_loss_grad(grad,
                                      op.inputs[0],
                                      op.inputs[1],
                                      op.inputs[2],
                                      op.get_attr('scale'),
                                      op.get_attr('depth'))


@ops.RegisterGradient('PrimitiveAligningLoss')
def _PrimitiveligningLossGrad(op, grad):
  return [primitive_aligning_loss_grad(grad,
                                       op.inputs[0],
                                       op.inputs[1]),
          None]


@ops.RegisterGradient('PrimitiveCubeAreaAverageLoss')
def _PrimitiveCubeAreaAverageLossGrad(op, grad):
  return primitive_cube_area_average_loss_grad(grad,
                                               op.inputs[0])

@ops.RegisterGradient('PrimitiveCoverageSplitLoss')
def _PrimitiveCoverageSplitLossGrad(op, *grad):
  return primitive_coverage_split_loss_grad(grad[0],
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

@ops.RegisterGradient('PrimitiveCubeCoverageLoss')
def _PrimitiveCubeCoverageLossGrad(op, *grad):
  return primitive_cube_coverage_loss_grad(grad[0],
                                           op.inputs[0],
                                           op.inputs[1],
                                           op.inputs[2],
                                           op.inputs[3],
                                           op.inputs[4],
                                           op.get_attr('n_src_cube')) + \
         (None, None)

@ops.RegisterGradient("PrimitiveMutexSelectLoss")
def _PrimitiveMutexSelectLossGrad(op, grad):
  return primitive_mutex_select_loss_grad(grad,
                                      op.inputs[0],
                                      op.inputs[1],
                                      op.inputs[2],
                                      op.inputs[3],
                                      op.get_attr("scale")) + \
         (None,)

@ops.RegisterGradient("PrimitiveCoverageSelectLoss")
def _PrimitiveCoverageSelectLossGrad(op, grad):
  return primitive_coverage_select_loss_grad(grad,
                                         op.inputs[0],
                                         op.inputs[1],
                                         op.inputs[2],
                                         op.inputs[3],
                                         op.inputs[4]) + \
         (None, None)

@ops.RegisterGradient("PrimitiveConsistencySelectLoss")
def _PrimitiveConsistencySelectLossGrad(op, grad):
  return primitive_consistency_select_loss_grad(grad,
                                            op.inputs[0],
                                            op.inputs[1],
                                            op.inputs[2],
                                            op.inputs[3],
                                            op.inputs[4],
                                            op.get_attr("scale"),
                                            op.get_attr("num_sample")) + \
         (None, None)
