#ifndef TENSORFLOW_USER_OPS_OCTREE_H_
#define TENSORFLOW_USER_OPS_OCTREE_H_

// gpu code file use octree.h should include octree.h first
#define EIGEN_USE_GPU

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow  {

class OctreeParser {
 public:
  OctreeParser(const void* data);

  int total_node_number() const { return *total_node_num_; }
  int final_node_number() const { return *final_node_num_; }
  // TODO(ps): modify octree format, and remove this function
  int node_number_nempty(int depth) const;

 public:
  // original pointer
  const void* metadata_;

  // octree header information
  const int* total_node_num_;
  const int* final_node_num_;
  const int* depth_;
  const int* full_layer_;
  const int* node_num_;
  const int* node_num_accu_;

  // octree structure
  const int* key_;
  const int* children_;

  // octree data
  const int* signal_;
  const int* misc_;
};

class OctreeInfo {
 public:
  OctreeInfo() { reset(); }
  void reset() { memset(this, 0, sizeof(OctreeInfo)); }
  void set(int batch_size, int depth, int full_layer, const int* nnum,
      const int* nnum_cum, const int* nnum_nempty, int content_flags);
  void update_ptr_dis();
  int total_nnum() const { return nnum_cum_[depth_ + 1]; }

  bool has_key() const { return (content_flags_ & 1) != 0; }
  bool has_children() const { return (content_flags_ & 2) != 0; }
  bool has_neigh() const { return (content_flags_ & 4) != 0; }
  // UNDERSTAND: octree info split meaning
  bool has_split() const { return (content_flags_ & 8) != 0; }

  int dis_key(int depth) const {
    CHECK(has_key());
    return dis_key_[depth];
  }
  int dis_children(int depth) const {
    CHECK(has_children());
    return dis_children_[depth];
  }
  int dis_neigh(int depth) const {
    CHECK(has_neigh());
    return dis_neigh_[depth];
  }
  int dis_split(int depth) const {
    CHECK(has_split());
    return dis_split_[depth];
  }

  static int octree_sizeofint(int total_nnum, int content_flags);

 public:
  int batch_size_;
  int depth_;
  int full_layer_;
  int nnum_[16];         // node number of each depth
  int nnum_cum_[16];     // cumulative node number
  int nnum_nempty_[16];  // non-empty node number of each depth
  int content_flags_;

  // average neighbor number, 2*2*2 nodes share 4*4*4 neighbor nodes
  static const int AVG_NGH_NUM = 8;

 private:
  int dis_key_[16];
  int dis_children_[16];
  int dis_neigh_[16];
  int dis_split_[16];
};

class OctreeBatch {
 public:
  enum DataCategory { CLASSIFICATION, SEGMENTATION };

 public:
  void set_octreebatch(OpKernelContext* context,
    const Tensor& octree_buffer_tensor,
    const int content_flags = 7,
    const DataCategory dc = CLASSIFICATION);

 public:
  Tensor data_;
  Tensor label_;
  Tensor octree_;
};

class OctreeBatchParser {
 public:
  OctreeBatchParser() : h_metadata_(nullptr), d_metadata_(nullptr),
    oct_info_(nullptr), const_ptr_(true), oct_info_buffer_() {}
  void set_cpu(const void* ptr);
  void set_gpu(const void* prt, const void* oct_info = nullptr);
  void set_cpu(void* ptr, const OctreeInfo* octinfo = nullptr);
  void set_gpu(void* ptr, const OctreeInfo* octinfo = nullptr);

  const OctreeInfo* octree_info() { return oct_info_; };
  OctreeInfo* mutable_octree_info() { return oct_info_; };

  // total node number
  int total_node_num() {
    CHECK(oct_info_);
    return oct_info_->total_nnum();
  }

  // batch size
  int batch_size() {
    CHECK(oct_info_);
    return oct_info_->batch_size_;
  }

  // full layer
  int full_layer() {
    CHECK(oct_info_);
    return oct_info_->full_layer_;
  }

  // depth
  int depth() {
    CHECK(oct_info_);
    return oct_info_->depth_;
  }

  // node number of the specified layer
  int node_num(int depth) {
    CHECK(oct_info_);
    return oct_info_->nnum_[depth];
  }

  // cumulative node number of the specified layer
  int node_num_cum(int depth) {
    CHECK(oct_info_);
    return oct_info_->nnum_cum_[depth];
  }

  // non-empty node number of the specified layer
  int node_num_nonempty(int depth) {
    CHECK(oct_info_);
    return oct_info_->nnum_nempty_[depth];
  }

  // pointer to the first children of the specified layer
  const int* children_cpu(int depth) {
    CHECK(h_metadata_);
    return h_metadata_ + oct_info_->dis_children(depth);
  }
  const int* children_gpu(int depth) {
    CHECK(d_metadata_);
    return d_metadata_ + oct_info_->dis_children(depth);
  }
  int* mutable_children_cpu(int depth) {
    CHECK(h_metadata_ && (const_ptr_ == false));
    return h_metadata_ + oct_info_->dis_children(depth);
  }
  int* mutable_children_gpu(int depth) {
    CHECK(d_metadata_ && (const_ptr_ == false));
    return d_metadata_ + oct_info_->dis_children(depth);
  }

  // pointer to the first key of the specified layer
  const int* key_cpu(int depth) {
    CHECK(h_metadata_);
    return h_metadata_ + oct_info_->dis_key(depth);
  }
  const int* key_gpu(int depth) {
    CHECK(d_metadata_);
    return d_metadata_ + oct_info_->dis_key(depth);
  }
  int* mutable_key_cpu(int depth) {
    CHECK(h_metadata_ && (const_ptr_ == false));
    return h_metadata_ + oct_info_->dis_key(depth);
  }
  int* mutable_key_gpu(int depth) {
    CHECK(d_metadata_ && (const_ptr_ == false));
    return d_metadata_ + oct_info_->dis_key(depth);
  }

  // pointer to the first neighbor of the specified layer
  const int* neighbor_cpu(int depth) {
    CHECK(h_metadata_);
    return h_metadata_ + oct_info_->dis_neigh(depth);
  }
  const int* neighbor_gpu(int depth) {
    CHECK(d_metadata_);
    return d_metadata_ + oct_info_->dis_neigh(depth);
  }
  int* mutable_neighbor_cpu(int depth) {
    CHECK(h_metadata_ && const_ptr_ == false);
    return h_metadata_ + oct_info_->dis_neigh(depth);
  }
  int* mutable_neighbor_gpu(int depth) {
    CHECK(d_metadata_ && const_ptr_ == false);
    return d_metadata_ + oct_info_->dis_neigh(depth);
  }

  // pointer to the first split-label of the specified layer
  const int* split_cpu(int depth) {
    CHECK(h_metadata_);
    return h_metadata_ + oct_info_->dis_split(depth);
  }
  const int* split_gpu(int depth) {
    CHECK(d_metadata_);
    return d_metadata_ + oct_info_->dis_split(depth);
  }
  int* mutable_split_cpu(int depth) {
    CHECK(h_metadata_ && const_ptr_ == false);
    return h_metadata_ + oct_info_->dis_split(depth);
  }
  int* mutable_split_gpu(int depth) {
    CHECK(d_metadata_ && const_ptr_ == false);
    return d_metadata_ + oct_info_->dis_split(depth);
  }

 public:
  // original pointer
  int* h_metadata_;
  int* d_metadata_;
  OctreeInfo* oct_info_;
  bool const_ptr_;

 private:
  OctreeInfo oct_info_buffer_;
};

namespace octree {

void pad_forward_cpu(float* Y, int Hy, int Cy, const float* X, int Hx,
    const int* label);
void pad_forward_gpu(OpKernelContext* ctx, float* Y, int Hy, int Cy,
    const float* X, int Hx, const int* label);

void pad_backward_cpu(float* X, int Hx, int Cx, const float* Y, int Hy,
    const int* label);
void pad_backward_gpu(OpKernelContext* ctx, float* X, int Hx, int Cx,
    const float* Y, int Hy, const int* label);

void octree2col_cpu(float* data_col, const float* data_octree, int channel,
    int height, int kernel_size, int stride, const int* neigh, const int* ni,
    int height_col, int n);
void octree2col_gpu(OpKernelContext* ctx, float* data_col,
    const float* data_octree, int channel, int height, int kernel_size,
    int stride, const int* neigh, const int* ni, int height_col, int n);

void col2octree_cpu(const float* data_col, float* data_octree, int channel,
    int height, int kernel_size, int stride, const int* neigh, const int* ni,
    int height_col, int n);
void col2octree_gpu(OpKernelContext* ctx, const float* data_col,
    float* data_octree, int channel, int height, int kernel_size, int stride,
    const int* neigh, const int* ni, int height_col, int n);

void generate_key_cpu(int* key_split, const int* key, const int* children,
    int node_num);
void generate_key_gpu(OpKernelContext* ctx, int* key_split, const int* key,
    const int* children, int node_num);
void generate_key_cpu(int* key, const int depth, const int batch_size);
void generate_key_gpu(OpKernelContext* ctx, int* key, int depth,
    int batch_size);

//void calc_neigh_cpu(int* neigh_split, const int* neigh, const int* children,
//    int node_num);
void calc_neigh_gpu(OpKernelContext* ctx, int* neigh_split, const int* neigh,
  const int* children, const int* parent, const int* dis, int node_num);
void calc_neigh_cpu(int* neigh, const int depth, const int batch_size);
void calc_neigh_gpu(OpKernelContext* ctx, int* neigh, int depth,
    int batch_size);

// calculate neighborhood information with the hash table
void calc_neighbor(int* neigh, const unsigned* key, int node_num,
    int displacement);

inline void compute_key(int& key, const int* pt, int depth);
inline void compute_pt(int* pt, const int& key, int depth);

void tensorflow_gpu_gemm(OpKernelContext* ctx, bool transa, bool transb,
    uint64 m, uint64 n, uint64 k, float alpha, const float* a, const float* b,
    float beta, float* c);

template <typename T>
void tensorflow_gpu_set_zero(OpKernelContext* ctx, T* Y, const int N);

int get_workspace_maxsize();

}  // namespace octree

}  // namespace tensorflow

#endif // !TENSORFLOW_USER_OPS_OCTREE_H_
