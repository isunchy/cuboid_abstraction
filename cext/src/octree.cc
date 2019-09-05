#include "octree.h"

#include <cuda_runtime.h>

namespace tensorflow {

namespace octree {

void pad_forward_cpu(float* Y, int Hy, int Cy, const float* X, int Hx,
    const int* label) {
  // Note: Cx == Cy
  for (int c = 0; c < Cy; ++c) {
    for (int h = 0; h < Hy; ++h) {
      Y[c * Hy + h] = label[h] == -1 ? float(0) : X[c * Hx + label[h]];
    }
  }
}

void pad_backward_cpu(float* X, int Hx, int Cx, const float* Y, int Hy,
    const int* label) {
  // Note: Cx == Cy
  for (int c = 0; c < Cx; ++c) {
    for (int h = 0; h < Hy; ++h) {
      if (label[h] != -1)
        X[c * Hx + label[h]] = Y[c * Hy + h];
    }
  }
}

void octree2col_cpu(float* data_col, const float* data_octree, int channel,
    int height, int kernel_size, int stride, const int* neigh, const int* ni,
    int height_col, int n) {
  // height: the ideal height of workspace
  // height_col: the actual height of workspace
  const int octree_h = height << 3 * (stride - 1);
  const int kernel = kernel_size * kernel_size * kernel_size;
  for (int c = 0; c < channel; ++c) {
    for (int k = 0; k < kernel; ++k) {
      int h_start = n * height_col;
      int i_start = (c * kernel + k) * height_col - h_start;
      for (int h = h_start; h < h_start + height_col; ++h) {
        if (h >= height) {
          data_col[i_start + h] = float(0);
          continue;
        }
        const int index = stride == 2 ? (h << 6) + ni[k] :
          (h >> 3 << 6) + ni[(h % 8) * kernel + k];
        const int p = neigh[index];
        data_col[i_start + h] =
          p == -1 ? float(0) : data_octree[c * octree_h + p];
      }
    }
  }
}

void col2octree_cpu(const float* data_col, float* data_octree, int channel,
    int height, int kernel_size, int stride, const int* neigh, const int* ni,
    int height_col, int n) {
  // height: the ideal height of workspace
  // height_col: the actual height of workspace
  const int octree_h = height << 3 * (stride - 1);
  const int kernel = kernel_size * kernel_size * kernel_size;
  // set data_octree to zero once when n == 0
  memset(data_octree, 0, channel * octree_h * sizeof(float));
  for (int c = 0; c < channel; ++c) {
    for (int k = 0; k < kernel; ++k) {
      int h_start = n * height_col;
      int i_start = (c * kernel + k) * height_col - h_start;
      for (int h = h_start; h < h_start + height_col; ++h) {
        if (h >= height) continue;
        const int index = stride == 2 ? (h << 6) + ni[k] :
          (h >> 3 << 6) + ni[(h % 8) * kernel + k];
        const int p = neigh[index];
        if (p != -1)
          data_octree[c * octree_h + p] += data_col[i_start + h];
      }
    }
  }
}

void generate_key_cpu(int* key_split, const int* key, const int* children,
    int node_num) {
  typedef unsigned char ubyte;
  for (int i = 0; i < node_num; ++i) {
    int label = children[i];
    if (label == -1) continue;
    const ubyte* k0 = reinterpret_cast<const ubyte*>(key + i);
    for (ubyte j = 0; j < 8; ++j) {
      ubyte* k1 = reinterpret_cast<ubyte*>(key_split + 8 * label + j);
      k1[0] = (k0[0] << 1) | ((j & 4) >> 2);
      k1[1] = (k0[1] << 1) | ((j & 2) >> 1);
      k1[2] = (k0[2] << 1) | (j & 1);
      k1[3] = k0[3];
    }
  }
}

void generate_key_cpu(int* key, const int depth, const int batch_size) {
  const int node_num = 1 << 3 * depth;
  for (int n = 0; n < batch_size; ++n) {
    for (int k = 0; k < node_num; ++k) {
      unsigned xyz = 0;
      unsigned char* ptr = reinterpret_cast<unsigned char*>(&xyz);
      for (int d = 0; d < depth; ++d) {
        ptr[0] |= (k & (1 << 3 * d + 2)) >> (2 * d + 2);
        ptr[1] |= (k & (1 << 3 * d + 1)) >> (2 * d + 1);
        ptr[2] |= (k & (1 << 3 * d + 0)) >> (2 * d + 0);
      }
      ptr[3] = n;
      key[n * node_num + k] = xyz;
    }
  }
}

// TODO(isunchy): replace Octree class (Octree::get_parent_array())
//void calc_neigh_cpu(int* neigh_split, const int* neigh, const int* children,
//    int node_num) {
//  const int* parent = Octree::get_parent_array().cpu_data();
//  const int* dis = Octree::get_dis_array().cpu_data();
//
//  for (int i = 0; i < node_num; ++i) {
//    int l0 = children[i];
//    if (l0 == -1) continue;
//    const int* ngh0 = neigh + (i >> 3 << 6);
//    const int* pi0 = parent + (i % 8) * 64;
//    int* ngh1 = neigh_split + (l0 << 6);
//    for (int j = 0; j < 64; ++j) {
//      ngh1[j] = -1;
//      int k = ngh0[pi0[j]];
//      if (k != -1) {
//        int l1 = children[k];
//        if (l1 != -1) {
//          ngh1[j] = (l1 << 3) + dis[j];
//        }
//      }
//    }
//  }
//}

void calc_neigh_cpu(int* neigh, const int depth, const int batch_size) {
  unsigned node_num = 1 << 3 * depth;
  const unsigned bound = 1 << depth;
  for (unsigned n = 0; n < batch_size; ++n) {
    for (unsigned i = 0; i < node_num; i += 8) {
      // key to xyz
      unsigned x0 = 0, y0 = 0, z0 = 0;
      for (unsigned d = 0; d < depth; ++d) {
        x0 |= (i & (1 << 3 * d + 2)) >> (2 * d + 2);
        y0 |= (i & (1 << 3 * d + 1)) >> (2 * d + 1);
        z0 |= (i & (1 << 3 * d + 0)) >> (2 * d + 0);
      }

      for (unsigned x = 0; x < 4; ++x) {
        for (unsigned y = 0; y < 4; ++y) {
          for (unsigned z = 0; z < 4; ++z) {
            unsigned x1 = x0 + x - 1;
            unsigned y1 = y0 + y - 1;
            unsigned z1 = z0 + z - 1;

            int v = -1;
            if ((x1 & bound) == 0 && (y1 & bound) == 0 && (z1 & bound) == 0) {
              unsigned key1 = 0;
              for (int d = 0; d < depth; ++d) {
                unsigned mask = 1u << d;
                key1 |=
                  ((x1 & mask) << (2 * d + 2)) |
                  ((y1 & mask) << (2 * d + 1)) |
                  ((z1 & mask) << (2 * d + 0));
              }
              v = key1 + n*node_num;
            }

            unsigned xyz = (x << 4) | (y << 2) | z;
            neigh[xyz + i * 8 + n * node_num * 8] = v;
          }
        }
      }
    }
  }
}

void calc_neighbor(int* neigh, const unsigned* key, int node_num,
    int displacement) {
  typedef unsigned char ubyte;

  // build hash table
  std::vector<std::pair<unsigned, int>> entries(node_num);
  for (int id = 0; id < node_num; ++id) {
    entries[id] = std::make_pair(key[id], id + displacement);
  }
  std::unordered_map<unsigned, int> hash_table(entries.begin(), entries.end());

  // calc neighborhood
  for (int id = 0; id < node_num; id += 8) {
    // the neighborhood volume
    int* ngh = neigh + id * 8;
    const ubyte* k0 = (const ubyte*)(key + id);
    // currently the maximize octree depth is 8
    ubyte k1[4] = { 0, 0, 0, k0[3] };
    for (ubyte x = 0; x < 4; ++x) {
      k1[0] = k0[0] + x - 1;
      for (ubyte y = 0; y < 4; ++y) {
        k1[1] = k0[1] + y - 1;
        for (ubyte z = 0; z < 4; ++z) {
          k1[2] = k0[2] + z - 1;

          // find
          unsigned* k2 = reinterpret_cast<unsigned*>(k1);
          auto rst = hash_table.find(*k2);
          ubyte i = (x << 4) | (y << 2) | z;
          if (rst != hash_table.end()) {
            ngh[i] = rst->second;
          }
          else {
            ngh[i] = -1;
          }
        }
      }
    }
  }
}

inline void compute_key(int& key, const int* pt, int depth) {
  key = 0;
  for (int i = 0; i < depth; ++i) {
    int mask = 1u << i;
    for (int j = 0; j < 3; j++) {
      key |= (pt[j] & mask) << (2 * i + 2 - j);
    }
  }
}

inline void compute_pt(int* pt, const int& key, int depth) {
  for (int i = 0; i < 3; pt[i++] = 0u);
  for (int i = 0; i < depth; ++i) {
    for (int j = 0; j < 3; ++j) {
      int mask = 1u << (3 * i + 2 - j);
      pt[j] |= (key & mask) >> (2 * i + 2 - j);
    }
  }
}

int get_workspace_maxsize() {
  return 256 * 1024 * 1024;
}

}  // namespace octree

// --- OctreeParser ---
OctreeParser::OctreeParser(const void* data) {
  metadata_ = data;

  total_node_num_ = static_cast<const int*>(metadata_);
  final_node_num_ = total_node_num_ + 1;
  depth_ = final_node_num_ + 1;
  full_layer_ = depth_ + 1;
  node_num_ = full_layer_ + 1;
  node_num_accu_ = node_num_ + (*depth_) + 1;

  key_ = node_num_accu_ + (*depth_) + 2;
  children_ = key_ + (*total_node_num_);

  signal_ = children_ + (*total_node_num_);
  misc_ = signal_ + 3 * (*final_node_num_);
}

int OctreeParser::node_number_nempty(int depth) const {
  int num = 0;
  const int* ptr = children_ + node_num_accu_[depth];
  for (int i = node_num_[depth] - 1; i >= 0; --i) {
    // find the last element which is not equal to -1
    if (ptr[i] != -1) {
      num = ptr[i] + 1;
      break;
    }
  }
}

// --- OctreeInfo ---
void OctreeInfo::set(int batch_size, int depth, int full_layer,
    const int* nnum, const int* nnum_cum, const int* nnum_nempty,
    int content_flags) {
  batch_size_ = batch_size;
  depth_ = depth;
  full_layer_ = full_layer;
  memcpy(nnum_, nnum, sizeof(int) * (depth + 1));
  memcpy(nnum_cum_, nnum_cum, sizeof(int) * (depth + 2));
  memcpy(nnum_nempty_, nnum_nempty, sizeof(int) * (depth + 1));
  content_flags_ = content_flags;

  update_ptr_dis();
}

void OctreeInfo::update_ptr_dis() {
  CHECK(content_flags_ != 0);
  int dis = sizeof(OctreeInfo) / sizeof(int);
  for (int d = 0; d < depth_ + 1; ++d) {
    if (0 != (content_flags_ & 1)) {
      dis_key_[d] = dis;
      dis += nnum_[d];
    }
    if (0 != (content_flags_ & 2)) {
      dis_children_[d] = dis;
      dis += nnum_[d];
    }
    if (0 != (content_flags_ & 4)) {
      dis_neigh_[d] = dis;
      dis += nnum_[d] * AVG_NGH_NUM;
    }
    if (0 != (content_flags_ & 8)) {
      dis_split_[d] = dis;
      dis += nnum_[d];
    }
  }
}

// UNDERSTAND(isunchy): octree_sizeofint
int OctreeInfo::octree_sizeofint(int total_nnum, int content_flags) {
  int sz = sizeof(OctreeInfo) / sizeof(int);
  if (0 != (content_flags & 1)) sz += total_nnum;
  if (0 != (content_flags & 2)) sz += total_nnum;
  if (0 != (content_flags & 4)) sz += total_nnum * AVG_NGH_NUM;
  if (0 != (content_flags & 8)) sz += total_nnum;
  return sz;
}

// --- OctreeBatch ---
void OctreeBatch::set_octreebatch(OpKernelContext* context,
    const Tensor& octree_buffer_tensor, int content_flags, DataCategory dc) {
  /// octree parser
  auto octree_buffer = octree_buffer_tensor.flat<string>();
  int batch_size = octree_buffer_tensor.shape().dim_size(0);
    // octree_buffer_tensor.shape().dims() == 1
  std::vector<OctreeParser> octree_parsers;
  for (int i = 0; i < batch_size; ++i) {
    octree_parsers.push_back(OctreeParser(octree_buffer(i).data()));
  }

  /// get node number information
  // get depth and full layer information
  int depth = *octree_parsers[0].depth_;
  int full_layer = *octree_parsers[0].full_layer_;
  for (int i = 1; i < batch_size; ++i) {
    CHECK_EQ(depth, *octree_parsers[i].depth_);
    CHECK_EQ(full_layer, *octree_parsers[i].full_layer_);
  }

  // node and non-empty node number in each octree
  int sz = (depth + 1) * batch_size;
  std::vector<int> nnum(sz), nnum_nempty(sz);
  for (int i = 0; i < batch_size; ++i) {
    for (int d = 0; d < depth + 1; ++d) {
      int p = i * (depth + 1) + d;
      nnum[p] = octree_parsers[i].node_num_[d];
      nnum_nempty[p] = octree_parsers[i].node_number_nempty(d);
    }
  }

  // cumulative node and non-empty node number in each layers
  sz = (depth + 1) * (batch_size + 1);
  std::vector<int> nnum_cum_layer(sz), nnum_cum_nempty_layer(sz);
  for (int d = 0; d < depth + 1; ++d) {
    nnum_cum_layer[d] = 0;
    nnum_cum_nempty_layer[d] = 0;
    for (int i = 0; i < batch_size; ++i) {
      int p = i * (depth + 1) + d;
      int q = p + depth + 1;
      nnum_cum_layer[q] = nnum[p] + nnum_cum_layer[p];
      nnum_cum_nempty_layer[q] = nnum_nempty[p] + nnum_cum_nempty_layer[p];
    }
  }

  // cumulative node number for each octree
  sz = (depth + 1) * batch_size;
  std::vector<int> nnum_cum_octree(sz);
  for (int i = 0; i < batch_size; ++i) {
    nnum_cum_octree[i * (depth + 1)] = 0;
    for (int d = 0; d < depth; ++d) {
      int p = i * (depth + 1) + d;
      nnum_cum_octree[p + 1] = nnum_cum_octree[p] + nnum[p];
    }
  }

  // node and non-empty node number of the batch
  std::vector<int> nnum_batch(depth + 1), nnum_batch_nempty(depth + 1);
  for (int d = 0; d < depth + 1; ++d) {
    int p = batch_size * (depth + 1) + d;
    nnum_batch[d] = nnum_cum_layer[p];
    nnum_batch_nempty[d] = nnum_cum_nempty_layer[p];
  }

  // cumulative node number of the batch
  std::vector<int> nnum_batch_cum(depth + 2);
  nnum_batch_cum[0] = 0;
  for (int d = 0; d < depth + 1; ++d) {
    nnum_batch_cum[d + 1] = nnum_batch_cum[d] + nnum_batch[d];
  }

  /// init space
  // octree_
  int total_nnum = nnum_batch_cum[depth + 1];
  TensorShape octree_shape({ OctreeInfo::octree_sizeofint(total_nnum,
                             content_flags) });
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DT_INT32, octree_shape, &octree_));
  int* octree_ptr = octree_.flat<int32>().data();
  OctreeInfo oct_info;
  oct_info.set(batch_size, depth, full_layer, nnum_batch.data(),
      nnum_batch_cum.data(), nnum_batch_nempty.data(), content_flags);
  OctreeBatchParser octbatch_parser;
  octbatch_parser.set_cpu(octree_ptr, &oct_info);

  // data_
  int deepest_nnum = nnum_batch[depth];
  TensorShape data_shape({ 1, 3, deepest_nnum, 1 });
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DT_FLOAT, data_shape, &data_));
  float* data_ptr = data_.flat<float>().data();

  // label_
  TensorShape label_shape;
  if (dc == CLASSIFICATION) label_shape = TensorShape({ batch_size });
  if (dc == SEGMENTATION) label_shape = TensorShape({ deepest_nnum });
  OP_REQUIRES_OK(context,
                 context->allocate_temp(DT_INT32, label_shape, &label_));
  int* label_ptr = label_.flat<int>().data();

  /// set data
  for (int i = 0; i < batch_size; ++i) {
    // copy key
    for (int d = 0; d < depth + 1; ++d) {
      if (!oct_info.has_key()) break;
      int p = i * (depth + 1) + d;
      int* des = octbatch_parser.mutable_key_cpu(d) + nnum_cum_layer[p];
      const int* src = octree_parsers[i].key_ + nnum_cum_octree[p];
      for (int j = 0; j < nnum[p]; ++j) {
        des[j] = src[j];
        unsigned char* ptr = reinterpret_cast<unsigned char*>(des + j);
        ptr[3] = i;
      }
    }

    // copy children
    for (int d = 0; d < depth + 1; ++d) {
      if (!oct_info.has_children()) break;
      int p = i * (depth + 1) + d;
      int* des = octbatch_parser.mutable_children_cpu(d) + nnum_cum_layer[p];
      const int* src = octree_parsers[i].children_ + nnum_cum_octree[p];
      for (int j = 0; j < nnum[p]; ++j) {
        des[j] = -1 == src[j] ? src[j] : src[j] + nnum_cum_nempty_layer[p];
      }
    }

    // copy data
    int p = i * (depth + 1) + depth;
    for (int c = 0; c < 3; ++c) {
      float* des = data_ptr + c * nnum_batch[depth] + nnum_cum_layer[p];
      const float* src = reinterpret_cast<const float*>(
                             octree_parsers[i].signal_) + c * nnum[p];
      memcpy(des, src, nnum[p] * sizeof(float));
    }

    // copy label
    if (dc == SEGMENTATION) {
      // seg label type is int
      for (int j = 0; j < nnum[p]; ++j) {
        label_ptr[nnum_cum_layer[p] + j] = octree_parsers[i].misc_[j];
      }
    }

    // calc and set neighbor info
    for (int d = 1; d < depth + 1; ++d) {
      if (!oct_info.has_neigh()) break;
      int p = i * (depth + 1) + d;
      const unsigned* key =
          reinterpret_cast<const unsigned*>(octree_parsers[i].key_) +
          nnum_cum_octree[p];
      int* neigh = octbatch_parser.mutable_neighbor_cpu(d) +
          OctreeInfo::AVG_NGH_NUM * nnum_cum_layer[p];
      octree::calc_neighbor(neigh, key, nnum[p], nnum_cum_layer[p]);
    }
  }
}

// --- OctreeBatchParser ---
//void OctreeBatchParser::set_cpu(void* data, int depth,
//  int total_nnum, int batch_size, int content_flags)
//{
//  metadata_ = (int*)data;
//
//  batch_size_ = metadata_;
//  if (0 == batch_size) batch_size = *batch_size_;
//  CHECK(batch_size > 0) << "Invalid octree depth";
//  depth_ = batch_size_ + 1;
//  full_layer_ = depth_ + 1;
//  node_num_ = full_layer_ + 1;
//  if (0 == depth) depth = *depth_;
//  CHECK(depth < 10 && depth>0) << "Invalid octree depth";
//  node_num_cum_ = node_num_ + depth + 1;
//  node_num_nempty_ = node_num_cum_ + depth + 2;
//  node_num_oct_ = node_num_nempty_ + depth + 1;
//  node_num_nempty_oct_ = node_num_oct_ + (depth + 1)*batch_size;
//  if (0 == total_nnum) total_nnum = node_num_cum_[depth + 1];
//  CHECK(total_nnum > 0) << "Invalid node number";
//
//  content_flags_ = node_num_nempty_oct_ + (depth + 1)*batch_size;
//  if (0 == content_flags) content_flags = *content_flags_;
//  CHECK(content_flags != 0) << "Invalid flags";
//
//  int* ptr = content_flags_ + 1;
//  key_ = children_ = neigh_ = nullptr;
//  if (0 != (content_flags & 1))
//  {
//    key_ = ptr;
//    ptr += total_nnum;
//  }
//  if (0 != (content_flags & 2))
//  {
//    children_ = ptr;
//    ptr += total_nnum;
//  }
//  if (0 != (content_flags & 4))
//  {
//    neigh_ = ptr;
//  }
//}

void OctreeBatchParser::set_cpu(const void* ptr) {
  const_ptr_ = true;
  h_metadata_ = static_cast<int*>(const_cast<void*>(ptr));
  oct_info_ = static_cast<OctreeInfo*>(const_cast<void*>(ptr));
}

void OctreeBatchParser::set_gpu(const void* ptr, const void* oct_info) {
  const_ptr_ = true;
  // set d_metadata_ address
  d_metadata_ = static_cast<int*>(const_cast<void*>(ptr));

  // oct_info is a host pointer
  if (oct_info == nullptr) {
    // copy octree info form GPU to CPU, store in oct_info_buffer,
    // oct_info_ point to oct_info_buffer.
    oct_info_ = &oct_info_buffer_;
    cudaMemcpy(oct_info_, ptr, sizeof(OctreeInfo), cudaMemcpyDeviceToHost);
  }
  else {
    // set oct_info_ pointing to input octree info address oct_info
    oct_info_ = static_cast<OctreeInfo*>(const_cast<void*>(oct_info));
  }
}

void OctreeBatchParser::set_cpu(void* ptr, const OctreeInfo* oct_info) {
  const_ptr_ = false;
  h_metadata_ = static_cast<int*>(ptr);
  oct_info_ = static_cast<OctreeInfo*>(ptr);
  if (oct_info != nullptr) {
    memcpy(oct_info_, oct_info, sizeof(OctreeInfo));
  }
}

void OctreeBatchParser::set_gpu(void* ptr, const OctreeInfo* oct_info) {
  const_ptr_ = false;
  d_metadata_ = static_cast<int*>(ptr);
  oct_info_ = &oct_info_buffer_;
  if (oct_info != nullptr) {
    memcpy(oct_info_, oct_info, sizeof(OctreeInfo));
    cudaMemcpy(d_metadata_, oct_info_, sizeof(OctreeInfo),
      cudaMemcpyHostToDevice);
  }
  else {
    cudaMemcpy(oct_info_, d_metadata_, sizeof(OctreeInfo),
      cudaMemcpyDeviceToHost);
  }
}

//void OctreeBatchParser::set_gpu(void* data)
//{
//  d_metadata_ = (int*)data;
//
//  // get batch_size and depth from gpu tensor
//  batch_size_ = d_metadata_;
//  int batch_size;
//  cudaMemcpy(&batch_size, batch_size_, sizeof(int), cudaMemcpyDeviceToHost);
//  CHECK(batch_size > 0) << "Invalid octree batch size";
//  depth_ = batch_size_ + 1;
//  int depth;
//  cudaMemcpy(&depth, depth_, sizeof(int), cudaMemcpyDeviceToHost);
//  CHECK(depth < 10 && depth>0) << "Invalid octree depth";
//
//  // copy header_info_ from gpu to cpu
//  int sz = header_sizeofint(depth, batch_size);
//  header_info_.resize(sz);
//  cudaMemcpy(header_info_.data(), d_metadata_, sz * sizeof(int), cudaMemcpyDeviceToHost);
//
//  // get header info from cpu data
//  batch_size_ = header_info_.data();
//  depth_ = batch_size_ + 1;
//  full_layer_ = depth_ + 1;
//  node_num_ = full_layer_ + 1;
//  node_num_cum_ = node_num_ + depth + 1;
//  node_num_nempty_ = node_num_cum_ + depth + 2;
//  node_num_oct_ = node_num_nempty_ + depth + 1;
//  node_num_nempty_oct_ = node_num_oct_ + (depth + 1)*batch_size;
//  int total_nnum = node_num_cum_[depth + 1];
//  CHECK(total_nnum > 0) << "Invalid node number";
//  content_flags_ = node_num_nempty_oct_ + (depth + 1)*batch_size;
//  int content_flags;
//  content_flags = *content_flags_;
//  CHECK(content_flags != 0) << "Invalid flags";
//
//  // set ptr of key/children/neigh on gpu
//  int* ptr = d_metadata_ + sz;
//  key_ = children_ = neigh_ = nullptr;
//  if (0 != (content_flags & 1))
//  {
//    key_ = ptr;
//    ptr += total_nnum;
//  }
//  if (0 != (content_flags & 2))
//  {
//    children_ = ptr;
//    ptr += total_nnum;
//  }
//  if (0 != (content_flags & 4))
//  {
//    neigh_ = ptr;
//  }
//}

//int OctreeBatchParser::header_sizeofint(const int depth, const int batch_size)
//{
//  return 3 + (depth + 1) + (depth + 2) + (depth + 1)
//    + 2 * (depth + 1)*batch_size + 1;
//}
//
//int OctreeBatchParser::octree_batch_sizeofint(const int depth, const int total_nnum,
//  const int batch_size, int content_flags)
//{
//  int sz = header_sizeofint(depth, batch_size);  // header size
//  if (0 != (content_flags & 1)) sz += total_nnum; // key array size
//  if (0 != (content_flags & 2)) sz += total_nnum; // children array size
//  //CHECK_EQ(total_nnum % 8, 1);      // NOTE: only for 3*3*3 neighborhood
//  if (0 != (content_flags & 4)) sz += total_nnum * AVG_NGH_NUM;
//  return sz;
//}

}  // namespace tensorflow
