// Deterministic LPLB decode dispatch.
//
// Every occurrence of a logical expert selects the same physical replica:
// the valid replica with the largest LP weight. This prevents a decode chunk
// from activating multiple physical copies of one logical expert.
//
// logical_to_physical is padded with -1 and valid copies are contiguous, so
// the scan normally reads only 1-2 entries even though MAX_COPIES is padded to
// the total physical-expert count.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

template <int MAX_COPIES, int BLOCK_DIM>
__global__ void dispatch_active_experts_kernel(
    int32_t* __restrict__ out_topk_ids,
    const int32_t* __restrict__ in_topk_ids,
    const float* __restrict__ log2phy_prob,
    const int64_t* __restrict__ log2phy_map,
    int N) {
  const int idx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  if (idx >= N) return;

  const int32_t logical_id = in_topk_ids[idx];
  const int64_t* row_map = log2phy_map + logical_id * MAX_COPIES;
  const float* row_prob = log2phy_prob + logical_id * MAX_COPIES;

  float best_prob = -3.402823466e+38F;
  int32_t best_physical = -1;
#pragma unroll 1
  for (int copy = 0; copy < MAX_COPIES; ++copy) {
    const int64_t physical = row_map[copy];
    if (physical < 0) break;
    const float probability = row_prob[copy];
    if (probability > best_prob) {
      best_prob = probability;
      best_physical = static_cast<int32_t>(physical);
    }
  }
  out_topk_ids[idx] = best_physical;
}

template <int MAX_COPIES, int BLOCK_DIM>
void dispatch_active_experts(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView log2phy_prob,
    tvm::ffi::TensorView log2phy_map) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicSize NUM_LOGICAL{"num_logical"};
  SymbolicDevice device_;

  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL, MAX_COPIES}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(log2phy_prob);
  TensorMatcher({NUM_LOGICAL, MAX_COPIES}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(log2phy_map);

  const int n = static_cast<int>(N.unwrap());
  const int grid = (n + BLOCK_DIM - 1) / BLOCK_DIM;
  const DLDevice device = device_.unwrap();

  using KernelT = void (*)(int32_t*, const int32_t*, const float*, const int64_t*, int);
  KernelT kernel = dispatch_active_experts_kernel<MAX_COPIES, BLOCK_DIM>;
  LaunchKernel(grid, BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<const float*>(log2phy_prob.data_ptr()),
      static_cast<const int64_t*>(log2phy_map.data_ptr()),
      n);
}

}  // namespace
