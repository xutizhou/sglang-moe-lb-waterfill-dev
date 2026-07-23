// Integral decode assignment for LPLB.
//
// The LP relaxation is useful for probability-based token dispatch, but
// decode must send every occurrence of a logical expert to one physical copy
// or it activates extra expert weights. This kernel performs the integral
// assignment directly:
//
//   1. Count single-copy experts as fixed per-rank load.
//   2. Find the minimum feasible per-rank active-expert capacity using an
//      augmenting-path bipartite assignment.
//   3. Visit active replicated experts in logical-id order and prefer
//      low-token-load ranks while preserving that optimal capacity.
//   4. Map all routed top-k entries through the resulting assignment.
//
// The assignment and mapping share one block and one kernel launch. Typical
// DeepSeek-V3 decode has 256 logical experts, 32 replicated experts, and 256
// routed entries (batch 32, top-k 8).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace {

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
__global__ void dispatch_decode_integral_kernel(
    int32_t* __restrict__ out_topk_ids,
    const int32_t* __restrict__ in_topk_ids,
    const float* __restrict__ global_counts,
    const int32_t* __restrict__ physical_by_rank,
    const int32_t* __restrict__ rank_mask,
    const int32_t* __restrict__ replicated_logical,
    int N) {
  constexpr int REPLICATED_STORAGE = NUM_REPLICATED > 0 ? NUM_REPLICATED : 1;

  __shared__ int32_t chosen_physical[NUM_LOGICAL];
  __shared__ int fixed_active_load[NUM_GPUS];
  __shared__ int fixed_token_load[NUM_GPUS];

  if (threadIdx.x < NUM_GPUS) {
    fixed_active_load[threadIdx.x] = 0;
    fixed_token_load[threadIdx.x] = 0;
  }
  __syncthreads();

  // Compact-map initialization and fixed-load accounting are independent per
  // logical expert. Parallelizing this 256-row scan leaves only the small
  // replicated-expert assignment on thread 0.
  for (int logical = threadIdx.x; logical < NUM_LOGICAL; logical += BLOCK_DIM) {
    const uint32_t mask = static_cast<uint32_t>(rank_mask[logical]);
    const int first_rank = __ffs(static_cast<int>(mask)) - 1;
    chosen_physical[logical] = physical_by_rank[logical * NUM_GPUS + first_rank];

    const int count = static_cast<int>(global_counts[logical]);
    if (count <= 0) continue;

    if (__popc(mask) == 1) {
      atomicAdd(&fixed_active_load[first_rank], 1);
      atomicAdd(&fixed_token_load[first_rank], count);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int replicated_ids[REPLICATED_STORAGE];
    int replicated_rank[REPLICATED_STORAGE];
    int replicated_count = 0;

    // The complete replicated-logical list is static and compacted once at
    // solver init. Preserve its logical-id order and skip inactive entries.
    // The exact primary objective does not depend on the visit order.
    for (int i = 0; i < NUM_REPLICATED; ++i) {
      const int logical = replicated_logical[i];
      if (global_counts[logical] > 0) {
        replicated_ids[replicated_count++] = logical;
      }
    }

    int minimum_capacity = 0;
    for (int rank = 0; rank < NUM_GPUS; ++rank) {
      minimum_capacity = max(minimum_capacity, fixed_active_load[rank]);
    }

    // Increase the rank capacity until every replicated expert can be placed.
    // The first feasible value is the exact minimum of the maximum active
    // expert count. An augmenting path may move previously placed experts.
    for (int capacity = minimum_capacity; capacity <= minimum_capacity + replicated_count; ++capacity) {
      int active_load[NUM_GPUS];
      int token_load[NUM_GPUS];
      for (int rank = 0; rank < NUM_GPUS; ++rank) {
        active_load[rank] = fixed_active_load[rank];
        token_load[rank] = fixed_token_load[rank];
      }
      for (int i = 0; i < replicated_count; ++i) {
        replicated_rank[i] = -1;
      }

      bool feasible = true;
      for (int root = 0; root < replicated_count && feasible; ++root) {
        int queue[NUM_GPUS];
        int parent_expert[NUM_GPUS];
        int parent_rank[NUM_GPUS];
        bool visited[NUM_GPUS] = {};
        int queue_head = 0;
        int queue_tail = 0;

        // Seed eligible ranks in increasing token-load order. This preference
        // is a secondary objective after the exact active-expert capacity.
        const int root_logical = replicated_ids[root];
        const uint32_t root_mask = static_cast<uint32_t>(rank_mask[root_logical]);
        for (int seed = 0; seed < NUM_GPUS; ++seed) {
          int best_rank = -1;
          int best_tokens = std::numeric_limits<int>::max();
          uint32_t eligible = root_mask;
          while (eligible != 0) {
            const int rank = __ffs(static_cast<int>(eligible)) - 1;
            eligible &= eligible - 1;
            if (!visited[rank] &&
                (token_load[rank] < best_tokens || (token_load[rank] == best_tokens && rank < best_rank))) {
              best_rank = rank;
              best_tokens = token_load[rank];
            }
          }
          if (best_rank < 0) break;
          visited[best_rank] = true;
          parent_expert[best_rank] = root;
          parent_rank[best_rank] = -1;
          queue[queue_tail++] = best_rank;
        }

        int free_rank = -1;
        while (queue_head < queue_tail && free_rank < 0) {
          const int rank = queue[queue_head++];
          if (active_load[rank] < capacity) {
            free_rank = rank;
            break;
          }

          // Follow alternating edges: every expert currently assigned to this
          // full rank can move to any of its other eligible ranks.
          for (int expert = 0; expert < root; ++expert) {
            if (replicated_rank[expert] != rank) continue;
            const int logical = replicated_ids[expert];
            uint32_t alternates = static_cast<uint32_t>(rank_mask[logical]);
            while (alternates != 0) {
              const int alternate = __ffs(static_cast<int>(alternates)) - 1;
              alternates &= alternates - 1;
              if (visited[alternate]) continue;
              visited[alternate] = true;
              parent_expert[alternate] = expert;
              parent_rank[alternate] = rank;
              queue[queue_tail++] = alternate;
            }
          }
        }

        if (free_rank < 0) {
          feasible = false;
          break;
        }

        // Reverse the alternating path, updating both active and token loads.
        int destination = free_rank;
        while (destination >= 0) {
          const int expert = parent_expert[destination];
          const int old_rank = replicated_rank[expert];
          const int count = static_cast<int>(global_counts[replicated_ids[expert]]);
          if (old_rank >= 0) {
            --active_load[old_rank];
            token_load[old_rank] -= count;
          }
          replicated_rank[expert] = destination;
          ++active_load[destination];
          token_load[destination] += count;
          destination = parent_rank[destination];
        }
      }

      if (!feasible) continue;

      for (int expert = 0; expert < replicated_count; ++expert) {
        const int logical = replicated_ids[expert];
        const int assigned_rank = replicated_rank[expert];
        chosen_physical[logical] = physical_by_rank[logical * NUM_GPUS + assigned_rank];
      }
      break;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
    out_topk_ids[idx] = chosen_physical[in_topk_ids[idx]];
  }
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void dispatch_decode_integral(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView global_counts,
    tvm::ffi::TensorView physical_by_rank,
    tvm::ffi::TensorView rank_mask,
    tvm::ffi::TensorView replicated_logical) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;

  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(global_counts);
  TensorMatcher({NUM_LOGICAL, NUM_GPUS}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(physical_by_rank);
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(rank_mask);
  TensorMatcher({NUM_REPLICATED}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(replicated_logical);

  const DLDevice device = device_.unwrap();
  using KernelT = void (*)(int32_t*, const int32_t*, const float*, const int32_t*, const int32_t*, const int32_t*, int);
  KernelT kernel = dispatch_decode_integral_kernel<NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<const float*>(global_counts.data_ptr()),
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(rank_mask.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<int>(N.unwrap()));
}

}  // namespace
