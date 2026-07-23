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
//   3. Visit active replicated experts in descending token-count order and
//      prefer low-token-load ranks while preserving that optimal capacity.
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

template <int NUM_LOGICAL, int MAX_COPIES, int NUM_PHYSICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
__global__ void dispatch_decode_integral_kernel(
    int32_t* __restrict__ out_topk_ids,
    const int32_t* __restrict__ in_topk_ids,
    const float* __restrict__ global_counts,
    const int64_t* __restrict__ log2phy_map,
    int N) {
  static_assert(NUM_PHYSICAL % NUM_GPUS == 0);
  constexpr int PHYSICAL_PER_GPU = NUM_PHYSICAL / NUM_GPUS;
  constexpr int REPLICATED_STORAGE = NUM_REPLICATED > 0 ? NUM_REPLICATED : 1;

  __shared__ int32_t chosen_physical[NUM_LOGICAL];

  if (threadIdx.x == 0) {
    int fixed_active_load[NUM_GPUS] = {};
    int fixed_token_load[NUM_GPUS] = {};
    int replicated_ids[REPLICATED_STORAGE];
    int replicated_rank[REPLICATED_STORAGE];
    int replicated_count = 0;

    // Single-copy experts are fixed. Replicated experts are collected for
    // integral assignment below. Inactive rows still receive their first
    // valid copy so the map is total, although decode never indexes them.
    for (int logical = 0; logical < NUM_LOGICAL; ++logical) {
      const int64_t* row = log2phy_map + logical * MAX_COPIES;
      const int32_t first_physical = static_cast<int32_t>(row[0]);
      chosen_physical[logical] = first_physical;

      const int count = static_cast<int>(global_counts[logical]);
      if (count <= 0) continue;

      if (MAX_COPIES == 1 || row[1] < 0) {
        const int rank = first_physical / PHYSICAL_PER_GPU;
        ++fixed_active_load[rank];
        fixed_token_load[rank] += count;
      } else {
        replicated_ids[replicated_count++] = logical;
      }
    }

    // Largest token groups go first so the secondary token-load objective is
    // not determined by logical-expert numbering.
    for (int i = 1; i < replicated_count; ++i) {
      const int key = replicated_ids[i];
      const float key_count = global_counts[key];
      int j = i - 1;
      while (j >= 0 && global_counts[replicated_ids[j]] < key_count) {
        replicated_ids[j + 1] = replicated_ids[j];
        --j;
      }
      replicated_ids[j + 1] = key;
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

        // Seed eligible ranks in increasing token-load order. Sorting the
        // replicated experts above and this rank preference form the token
        // count secondary objective.
        const int root_logical = replicated_ids[root];
        const int64_t* root_row = log2phy_map + root_logical * MAX_COPIES;
        for (int seed = 0; seed < NUM_GPUS; ++seed) {
          int best_rank = -1;
          int best_tokens = std::numeric_limits<int>::max();
#pragma unroll 1
          for (int copy = 0; copy < MAX_COPIES; ++copy) {
            const int64_t physical = root_row[copy];
            if (physical < 0) break;
            const int rank = static_cast<int>(physical) / PHYSICAL_PER_GPU;
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
            const int64_t* row = log2phy_map + logical * MAX_COPIES;
#pragma unroll 1
            for (int copy = 0; copy < MAX_COPIES; ++copy) {
              const int64_t physical = row[copy];
              if (physical < 0) break;
              const int alternate = static_cast<int>(physical) / PHYSICAL_PER_GPU;
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
        const int64_t* row = log2phy_map + logical * MAX_COPIES;
#pragma unroll 1
        for (int copy = 0; copy < MAX_COPIES; ++copy) {
          const int64_t physical = row[copy];
          if (physical < 0) break;
          if (static_cast<int>(physical) / PHYSICAL_PER_GPU == assigned_rank) {
            chosen_physical[logical] = static_cast<int32_t>(physical);
            break;
          }
        }
      }
      break;
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
    out_topk_ids[idx] = chosen_physical[in_topk_ids[idx]];
  }
}

template <int NUM_LOGICAL, int MAX_COPIES, int NUM_PHYSICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void dispatch_decode_integral(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView global_counts,
    tvm::ffi::TensorView log2phy_map) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;

  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(global_counts);
  TensorMatcher({NUM_LOGICAL, MAX_COPIES}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(log2phy_map);

  const DLDevice device = device_.unwrap();
  using KernelT = void (*)(int32_t*, const int32_t*, const float*, const int64_t*, int);
  KernelT kernel =
      dispatch_decode_integral_kernel<NUM_LOGICAL, MAX_COPIES, NUM_PHYSICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<const float*>(global_counts.data_ptr()),
      static_cast<const int64_t*>(log2phy_map.data_ptr()),
      static_cast<int>(N.unwrap()));
}

}  // namespace
