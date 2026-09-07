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

#include "../inkling/inkling_ar_barrier.cuh"
#include <cstdint>
#include <limits>

namespace {

template <
    int NUM_LOGICAL,
    int NUM_GPUS,
    int NUM_REPLICATED,
    int BLOCK_DIM,
    bool GATHER_P2P,
    bool METRO_GREEDY,
    bool COUNT_GLOBAL_INPUT>
__global__ void dispatch_decode_integral_kernel(
    int32_t* __restrict__ out_topk_ids,
    const int32_t* __restrict__ in_topk_ids,
    const float* __restrict__ global_counts,
    const int32_t* __restrict__ physical_by_rank,
    const int32_t* __restrict__ rank_mask,
    const int32_t* __restrict__ replicated_logical,
    int N,
    void* const* __restrict__ active_ptrs,
    void* const* __restrict__ flag_ptrs,
    uint32_t* __restrict__ barrier_state,
    uint32_t rank) {
  constexpr int REPLICATED_STORAGE = NUM_REPLICATED > 0 ? NUM_REPLICATED : 1;
  constexpr int ACTIVE_WORDS = (NUM_LOGICAL + 31) / 32;

  __shared__ int32_t chosen_physical[NUM_LOGICAL];
  __shared__ int logical_counts[NUM_LOGICAL];
  __shared__ int fixed_active_load[NUM_GPUS];
  __shared__ int fixed_token_load[NUM_GPUS];
  __shared__ uint32_t global_active[ACTIVE_WORDS];
  // Replicated-expert metadata staged in shared memory by all threads so the
  // serial greedy on thread 0 never waits on a dependent global load.  Before
  // this the METRO kernel spent ~18 us/layer, almost all of it in ~50 chained
  // L2 round trips from one thread.
  __shared__ int rep_logical_s[REPLICATED_STORAGE];
  __shared__ uint32_t rep_mask_s[REPLICATED_STORAGE];
  __shared__ int32_t rep_phys_s[REPLICATED_STORAGE * NUM_GPUS];

  for (int logical = threadIdx.x; logical < NUM_LOGICAL; logical += BLOCK_DIM) {
    if constexpr (GATHER_P2P || COUNT_GLOBAL_INPUT) {
      logical_counts[logical] = 0;
    } else {
      logical_counts[logical] = static_cast<int>(global_counts[logical]);
    }
  }
  if (threadIdx.x < NUM_GPUS) {
    fixed_active_load[threadIdx.x] = 0;
    fixed_token_load[threadIdx.x] = 0;
  }
  __syncthreads();

  if constexpr (COUNT_GLOBAL_INPUT) {
    // The all-gather path already supplies every token on every rank. Count
    // active logical experts in this kernel instead of launching fill,
    // scatter-add, and dtype-conversion kernels from Python.
    for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
      const int logical = in_topk_ids[idx];
      if (logical >= 0) atomicAdd(logical_counts + logical, 1);
    }
    __syncthreads();
  }

  if constexpr (GATHER_P2P) {
    auto* local_active = static_cast<uint32_t*>(active_ptrs[rank]);
    for (int word = threadIdx.x; word < ACTIVE_WORDS; word += BLOCK_DIM) {
      local_active[word] = 0;
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
      const int logical = in_topk_ids[idx];
      if (logical >= 0) {
        atomicOr(local_active + logical / 32, 1u << (logical % 32));
      }
    }

    // Publish this rank's compact active bitset and wait until every peer has
    // published the matching layer invocation. The monotonic device epoch is
    // CUDA-graph replay safe and avoids an NCCL/all-reduce launch.
    inkling_ar::grid_system_barrier<NUM_GPUS>(barrier_state, flag_ptrs, rank, 0, /*publish_writes=*/true);

    // Each 32-bit word describes 32 logical experts. Aggregate every peer
    // word once, rather than issuing the same remote P2P loads independently
    // for all 32 bits in that word.
    for (int word = threadIdx.x; word < ACTIVE_WORDS; word += BLOCK_DIM) {
      uint32_t active = 0;
#pragma unroll
      for (int peer = 0; peer < NUM_GPUS; ++peer) {
        const auto* peer_active = static_cast<const volatile uint32_t*>(active_ptrs[peer]);
        active |= peer_active[word];
      }
      global_active[word] = active;
    }
    __syncthreads();
    for (int logical = threadIdx.x; logical < NUM_LOGICAL; logical += BLOCK_DIM) {
      const uint32_t bit = 1u << (logical % 32);
      logical_counts[logical] = (global_active[logical / 32] & bit) != 0;
    }
    __syncthreads();
  }

  // Compact-map initialization and fixed-load accounting are independent per
  // logical expert. Parallelizing this 256-row scan leaves only the small
  // replicated-expert assignment on thread 0.
  for (int logical = threadIdx.x; logical < NUM_LOGICAL; logical += BLOCK_DIM) {
    const uint32_t mask = static_cast<uint32_t>(rank_mask[logical]);
    const int first_rank = __ffs(static_cast<int>(mask)) - 1;
    chosen_physical[logical] = physical_by_rank[logical * NUM_GPUS + first_rank];

    const int count = logical_counts[logical];
    if (count <= 0) continue;

    if (__popc(mask) == 1) {
      atomicAdd(&fixed_active_load[first_rank], 1);
      atomicAdd(&fixed_token_load[first_rank], count);
    }
  }
  for (int i = threadIdx.x; i < NUM_REPLICATED; i += BLOCK_DIM) {
    const int logical = replicated_logical[i];
    rep_logical_s[i] = logical;
    rep_mask_s[i] = static_cast<uint32_t>(rank_mask[logical]);
#pragma unroll
    for (int r = 0; r < NUM_GPUS; ++r) {
      rep_phys_s[i * NUM_GPUS + r] = physical_by_rank[logical * NUM_GPUS + r];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if constexpr (METRO_GREEDY) {
      // METRO Algorithm 1.  Logical-id order is a deterministic legal
      // serialization of the paper's candidate-rank locks, so every EP rank
      // independently reaches the same assignment from the global active set.
      int active_load[NUM_GPUS];
      for (int rank = 0; rank < NUM_GPUS; ++rank) {
        active_load[rank] = fixed_active_load[rank];
      }
      for (int i = 0; i < NUM_REPLICATED; ++i) {
        const int logical = rep_logical_s[i];
        if (logical_counts[logical] <= 0) continue;
        uint32_t eligible = rep_mask_s[i];
        int best_rank = -1;
        int best_load = NUM_LOGICAL + 1;
        while (eligible != 0) {
          const int rank = __ffs(static_cast<int>(eligible)) - 1;
          eligible &= eligible - 1;
          if (active_load[rank] < best_load ||
              (active_load[rank] == best_load && rank < best_rank)) {
            best_rank = rank;
            best_load = active_load[rank];
          }
        }
        if (best_rank >= 0) {
          chosen_physical[logical] = rep_phys_s[i * NUM_GPUS + best_rank];
          ++active_load[best_rank];
        }
      }
    } else {
    int replicated_ids[REPLICATED_STORAGE];
    int replicated_rank[REPLICATED_STORAGE];
    int replicated_count = 0;

    // The complete replicated-logical list is static and compacted once at
    // solver init. Preserve its logical-id order and skip inactive entries.
    // The exact primary objective does not depend on the visit order.
    for (int i = 0; i < NUM_REPLICATED; ++i) {
      const int logical = replicated_logical[i];
      if (logical_counts[logical] > 0) {
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
          const int count = logical_counts[replicated_ids[expert]];
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
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
    const int logical = in_topk_ids[idx];
    out_topk_ids[idx] = logical >= 0 ? chosen_physical[logical] : -1;
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
  auto kernel = dispatch_decode_integral_kernel<
      NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, false, false, false>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<const float*>(global_counts.data_ptr()),
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(rank_mask.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<int>(N.unwrap()),
      nullptr,
      nullptr,
      nullptr,
      0u);
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void dispatch_decode_metro(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView global_counts,
    tvm::ffi::TensorView physical_by_rank,
    tvm::ffi::TensorView rank_mask,
    tvm::ffi::TensorView replicated_logical) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL}).with_dtype<float>().with_device<kDLCUDA>(device_)
      .verify(global_counts);
  TensorMatcher({NUM_LOGICAL, NUM_GPUS}).with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_).verify(physical_by_rank);
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(rank_mask);
  TensorMatcher({NUM_REPLICATED}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(replicated_logical);

  const DLDevice device = device_.unwrap();
  auto kernel = dispatch_decode_integral_kernel<
      NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, false, true, false>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<const float*>(global_counts.data_ptr()),
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(rank_mask.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<int>(N.unwrap()),
      nullptr,
      nullptr,
      nullptr,
      0u);
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void dispatch_decode_metro_global(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView physical_by_rank,
    tvm::ffi::TensorView rank_mask,
    tvm::ffi::TensorView replicated_logical) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL, NUM_GPUS}).with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_).verify(physical_by_rank);
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(rank_mask);
  TensorMatcher({NUM_REPLICATED}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(replicated_logical);

  const DLDevice device = device_.unwrap();
  auto kernel = dispatch_decode_integral_kernel<
      NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, false, true, true>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      nullptr,
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(rank_mask.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<int>(N.unwrap()),
      nullptr,
      nullptr,
      nullptr,
      0u);
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void dispatch_decode_integral_p2p(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView local_active,
    tvm::ffi::TensorView physical_by_rank,
    tvm::ffi::TensorView rank_mask,
    tvm::ffi::TensorView replicated_logical,
    int64_t active_ptrs_dev,
    int64_t flag_ptrs_dev,
    int64_t barrier_state_ptr,
    int64_t rank) {
  using namespace host;

  constexpr int ACTIVE_WORDS = (NUM_LOGICAL + 31) / 32;
  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;

  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({ACTIVE_WORDS}).with_dtype<uint32_t>().with_device<kDLCUDA>(device_).verify(local_active);
  TensorMatcher({NUM_LOGICAL, NUM_GPUS}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(physical_by_rank);
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(rank_mask);
  TensorMatcher({NUM_REPLICATED}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(replicated_logical);
  RuntimeCheck(active_ptrs_dev != 0, "active_ptrs_dev is null");
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(barrier_state_ptr != 0, "barrier_state_ptr is null");
  RuntimeCheck(rank >= 0 && rank < NUM_GPUS, "rank is out of range");

  const DLDevice device = device_.unwrap();
  auto kernel = dispatch_decode_integral_kernel<
      NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, true, false, false>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      nullptr,
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(rank_mask.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<int>(N.unwrap()),
      reinterpret_cast<void* const*>(active_ptrs_dev),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(barrier_state_ptr),
      static_cast<uint32_t>(rank));
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void dispatch_decode_metro_p2p(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView local_active,
    tvm::ffi::TensorView physical_by_rank,
    tvm::ffi::TensorView rank_mask,
    tvm::ffi::TensorView replicated_logical,
    int64_t active_ptrs_dev,
    int64_t flag_ptrs_dev,
    int64_t barrier_state_ptr,
    int64_t rank) {
  using namespace host;

  constexpr int ACTIVE_WORDS = (NUM_LOGICAL + 31) / 32;
  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({ACTIVE_WORDS}).with_dtype<uint32_t>().with_device<kDLCUDA>(device_)
      .verify(local_active);
  TensorMatcher({NUM_LOGICAL, NUM_GPUS}).with_dtype<int32_t>()
      .with_device<kDLCUDA>(device_).verify(physical_by_rank);
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(rank_mask);
  TensorMatcher({NUM_REPLICATED}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(replicated_logical);
  RuntimeCheck(active_ptrs_dev != 0, "active_ptrs_dev is null");
  RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
  RuntimeCheck(barrier_state_ptr != 0, "barrier_state_ptr is null");
  RuntimeCheck(rank >= 0 && rank < NUM_GPUS, "rank is out of range");

  const DLDevice device = device_.unwrap();
  auto kernel = dispatch_decode_integral_kernel<
      NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, true, true, false>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      nullptr,
      static_cast<const int32_t*>(physical_by_rank.data_ptr()),
      static_cast<const int32_t*>(rank_mask.data_ptr()),
      static_cast<const int32_t*>(replicated_logical.data_ptr()),
      static_cast<int>(N.unwrap()),
      reinterpret_cast<void* const*>(active_ptrs_dev),
      reinterpret_cast<void* const*>(flag_ptrs_dev),
      reinterpret_cast<uint32_t*>(barrier_state_ptr),
      static_cast<uint32_t>(rank));
}

template <int NUM_LOGICAL, int BLOCK_DIM>
__global__ void count_logical_f32_kernel(
    float* __restrict__ out_counts,
    const int32_t* __restrict__ in_topk_ids,
    int n) {
  __shared__ int counts[NUM_LOGICAL];
  for (int i = threadIdx.x; i < NUM_LOGICAL; i += BLOCK_DIM) counts[i] = 0;
  __syncthreads();
  for (int idx = threadIdx.x; idx < n; idx += BLOCK_DIM) {
    const int logical = in_topk_ids[idx];
    if (logical >= 0 && logical < NUM_LOGICAL) atomicAdd(counts + logical, 1);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < NUM_LOGICAL; i += BLOCK_DIM) {
    out_counts[i] = static_cast<float>(counts[i]);
  }
}

// One launch producing the float32 per-logical-expert token count that the EP
// all-reduce consumes.  Replaces the fill / scatter_add / dtype-cast trio from
// Python (three launches plus their graph-replay gaps).
template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void count_logical_f32(
    tvm::ffi::TensorView out_counts,
    tvm::ffi::TensorView in_topk_ids) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(out_counts);
  const DLDevice device = device_.unwrap();
  auto kernel = count_logical_f32_kernel<NUM_LOGICAL, BLOCK_DIM>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<float*>(out_counts.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<int>(N.unwrap()));
}


// ---------------------------------------------------------------------------
// METRO v2: warp-parallel greedy over precomputed static tables.
//
// Same assignment as dispatch_decode_metro (METRO Algorithm 1 in logical-id
// order, min active load, lowest rank on ties), but
//   * every static per-layer table (default replica, single-copy rank,
//     replicated-expert list/mask/physical ids) is precomputed on the host and
//     read with coalesced loads instead of chained dependent gathers;
//   * the per-expert argmin over eligible ranks is a 5-step warp shuffle
//     reduction (lane r owns rank r's load) instead of a serial ffs loop on one
//     thread.  R=128 went from ~24 us to a few us per layer.
// WRITE_LOCAL_COUNTS: after routing with the counts currently in `counts`
// (the previous decode step's reduced global counts), overwrite `counts`
// with this rank's local counts for the current step, so a side-stream
// all-reduce can produce the next step's global counts off the critical path.
// ---------------------------------------------------------------------------
template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM, bool WRITE_LOCAL_COUNTS>
__global__ void metro_route_v2_kernel(
    int32_t* __restrict__ out_topk_ids,
    const int32_t* __restrict__ in_topk_ids,
    float* __restrict__ counts,
    const int32_t* __restrict__ default_physical,
    const int32_t* __restrict__ single_rank,
    const int32_t* __restrict__ rep_logical,
    const int32_t* __restrict__ rep_mask,
    const int32_t* __restrict__ rep_phys,
    int N,
    int inactive_weight) {
  static_assert(NUM_GPUS <= 32, "warp argmin covers at most 32 ranks");
  static_assert(BLOCK_DIM >= 32, "greedy runs on warp 0");
  constexpr int RS = NUM_REPLICATED > 0 ? NUM_REPLICATED : 1;

  __shared__ int32_t chosen_physical[NUM_LOGICAL];
  __shared__ int32_t active_s[NUM_LOGICAL];
  __shared__ int local_counts_s[NUM_LOGICAL];
  __shared__ int fixed_active_load[NUM_GPUS];
  __shared__ int rep_logical_s[RS];
  __shared__ uint32_t rep_mask_s[RS];
  __shared__ int32_t rep_phys_s[RS * NUM_GPUS];
  __shared__ int active_rep_idx[RS];

  if (threadIdx.x < NUM_GPUS) fixed_active_load[threadIdx.x] = 0;
  for (int l = threadIdx.x; l < NUM_LOGICAL; l += BLOCK_DIM) local_counts_s[l] = 0;
  __syncthreads();

  for (int l = threadIdx.x; l < NUM_LOGICAL; l += BLOCK_DIM) {
    const bool active = counts[l] > 0.0f;
    active_s[l] = active ? 1 : 0;
    chosen_physical[l] = default_physical[l];
    const int sr = single_rank[l];
    if (active && sr >= 0) atomicAdd(&fixed_active_load[sr], 1);
  }
  for (int i = threadIdx.x; i < NUM_REPLICATED; i += BLOCK_DIM) {
    rep_logical_s[i] = rep_logical[i];
    rep_mask_s[i] = static_cast<uint32_t>(rep_mask[i]);
  }
  for (int i = threadIdx.x; i < NUM_REPLICATED * NUM_GPUS; i += BLOCK_DIM) {
    rep_phys_s[i] = rep_phys[i];
  }
  if constexpr (WRITE_LOCAL_COUNTS) {
    for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
      const int l = in_topk_ids[idx];
      if (l >= 0 && l < NUM_LOGICAL) atomicAdd(local_counts_s + l, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    // Order-preserving compaction of the active replicated experts so the
    // serial greedy below has no divergent skips and a known trip count.
    int num_active = 0;
    for (int c = 0; c < NUM_REPLICATED; c += 32) {
      const int i = c + lane;
      const bool a = (i < NUM_REPLICATED) && (active_s[rep_logical_s[i]] != 0);
      const uint32_t b = __ballot_sync(0xffffffffu, a);
      if (a) active_rep_idx[num_active + __popc(b & ((1u << lane) - 1u))] = i;
      num_active += __popc(b);
    }
    // METRO greedy.  Lane r owns rank r's active load; the per-expert argmin
    // over eligible ranks is one warp-wide min (redux.sync on sm_80+), with
    // (load, rank) packed so ties break to the lowest rank exactly as v1.
    // Loads are kept in eighths so a second pass can place experts that were
    // inactive in the (stale) count with a fractional expected weight.
    int load = ((lane < NUM_GPUS) ? fixed_active_load[lane] : 0) * 8;
    if (inactive_weight > 0) {
      // Append the inactive replicated experts after the active ones; the
      // greedy below places them on the least-loaded eligible rank with weight
      // inactive_weight/8 so that experts newly active this step land where
      // there is slack instead of all defaulting to their first replica.
      int base = num_active;
      for (int c = 0; c < NUM_REPLICATED; c += 32) {
        const int i = c + lane;
        const bool a = (i < NUM_REPLICATED) && (active_s[rep_logical_s[i]] == 0);
        const uint32_t b = __ballot_sync(0xffffffffu, a);
        if (a) active_rep_idx[base + __popc(b & ((1u << lane) - 1u))] = i;
        base += __popc(b);
      }
    }
    const int num_pass = inactive_weight > 0 ? NUM_REPLICATED : num_active;
    // Software-pipelined: the expert index / mask / my physical id for step
    // k+1 do not depend on the loads, so fetch them while step k reduces.
    int i_next = num_pass > 0 ? active_rep_idx[0] : 0;
    uint32_t mask_next = rep_mask_s[i_next];
    int32_t phys_next = (lane < NUM_GPUS) ? rep_phys_s[i_next * NUM_GPUS + lane] : -1;
    for (int k = 0; k < num_pass; ++k) {
      const int i = i_next;
      const uint32_t mask = mask_next;
      const int32_t my_phys = phys_next;
      const int weight = k < num_active ? 8 : inactive_weight;
      if (k + 1 < num_pass) {
        i_next = active_rep_idx[k + 1];
        mask_next = rep_mask_s[i_next];
        phys_next = (lane < NUM_GPUS) ? rep_phys_s[i_next * NUM_GPUS + lane] : -1;
      }
      const bool eligible = (lane < NUM_GPUS) && (((mask >> lane) & 1u) != 0u);
      const uint32_t key = eligible ? ((static_cast<uint32_t>(load) << 5) | lane) : 0xffffffffu;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      const uint32_t best_key = __reduce_min_sync(0xffffffffu, key);
#else
      uint32_t best_key = key;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        best_key = min(best_key, __shfl_xor_sync(0xffffffffu, best_key, off));
      }
#endif
      // A replicated expert always has >= 2 eligible ranks, so best_key is a
      // real (load, rank) key; the winning lane records its own physical id.
      if (key == best_key) {
        load += weight;
        chosen_physical[rep_logical_s[i]] = my_phys;
      }
    }
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < N; idx += BLOCK_DIM) {
    const int l = in_topk_ids[idx];
    out_topk_ids[idx] = l >= 0 ? chosen_physical[l] : -1;
  }
  if constexpr (WRITE_LOCAL_COUNTS) {
    for (int l = threadIdx.x; l < NUM_LOGICAL; l += BLOCK_DIM) {
      counts[l] = static_cast<float>(local_counts_s[l]);
    }
  }
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM, bool WRITE_LOCAL_COUNTS>
void metro_route_v2_impl(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView default_physical,
    tvm::ffi::TensorView single_rank,
    tvm::ffi::TensorView rep_logical,
    tvm::ffi::TensorView rep_mask,
    tvm::ffi::TensorView rep_phys,
    int64_t inactive_weight) {
  using namespace host;

  SymbolicSize N{"num_topk_entries"};
  SymbolicDevice device_;
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(out_topk_ids).verify(in_topk_ids);
  TensorMatcher({NUM_LOGICAL}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(counts);
  RuntimeCheck(inactive_weight >= 0 && inactive_weight <= 8, "inactive_weight must be in [0, 8]");
  TensorMatcher({NUM_LOGICAL}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(default_physical).verify(single_rank);
  TensorMatcher({NUM_REPLICATED}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(rep_logical).verify(rep_mask);
  TensorMatcher({NUM_REPLICATED * NUM_GPUS}).with_dtype<int32_t>().with_device<kDLCUDA>(device_)
      .verify(rep_phys);

  const DLDevice device = device_.unwrap();
  auto kernel = metro_route_v2_kernel<NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, WRITE_LOCAL_COUNTS>;
  LaunchKernel(/*grid=*/1, /*block=*/BLOCK_DIM, device)(
      kernel,
      static_cast<int32_t*>(out_topk_ids.data_ptr()),
      static_cast<const int32_t*>(in_topk_ids.data_ptr()),
      static_cast<float*>(counts.data_ptr()),
      static_cast<const int32_t*>(default_physical.data_ptr()),
      static_cast<const int32_t*>(single_rank.data_ptr()),
      static_cast<const int32_t*>(rep_logical.data_ptr()),
      static_cast<const int32_t*>(rep_mask.data_ptr()),
      static_cast<const int32_t*>(rep_phys.data_ptr()),
      static_cast<int>(N.unwrap()),
      static_cast<int>(inactive_weight));
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void metro_route_v2(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView default_physical,
    tvm::ffi::TensorView single_rank,
    tvm::ffi::TensorView rep_logical,
    tvm::ffi::TensorView rep_mask,
    tvm::ffi::TensorView rep_phys,
    int64_t inactive_weight) {
  metro_route_v2_impl<NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, false>(
      out_topk_ids, in_topk_ids, counts, default_physical, single_rank, rep_logical, rep_mask, rep_phys, inactive_weight);
}

template <int NUM_LOGICAL, int NUM_GPUS, int NUM_REPLICATED, int BLOCK_DIM>
void metro_route_stale(
    tvm::ffi::TensorView out_topk_ids,
    tvm::ffi::TensorView in_topk_ids,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView default_physical,
    tvm::ffi::TensorView single_rank,
    tvm::ffi::TensorView rep_logical,
    tvm::ffi::TensorView rep_mask,
    tvm::ffi::TensorView rep_phys,
    int64_t inactive_weight) {
  metro_route_v2_impl<NUM_LOGICAL, NUM_GPUS, NUM_REPLICATED, BLOCK_DIM, true>(
      out_topk_ids, in_topk_ids, counts, default_physical, single_rank, rep_logical, rep_mask, rep_phys, inactive_weight);
}

}  // namespace
