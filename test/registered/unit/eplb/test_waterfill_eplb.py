"""Unit tests for Waterfill and EPLB updater compatibility."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe import waterfill as waterfill_module
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.layers.moe.waterfill import WaterfillBalancer
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase


class _FakeExpertParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_local_experts = 5
        self.weight = nn.Parameter(
            torch.arange(10, dtype=torch.float32).reshape(self.num_local_experts, 2)
        )
        self.correction_bias = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale = nn.Parameter(torch.ones(self.num_local_experts))
        self.global_scale._sglang_require_global_experts = True


class TestWaterfillEPLB(CustomTestCase):
    def test_small_prefill_keeps_existing_local_fast_path(self):
        balancer = WaterfillBalancer.__new__(WaterfillBalancer)
        balancer.use_static_waterfill = True
        topk_output = topk_module.StandardTopKOutput(
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.zeros((2, 2), dtype=torch.int32),
            router_logits=torch.empty((2, 0)),
        )

        with (
            patch.object(
                waterfill_module, "get_is_extend_in_batch", return_value=False
            ),
            patch.object(
                balancer, "_expand_local_shared", return_value=topk_output
            ) as expand_local,
            patch.object(balancer, "_build_dispatch_plan") as build_plan,
        ):
            actual = balancer.expand_topk(topk_output, num_tokens=2, is_decode=False)

        self.assertIs(actual, topk_output)
        expand_local.assert_called_once_with(topk_output)
        build_plan.assert_not_called()

    def test_small_decode_uses_active_expert_dispatch_plan(self):
        balancer = WaterfillBalancer.__new__(WaterfillBalancer)
        balancer.use_static_waterfill = True
        topk_output = topk_module.StandardTopKOutput(
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.zeros((2, 2), dtype=torch.int32),
            router_logits=torch.empty((2, 0)),
        )

        with (
            patch.object(
                balancer, "_build_dispatch_plan", return_value=None
            ) as build_plan,
            patch.object(
                balancer, "_expand_local_shared", return_value=topk_output
            ) as expand_local,
        ):
            actual = balancer.expand_topk(topk_output, num_tokens=2, is_decode=True)

        self.assertIs(actual, topk_output)
        build_plan.assert_called_once_with(
            topk_output.topk_ids,
            2,
            minimize_active_experts=True,
        )
        expand_local.assert_called_once_with(topk_output)

    def test_dynamic_decode_unions_active_experts_across_ep_group(self):
        balancer = WaterfillBalancer.__new__(WaterfillBalancer)
        balancer.use_static_waterfill = False
        balancer.num_routed_experts = 16
        balancer.world_size = 4
        balancer._active_experts_buf = torch.tensor([1, 0, 0, 0] * 4, dtype=torch.int32)
        local_rank_load = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
        global_active_experts = torch.tensor([2, 0, 1, 0] * 4, dtype=torch.int32)
        global_rank_load = torch.tensor([2, 2, 2, 2], dtype=torch.int64)
        topk_ids = torch.zeros((2, 2), dtype=torch.int32)

        with (
            patch.object(
                balancer,
                "count_local_active_experts",
                return_value=local_rank_load,
            ),
            patch.object(
                balancer,
                "_all_reduce_active_experts",
                return_value=global_active_experts,
            ) as all_reduce,
            patch.object(
                waterfill_module,
                "count_marked_experts_per_rank",
                return_value=global_rank_load,
            ) as count_global,
            patch.object(
                balancer,
                "_get_counts_buf",
                return_value=torch.empty(4, dtype=torch.int64),
            ),
        ):
            plan = balancer._build_dispatch_plan(
                topk_ids,
                num_tokens=2,
                minimize_active_experts=True,
            )

        all_reduce.assert_called_once_with(balancer._active_experts_buf)
        count_global.assert_called_once()
        self.assertIs(plan.rank_load, global_rank_load)
        self.assertTrue(plan.allow_all_ranks)
        self.assertTrue(plan.minimize_active_experts)
        self.assertFalse(plan.fuse_active_expert_load)

    def test_static_decode_fuses_chunk_active_expert_count(self):
        balancer = WaterfillBalancer.__new__(WaterfillBalancer)
        balancer.use_static_waterfill = True
        balancer.world_size = 8
        balancer.old_experts_per_rank = 32
        balancer._counts_buf = None
        topk_ids = torch.zeros((64, 8), dtype=torch.int32)

        with patch.object(balancer, "count_local_active_experts") as count_active:
            plan = balancer._build_dispatch_plan(
                topk_ids,
                num_tokens=64,
                minimize_active_experts=True,
            )

        count_active.assert_not_called()
        self.assertTrue(plan.allow_all_ranks)
        self.assertTrue(plan.minimize_active_experts)
        self.assertTrue(plan.fuse_active_expert_load)

    def test_deepseek_moe_get_moe_weights_excludes_fused_shared_slot(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=1, experts=experts)
        shared_before = experts.weight.data[-1].clone()

        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(
            weights[0].shape,
            (experts.num_local_experts - moe.num_fused_shared_experts, 2),
        )

        weights[0][-1].zero_()
        self.assertTrue(torch.equal(experts.weight.data[-2], torch.zeros(2)))
        self.assertTrue(torch.equal(experts.weight.data[-1], shared_before))

    def test_deepseek_moe_get_moe_weights_keeps_full_shape_without_fusion(self):
        experts = _FakeExpertParam()
        moe = SimpleNamespace(num_fused_shared_experts=0, experts=experts)
        weights = DeepseekV2MoE.get_moe_weights(moe)

        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0].shape, (experts.num_local_experts, 2))

    def test_topk_recorder_ids_exclude_per_rank_fused_shared_slots(self):
        topk_ids = torch.tensor([[0, 33, 263, 256]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=1.0,
        )
        dispatch_info = SimpleNamespace(num_physical_experts=264)

        def fake_eplb_postprocess(
            ids, expert_location_dispatch_info, num_token_non_padded
        ):
            return ids

        with (
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=True
            ),
            get_parallel().override(moe_ep_size=8, moe_ep_rank=7),
            patch.object(
                topk_module,
                "_biased_grouped_topk_postprocess",
                side_effect=fake_eplb_postprocess,
            ),
        ):
            processed_ids, _, recorder_ids = topk_module._post_process_topk_ids(
                topk_ids=topk_ids.clone(),
                topk_weights=topk_weights.clone(),
                topk_config=topk_config,
                router_logits=torch.empty((1, 256)),
                layer_id=0,
                expert_location_dispatch_info=dispatch_info,
            )

        self.assertTrue(torch.equal(processed_ids, torch.tensor([[0, 34, 270, 271]])))
        self.assertTrue(torch.equal(recorder_ids, torch.tensor([[0, 33, 263]])))

    def test_topk_recorder_ids_match_dispatch_ids_without_per_rank_shared_slots(self):
        topk_ids = torch.tensor([[0, 33, 263, 256]], dtype=torch.int32)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=1.0,
        )
        dispatch_info = SimpleNamespace(num_physical_experts=264)

        def fake_eplb_postprocess(
            ids, expert_location_dispatch_info, num_token_non_padded
        ):
            return ids + 1

        with (
            patch.object(topk_module, "_is_cuda", True),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=False
            ),
            patch.object(
                topk_module,
                "_biased_grouped_topk_postprocess",
                side_effect=fake_eplb_postprocess,
            ),
        ):
            processed_ids, _, recorder_ids = topk_module._post_process_topk_ids(
                topk_ids=topk_ids.clone(),
                topk_weights=topk_weights.clone(),
                topk_config=topk_config,
                router_logits=torch.empty((1, 256)),
                layer_id=0,
                expert_location_dispatch_info=dispatch_info,
            )

        self.assertTrue(torch.equal(processed_ids, torch.tensor([[1, 34, 264, 257]])))
        self.assertTrue(torch.equal(recorder_ids, processed_ids))


if __name__ == "__main__":
    unittest.main()
