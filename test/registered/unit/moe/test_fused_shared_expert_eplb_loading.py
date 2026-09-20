from unittest import TestCase, mock

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class TestFusedSharedExpertEPLBLoading(TestCase):
    def _load_shared(
        self,
        *,
        require_global_experts,
        per_rank_slots,
        expert_id=256,
    ):
        moe = object.__new__(FusedMoE)
        moe._has_fused_shared = True
        moe._num_global_routed = 288
        moe._num_local_routed = 18
        moe.num_local_experts = 19
        moe.num_fused_shared_experts = 1
        moe.moe_ep_size = 16
        moe.layer_id = 0
        moe.quant_config = None
        moe._weight_loader_physical = mock.Mock()
        metadata = mock.Mock(num_logical_experts=256)
        param = mock.Mock(_sglang_require_global_experts=require_global_experts)
        weight = torch.ones(1)

        with (
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer."
                "get_global_expert_location_metadata",
                return_value=metadata,
            ),
            mock.patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer."
                "uses_per_rank_fused_shared_slots",
                return_value=per_rank_slots,
            ),
        ):
            moe.weight_loader(
                param=param,
                loaded_weight=weight,
                weight_name="model.layers.0.mlp.experts.256.gate_proj.weight",
                shard_id="w1",
                expert_id=expert_id,
            )
        return [
            call.kwargs["expert_id"]
            for call in moe._weight_loader_physical.call_args_list
        ]

    def test_shared_weight_ids_match_parameter_storage_and_backend_layout(self):
        cases = (
            (False, False, [288]),
            (False, True, [288]),
            (True, False, [288]),
            (True, True, [rank * 19 + 18 for rank in range(16)]),
        )
        for require_global_experts, per_rank_slots, expected in cases:
            with self.subTest(
                require_global_experts=require_global_experts,
                per_rank_slots=per_rank_slots,
            ):
                self.assertEqual(
                    self._load_shared(
                        require_global_experts=require_global_experts,
                        per_rank_slots=per_rank_slots,
                    ),
                    expected,
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
