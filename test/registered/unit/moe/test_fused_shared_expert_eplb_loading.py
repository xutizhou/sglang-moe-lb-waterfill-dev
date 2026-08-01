from unittest import TestCase, mock

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


class TestFusedSharedExpertEPLBLoading(TestCase):
    def test_replicates_shared_weight_after_redundant_routed_experts(self):
        moe = object.__new__(FusedMoE)
        moe._has_fused_shared = True
        moe._num_global_routed = 288
        moe.num_fused_shared_experts = 1
        moe.moe_ep_size = 16
        moe.layer_id = 0
        moe.quant_config = None
        moe._weight_loader_physical = mock.Mock()

        metadata = mock.Mock(num_logical_experts=256)
        param = mock.Mock(_sglang_require_global_experts=False)
        weight = torch.ones(1)

        with mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer."
            "get_global_expert_location_metadata",
            return_value=metadata,
        ):
            moe.weight_loader(
                param=param,
                loaded_weight=weight,
                weight_name="model.layers.0.mlp.experts.256.gate_proj.weight",
                shard_id="w1",
                expert_id=256,
            )

        self.assertEqual(moe._weight_loader_physical.call_count, 16)
        self.assertEqual(
            [call.kwargs["expert_id"] for call in moe._weight_loader_physical.call_args_list],
            list(range(288, 304)),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
