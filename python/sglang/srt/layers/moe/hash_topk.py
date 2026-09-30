from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    TopKConfig,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
    remap_topk_for_per_rank_shared_slots,
)
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_hip, is_npu, is_xpu

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()


class HashTopK(nn.Module):
    def __init__(
        self,
        topk,
        num_experts,
        num_fused_shared_experts,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
        layer_id: Optional[int] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        from sglang.srt.eplb.moe_load_balancer_glue import (
            get_moe_load_balancer_pipeline,
        )

        pipeline = get_moe_load_balancer_pipeline(
            get_exec().moe.moe_load_balancer_algorithm
        )
        capabilities = pipeline.capabilities if pipeline is not None else None
        self.mlb_requires_post_topk_routing = (
            capabilities is not None and capabilities.requires_post_topk_routing
        )
        self.mlb_routes_shared_expert = (
            num_fused_shared_experts > 0
            and capabilities is not None
            and capabilities.routes_shared_expert
        )
        self.moe_load_balancer = None
        if self.mlb_routes_shared_expert:
            topk -= num_fused_shared_experts
            num_fused_shared_experts = 0

        self.num_experts = num_experts
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),
            requires_grad=False,
        )
        self._init_default_tid2eid()

        self.apply_routed_scaling_factor_on_output = (
            apply_routed_scaling_factor_on_output
        )
        if apply_routed_scaling_factor_on_output and num_fused_shared_experts > 0:
            raise NotImplementedError(
                "HashTopK + apply_routed_scaling_factor_on_output is not supported "
                "with fused shared experts; pass --disable-shared-experts-fusion."
            )

    def _init_default_tid2eid(self) -> None:
        topk = self.tid2eid.shape[1]
        if topk == 0:
            return

        # DummyModelLoader only initializes floating tensors, so keep this int
        # lookup table valid until real checkpoints overwrite it.
        token_ids = torch.arange(
            self.tid2eid.shape[0], dtype=self.tid2eid.dtype, device=self.tid2eid.device
        ).unsqueeze(1)
        expert_offsets = torch.arange(
            topk, dtype=self.tid2eid.dtype, device=self.tid2eid.device
        ).unsqueeze(0)
        tid2eid = (token_ids + expert_offsets) % self.num_experts
        with torch.no_grad():
            self.tid2eid.copy_(tid2eid.to(self.tid2eid.dtype))

    def empty_topk_output(self, device: torch.device):
        topk = self.topk - self.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
        if has_per_rank_fused_shared_slots(self.num_fused_shared_experts):
            n = self.num_fused_shared_experts
            topk_output = topk_output._replace(
                topk_ids=topk_output.topk_ids.new_empty(
                    (0, topk_output.topk_ids.shape[-1] + n)
                ),
                topk_weights=topk_output.topk_weights.new_empty(
                    (0, topk_output.topk_weights.shape[-1] + n)
                ),
            )
        return self._apply_moe_load_balancer(topk_output, num_tokens=0)

    def _apply_moe_load_balancer(
        self,
        topk_output: StandardTopKOutput,
        num_tokens: int,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        forward_batch=None,
    ) -> StandardTopKOutput:
        if self.moe_load_balancer is None:
            if self.mlb_requires_post_topk_routing:
                raise RuntimeError(
                    "MLB L2 is enabled but ModelRunner did not attach MoELoadBalancer."
                )
            return topk_output

        from sglang.srt.eplb.moe_load_balancer_glue import route_topk_with_mlb

        return route_topk_with_mlb(
            moe_load_balancer=self.moe_load_balancer,
            layer_id=self.layer_id,
            topk_output=topk_output,
            num_tokens=num_tokens,
            num_token_non_padded=num_token_non_padded,
            forward_batch=forward_batch,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _forward_torch(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = torch.nn.functional.softplus(router_logits).sqrt()

        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_fused_shared_experts == 1:
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= (
                    topk_weights[:, :-1].sum(dim=-1, keepdim=True)
                    + _RENORMALIZE_SUM_EPSILON
                )

            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_fused_shared_experts,
                size=(num_token,),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )

            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor
            )
        else:
            topk_ids[:, :] = self.tid2eid[input_ids]
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
            if self.score_func != "softmax":
                topk_weights[:, :] /= (
                    topk_weights[:, :].sum(dim=-1, keepdim=True)
                    + _RENORMALIZE_SUM_EPSILON
                )

        return topk_weights, topk_ids

    def _forward_xpu(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The XPU 'hash_topk' kernel currently supports the 'sqrtsoftplus' score func only.
        # Other score funcs fall back to the torch implementation; more will be supported in the future.
        if self.score_func == "sqrtsoftplus":
            from sgl_kernel import hash_topk

            num_tokens = router_logits.size(0)
            topk_routed = self.tid2eid.size(1)
            topk_fused = topk_routed + self.num_fused_shared_experts
            topk_ids = torch.empty(
                (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
            )
            topk_weights = torch.empty(
                (num_tokens, topk_fused),
                dtype=torch.float32,
                device=router_logits.device,
            )
            hash_topk(
                router_logits,
                input_ids,
                self.tid2eid,
                topk_weights,
                topk_ids,
                self.routed_scaling_factor,
                self.score_func,
            )
            return topk_weights, topk_ids
        else:
            return self._forward_torch(router_logits, input_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        forward_batch=None,
    ):
        assert input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0], (
            f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"
        )

        if _is_xpu:
            topk_weights, topk_ids = self._forward_xpu(router_logits, input_ids)
        elif envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.kernels.ops.attention.dsv4 import hash_topk

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_fused_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)
        if _is_hip or _is_npu:
            topk_weights = topk_weights.to(torch.float32)

        if self.apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * self.routed_scaling_factor

        num_fused_shared_experts = self.num_fused_shared_experts
        recorder_topk_ids = None
        if has_per_rank_fused_shared_slots(num_fused_shared_experts):
            # ExpertDistributionRecorder tracks the routed experts as selected,
            # before the per-rank shared-slot layout shifts their ids.
            recorder_topk_ids = topk_ids[:, :-num_fused_shared_experts].clone()
            topk_ids, topk_weights = remap_topk_for_per_rank_shared_slots(
                topk_ids,
                topk_weights,
                num_fused_shared_experts,
                self.num_experts,
                TopKConfig(
                    top_k=self.topk,
                    num_fused_shared_experts=num_fused_shared_experts,
                    routed_scaling_factor=self.routed_scaling_factor,
                ),
            )
        if is_hip():
            _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)
        else:
            _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
            if recorder_topk_ids is not None:
                _mask_topk_ids_padded_region(recorder_topk_ids, num_token_non_padded)
        if recorder_topk_ids is None:
            recorder_topk_ids = topk_ids
        if self.moe_load_balancer is None:
            get_global_expert_distribution_recorder().on_select_experts(
                topk_ids=recorder_topk_ids
            )
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        topk_output = self._apply_moe_load_balancer(
            topk_output,
            hidden_states.shape[0],
            num_token_non_padded=num_token_non_padded,
            forward_batch=forward_batch,
        )
        if is_hip():
            _zero_topk_weights_padded_region(
                topk_output.topk_weights, num_token_non_padded
            )
        return topk_output
