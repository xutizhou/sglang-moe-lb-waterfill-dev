"""Thin SGLang adapter for UltraEP expert P2P communication."""

from __future__ import annotations

import torch

from sglang.srt.distributed import get_moe_ep_group


class UltraEPExpertTransfer:
    def __init__(
        self,
        *,
        num_layers: int,
        num_logical_experts: int,
        num_redundant_per_rank: int,
    ) -> None:
        self._num_layers = num_layers
        self._num_redundant = num_redundant_per_rank
        self._group = get_moe_ep_group()
        self._num_masters = num_logical_experts // self._group.world_size
        self._manager = None
        self._stream = torch.cuda.Stream()
        try:
            from ultra_ep import init_runtime
        except ImportError as exc:
            raise ImportError("UltraEP transfer requires the ultra_ep package") from exc
        self._nvl_domain_size = init_runtime(self._group.device_group)

    @property
    def nvl_domain_size(self) -> int:
        return self._nvl_domain_size

    @staticmethod
    def _current_weights(experts):
        info = experts.quant_method.get_triton_quant_info(experts)
        unsupported = ("b13", "b2", "w13_zp", "w2_zp")
        if any(getattr(info, name, None) is not None for name in unsupported):
            raise ValueError("UltraEP transfer does not support expert bias/zero-point")
        weights = (
            info.w13_weight,
            info.w2_weight,
            getattr(info, "w13_scale", None),
            getattr(info, "w2_scale", None),
        )
        num_shared = getattr(experts, "num_fused_shared_experts", 0)
        if num_shared > 0:
            # DeepEP keeps fused shared slots after routed experts. Runtime
            # placement migrates routed replicas only; shared slots stay fixed.
            weights = tuple(
                weight if weight is None else weight[:-num_shared] for weight in weights
            )
        return weights

    def _initialize(self, weights) -> None:
        try:
            from ultra_ep import Manager
        except ImportError as exc:
            raise ImportError("UltraEP transfer requires the ultra_ep package") from exc
        w13, w2, w13_scale, w2_scale = weights
        if (w13_scale is None) != (w2_scale is None):
            raise ValueError("UltraEP requires both expert weight scales or neither")
        self._manager = Manager(
            group=self._group.device_group,
            num_layers=self._num_layers,
            num_local_master_experts=self._num_masters,
            num_local_redundant_experts=self._num_redundant,
            expert_fc1_numel=w13[0].numel(),
            expert_fc2_numel=w2[0].numel(),
            is_train=False,
            explicitly_destroy=True,
            weight_data_dtype=w13.dtype,
            weight_scale_dtype=(
                w13_scale.dtype if w13_scale is not None else torch.float32
            ),
            expert_fc1_weight_scale_numel=(
                w13_scale[0].numel() if w13_scale is not None else 0
            ),
            expert_fc2_weight_scale_numel=(
                w2_scale[0].numel() if w2_scale is not None else 0
            ),
        )

    def transfer(self, layer_id: int, experts, placement):
        weights = self._current_weights(experts)
        if self._manager is None:
            self._initialize(weights)
        current_stream = torch.cuda.current_stream()
        self._stream.wait_stream(current_stream)
        with torch.cuda.stream(self._stream):
            return self._manager.transfer_from_placement(
                layer_id,
                weights[0],
                weights[1],
                placement.physical_to_logical,
                placement.logical_to_physical,
                placement.replica_counts,
                weights[2],
                weights[3],
            )

    def all_gather_loads(self, local_loads: torch.Tensor):
        """Use UltraEP communication once its weight manager is initialized."""

        if self._manager is None:
            return None
        gathered, event = self._manager.all_gather_loads(local_loads)
        event.current_stream_wait()
        return gathered

    def close(self) -> None:
        if self._manager is not None:
            self._manager.destroy()
            self._manager = None
