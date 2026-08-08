"""Thin SGLang adapter for UltraEP expert-weight communication."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.distributed import get_moe_ep_group


@dataclass(frozen=True)
class _LayerTensors:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None


class UltraEPExpertTransfer:
    """Use UltraEP only to move weights for an already computed placement."""

    def __init__(
        self,
        model,
        num_logical_experts: int,
        num_redundant_per_rank: int,
    ) -> None:
        try:
            from ultra_ep import Manager
        except ImportError as exc:
            raise ImportError("UltraEP EPLB requires the ultra_ep package.") from exc
        if not hasattr(Manager, "weight_sync_from_placement"):
            raise ImportError(
                "The installed UltraEP does not expose weight_sync_from_placement()."
            )

        ep_group = get_moe_ep_group()
        ep_size = ep_group.world_size
        master_count = num_logical_experts // ep_size
        physical_count = master_count + num_redundant_per_rank
        self._master_count = master_count
        self._layers = self._find_layers(model, physical_count)
        reference = next(iter(self._layers.values()))
        self._check_layouts(reference)

        self._manager = Manager(
            group=ep_group.device_group,
            num_layers=max(self._layers) + 1,
            num_local_master_experts=master_count,
            num_local_redundant_experts=num_redundant_per_rank,
            expert_fc1_numel=reference.w13[0].numel(),
            expert_fc2_numel=reference.w2[0].numel(),
            is_train=False,
            explicitly_destroy=True,
            weight_data_dtype=reference.w13.dtype,
            weight_scale_dtype=(
                reference.w13_scale.dtype
                if reference.w13_scale is not None
                else torch.float32
            ),
            expert_fc1_weight_scale_numel=(
                reference.w13_scale[0].numel() if reference.w13_scale is not None else 0
            ),
            expert_fc2_weight_scale_numel=(
                reference.w2_scale[0].numel() if reference.w2_scale is not None else 0
            ),
        )
        for layer_id, tensors in self._layers.items():
            self._manager.construct_local_master_ptr_pool(
                layer_id,
                list(tensors.w13[:master_count]),
                list(tensors.w2[:master_count]),
                fc1_weight_scales=(
                    list(tensors.w13_scale[:master_count])
                    if tensors.w13_scale is not None
                    else None
                ),
                fc2_weight_scales=(
                    list(tensors.w2_scale[:master_count])
                    if tensors.w2_scale is not None
                    else None
                ),
            )

    @property
    def nvl_domain_size(self) -> int:
        return self._manager.nvl_domain_size

    def update(self, placement, layer_ids: list[int]) -> dict[int, list[int]]:
        """Transfer and materialize the layers selected by EPLBManager."""

        l2p_width = self._manager.logical_to_physical_map.shape[-1]
        for layer_id in layer_ids:
            self._manager.weight_sync_from_placement(
                layer_id,
                placement.physical_to_logical_map[layer_id].to(dtype=torch.int32),
                placement.logical_to_all_physical_map[layer_id, :, :l2p_width].to(
                    dtype=torch.int32
                ),
                placement.logical_to_all_physical_map_num_valid[layer_id].to(
                    dtype=torch.int32
                ),
            )
            self._materialize(layer_id)
        return {}

    def close(self) -> None:
        self._manager.destroy()

    @staticmethod
    def _find_layers(model, physical_count: int) -> dict[int, _LayerTensors]:
        layers = {}
        for module in model.modules():
            experts = getattr(module, "experts", None)
            layer_id = getattr(module, "layer_id", None)
            quant_method = getattr(experts, "quant_method", None)
            if layer_id is None or quant_method is None:
                continue
            try:
                info = quant_method.get_triton_quant_info(experts)
            except NotImplementedError:
                continue
            w13 = getattr(info, "w13_weight", None)
            w2 = getattr(info, "w2_weight", None)
            if not isinstance(w13, torch.Tensor) or not isinstance(w2, torch.Tensor):
                continue
            if w13.shape[0] != physical_count or w2.shape[0] != physical_count:
                continue

            unsupported = []
            for name in ("b13", "b2", "w13_zp", "w2_zp", "a13_scale", "a2_scale"):
                value = getattr(info, name, None)
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == physical_count
                ):
                    unsupported.append(name)
            if unsupported:
                raise ValueError(
                    "UltraEP does not support these per-expert tensors: "
                    + ", ".join(unsupported)
                )

            w13_scale = getattr(info, "w13_scale", None)
            w2_scale = getattr(info, "w2_scale", None)
            if (w13_scale is None) != (w2_scale is None):
                raise ValueError("UltraEP requires both weight scales or neither.")
            layers[layer_id] = _LayerTensors(w13, w2, w13_scale, w2_scale)

        if not layers:
            raise ValueError("UltraEP found no supported MoE expert tensors.")
        return layers

    def _check_layouts(self, reference: _LayerTensors) -> None:
        def signature(tensors: _LayerTensors):
            return (
                tensors.w13[0].numel(),
                tensors.w2[0].numel(),
                tensors.w13.dtype,
                tensors.w2.dtype,
                tensors.w13_scale[0].numel() if tensors.w13_scale is not None else 0,
                tensors.w2_scale[0].numel() if tensors.w2_scale is not None else 0,
            )

        expected = signature(reference)
        if any(signature(tensors) != expected for tensors in self._layers.values()):
            raise ValueError("UltraEP requires one expert tensor layout across layers.")

    def _materialize(self, layer_id: int) -> None:
        tensors = self._layers[layer_id]
        sources = (
            self._manager.local_replica_fc1_weight_buffer,
            self._manager.local_replica_fc2_weight_buffer,
            self._manager.local_replica_fc1_weight_scale_buffer,
            self._manager.local_replica_fc2_weight_scale_buffer,
        )
        destinations = (tensors.w13, tensors.w2, tensors.w13_scale, tensors.w2_scale)
        for source, destination in zip(sources, destinations, strict=True):
            if destination is None:
                continue
            for replica_id in range(source.shape[0]):
                destination[self._master_count + replica_id].view(torch.uint8).reshape(
                    -1
                ).copy_(source[replica_id].view(torch.uint8).reshape(-1))
