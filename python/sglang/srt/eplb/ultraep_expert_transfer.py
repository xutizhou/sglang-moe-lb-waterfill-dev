"""UltraEP communication backend for transient MLB expert replicas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from moe_load_balancer import L3Placement
    from moe_load_balancer.kernels.ultraep import Manager

    from sglang.srt.layers.quantization.base_config import (
        ExpertReplicationTensorView,
    )


class ExpertTransferEvent:
    """Wait for an already scheduled local expert materialization."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event

    def current_stream_wait(self) -> None:
        self._event.wait(torch.cuda.current_stream())


@dataclass(frozen=True)
class _LayerStorage:
    fc1: torch.Tensor
    fc2: torch.Tensor
    fc1_scale: torch.Tensor | None
    fc2_scale: torch.Tensor | None


class UltraEPExpertTransfer:
    """Synchronize experts with UltraEP from MLB-owned GPU placement maps.

    One UltraEP manager is shared with MLB's L3 placement policy and L2 router.
    UltraEP transfers into its NVSHMEM symmetric replica buffers; a local CUDA
    stream then materializes those rows in SGLang-owned quantized expert storage.
    No placement tensor is copied to the host and no Python peer-operation plan
    is constructed.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        layer_storages: Mapping[int, ExpertReplicationTensorView],
        num_logical_experts: int,
        num_redundant_experts_per_rank: int,
    ) -> None:
        from moe_load_balancer.kernels.ultraep import Manager

        self._group = group
        self._ep_size = group.size()
        if self._ep_size <= 1:
            raise ValueError(
                "UltraEP expert transfer requires EP size greater than one."
            )
        if num_logical_experts <= 0 or num_logical_experts % self._ep_size != 0:
            raise ValueError(
                "Logical experts must be positive and divisible by EP size."
            )
        if num_redundant_experts_per_rank <= 0:
            raise ValueError("At least one redundant expert per rank is required.")
        if not layer_storages:
            raise ValueError("UltraEP expert transfer requires at least one MoE layer.")

        self._num_local_master_experts = num_logical_experts // self._ep_size
        self._num_local_redundant_experts = num_redundant_experts_per_rank
        self._num_local_physical_experts = (
            self._num_local_master_experts + self._num_local_redundant_experts
        )
        self._layers = {
            layer_id: self._validate_storage(layer_id, storage)
            for layer_id, storage in layer_storages.items()
        }
        reference = next(iter(self._layers.values()))
        self._validate_consistent_layout(reference)

        self.manager: Manager = Manager(
            group=group,
            num_layers=max(self._layers) + 1,
            num_local_master_experts=self._num_local_master_experts,
            num_local_redundant_experts=self._num_local_redundant_experts,
            expert_fc1_numel=reference.fc1[0].numel(),
            expert_fc2_numel=reference.fc2[0].numel(),
            is_train=False,
            explicitly_destroy=True,
            weight_data_dtype=reference.fc1.dtype,
            weight_scale_dtype=(
                reference.fc1_scale.dtype
                if reference.fc1_scale is not None
                else torch.float32
            ),
            expert_fc1_weight_scale_numel=(
                reference.fc1_scale[0].numel() if reference.fc1_scale is not None else 0
            ),
            expert_fc2_weight_scale_numel=(
                reference.fc2_scale[0].numel() if reference.fc2_scale is not None else 0
            ),
        )
        self._register_master_rows()
        self._copy_stream = torch.cuda.Stream(device=reference.fc1.device)
        self._copy_pairs = {
            layer_id: self._build_copy_pairs(storage)
            for layer_id, storage in self._layers.items()
        }
        self._last_copy_event: torch.cuda.Event | None = None

    def apply_async(self, placement: L3Placement) -> Optional[ExpertTransferEvent]:
        """Enqueue communication and local copies without blocking the caller."""

        if placement.metadata.get("placement_refreshed") is False:
            return None

        try:
            copy_pairs = self._copy_pairs[placement.layer_id]
        except KeyError as exc:
            raise KeyError(
                f"No expert storage registered for layer {placement.layer_id}."
            ) from exc

        current_stream = torch.cuda.current_stream(device=self._copy_stream.device)
        if self._last_copy_event is not None:
            current_stream.wait_event(self._last_copy_event)

        transfer_event = self.manager.weight_sync_from_placement(
            placement.layer_id,
            placement.physical_to_logical_map,
            placement.logical_to_physical_map,
            placement.logical_replica_counts,
            async_finish=True,
        )
        return ExpertTransferEvent(self._materialize(transfer_event, copy_pairs))

    def _materialize(
        self,
        transfer_event,
        copy_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.cuda.Event:
        """Copy shared staging rows after the transfer on a dedicated stream."""

        with torch.cuda.stream(self._copy_stream):
            transfer_event.current_stream_wait()
            for destination, source in copy_pairs:
                destination.copy_(source)
            event = self._copy_stream.record_event()
        self._last_copy_event = event
        return event

    def _validate_storage(
        self, layer_id: int, storage: ExpertReplicationTensorView
    ) -> _LayerStorage:
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("Layer IDs must be non-negative integers.")
        if storage.auxiliary_tensors:
            names = ", ".join(name for name, _ in storage.auxiliary_tensors)
            raise ValueError(
                "UltraEP communication does not yet support auxiliary expert "
                f"tensors: {names}."
            )
        if (storage.w13_weight_scale is None) != (storage.w2_weight_scale is None):
            raise ValueError("UltraEP requires both expert scale tensors or neither.")

        tensors = tuple(
            tensor
            for tensor in (
                storage.w13_weight,
                storage.w2_weight,
                storage.w13_weight_scale,
                storage.w2_weight_scale,
            )
            if tensor is not None
        )
        device = tensors[0].device
        if device.type != "cuda":
            raise ValueError("UltraEP expert tensors must be CUDA tensors.")
        for tensor in tensors:
            if tensor.shape[0] != self._num_local_physical_experts:
                raise ValueError(
                    f"Layer {layer_id} expert tensor has {tensor.shape[0]} rows; "
                    f"expected {self._num_local_physical_experts}."
                )
            if tensor.device != device:
                raise ValueError(
                    f"Layer {layer_id} expert tensors must share a device."
                )
            if not tensor[0].is_contiguous():
                raise ValueError(f"Layer {layer_id} expert rows must be contiguous.")
        if storage.w13_weight.dtype != storage.w2_weight.dtype:
            raise ValueError("UltraEP requires w13 and w2 to use the same dtype.")
        if (
            storage.w13_weight_scale is not None
            and storage.w13_weight_scale.dtype != storage.w2_weight_scale.dtype
        ):
            raise ValueError(
                "UltraEP requires w13 and w2 scales to use the same dtype."
            )
        return _LayerStorage(
            fc1=storage.w13_weight,
            fc2=storage.w2_weight,
            fc1_scale=storage.w13_weight_scale,
            fc2_scale=storage.w2_weight_scale,
        )

    def _validate_consistent_layout(self, reference: _LayerStorage) -> None:
        expected = self._layout_signature(reference)
        for layer_id, storage in self._layers.items():
            if self._layout_signature(storage) != expected:
                raise ValueError(
                    f"Layer {layer_id} expert layout differs from the other MoE layers."
                )

    @staticmethod
    def _layout_signature(storage: _LayerStorage) -> tuple[object, ...]:
        return (
            storage.fc1[0].numel(),
            storage.fc2[0].numel(),
            storage.fc1.dtype,
            storage.fc2.dtype,
            storage.fc1.device,
            storage.fc1_scale[0].numel() if storage.fc1_scale is not None else 0,
            storage.fc2_scale[0].numel() if storage.fc2_scale is not None else 0,
            storage.fc1_scale.dtype if storage.fc1_scale is not None else None,
            storage.fc2_scale.dtype if storage.fc2_scale is not None else None,
        )

    def _register_master_rows(self) -> None:
        master_count = self._num_local_master_experts
        for layer_id, storage in self._layers.items():
            self.manager.construct_local_master_ptr_pool(
                layer_id,
                [storage.fc1[index] for index in range(master_count)],
                [storage.fc2[index] for index in range(master_count)],
                fc1_weight_scales=(
                    [storage.fc1_scale[index] for index in range(master_count)]
                    if storage.fc1_scale is not None
                    else None
                ),
                fc2_weight_scales=(
                    [storage.fc2_scale[index] for index in range(master_count)]
                    if storage.fc2_scale is not None
                    else None
                ),
            )

    def _build_copy_pairs(
        self, storage: _LayerStorage
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        master_count = self._num_local_master_experts
        replica_fc1 = self.manager.local_replica_fc1_weight_buffer.view(
            self._num_local_redundant_experts, *storage.fc1.shape[1:]
        )
        replica_fc2 = self.manager.local_replica_fc2_weight_buffer.view(
            self._num_local_redundant_experts, *storage.fc2.shape[1:]
        )
        replica_fc1_scale = replica_fc2_scale = None
        if storage.fc1_scale is not None:
            replica_fc1_scale = self.manager.local_replica_fc1_weight_scale_buffer.view(
                self._num_local_redundant_experts,
                *storage.fc1_scale.shape[1:],
            )
            replica_fc2_scale = self.manager.local_replica_fc2_weight_scale_buffer.view(
                self._num_local_redundant_experts,
                *storage.fc2_scale.shape[1:],
            )

        pairs = []
        for replica_index in range(self._num_local_redundant_experts):
            destination_index = master_count + replica_index
            # UltraEP stores FC1 and FC2 in one symmetric allocation, so the
            # multi-row FC views have a gap between rows. Copy one contiguous
            # expert row at a time to let PyTorch use cudaMemcpyAsync instead
            # of a TensorIterator SM copy kernel. Quantized weights use byte
            # views to preserve their payload exactly.
            pairs.extend(
                (
                    (
                        storage.fc1[destination_index].view(torch.uint8),
                        replica_fc1[replica_index].view(torch.uint8),
                    ),
                    (
                        storage.fc2[destination_index].view(torch.uint8),
                        replica_fc2[replica_index].view(torch.uint8),
                    ),
                )
            )
            if storage.fc1_scale is not None:
                pairs.extend(
                    (
                        (
                            storage.fc1_scale[destination_index],
                            replica_fc1_scale[replica_index],
                        ),
                        (
                            storage.fc2_scale[destination_index],
                            replica_fc2_scale[replica_index],
                        ),
                    )
                )
        return tuple(pairs)
