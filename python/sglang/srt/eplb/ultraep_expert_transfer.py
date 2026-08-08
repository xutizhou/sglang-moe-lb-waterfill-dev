"""UltraEP communication backend for transient MLB expert replicas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from moe_load_balancer import L3Placement
    from ultra_ep import CollectedExpertLoads, Manager

    from sglang.srt.layers.quantization.base_config import (
        ExpertReplicationTensorView,
    )


_REQUIRED_ULTRAEP_MANAGER_API = (
    "close",
    "collect_topk_loads",
    "get_comm_stream",
    "logical_loads_per_rank",
    "register_master_tensors",
    "replica_staging_buffers",
    "transfer",
)
_REQUIRED_EXTERNAL_TRANSFER_API_VERSION = 2


def _require_external_transfer_api(ultra_ep_module) -> type:
    """Reject the upstream package before it allocates an NVSHMEM manager."""

    version = getattr(
        ultra_ep_module,
        "EXTERNAL_PLACEMENT_TRANSFER_API_VERSION",
        None,
    )
    if (
        not isinstance(version, int)
        or version < _REQUIRED_EXTERNAL_TRANSFER_API_VERSION
    ):
        raise ImportError(
            "--enable-ultraep requires external-placement transfer API "
            f"version >= {_REQUIRED_EXTERNAL_TRANSFER_API_VERSION}; got {version!r}."
        )
    manager_type = getattr(ultra_ep_module, "Manager", None)
    if manager_type is None:
        raise ImportError("--enable-ultraep requires ultra_ep.Manager.")
    missing = [
        name
        for name in _REQUIRED_ULTRAEP_MANAGER_API
        if not hasattr(manager_type, name)
    ]
    if missing:
        raise ImportError(
            "--enable-ultraep requires an UltraEP build with the "
            "external-placement transfer API; the installed Manager is missing: "
            + ", ".join(missing)
        )
    return manager_type


class ExpertTransferEvent:
    """Wait for one already scheduled CUDA-stream boundary."""

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
    """SGLang adapter for UltraEP's external-placement transfer API.

    MLB owns placement and rerouting algorithms. This adapter owns the standalone
    UltraEP communication manager, registers SGLang's expert tensors, and
    materializes UltraEP's NVSHMEM staging rows in SGLang-owned quantized expert
    storage. Placement stays on device and no Python peer-operation plan is
    constructed.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        layer_storages: Mapping[int, ExpertReplicationTensorView],
        num_logical_experts: int,
        num_redundant_experts_per_rank: int,
        overlap_transfer_with_dispatch: bool = False,
    ) -> None:
        try:
            import ultra_ep
        except ImportError as exc:
            raise ImportError(
                "--enable-ultraep requires the standalone UltraEP package "
                "built with its NVSHMEM extension."
            ) from exc
        Manager = _require_external_transfer_api(ultra_ep)

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
        self._overlap_transfer_with_dispatch = overlap_transfer_with_dispatch
        self._num_local_physical_experts = (
            self._num_local_master_experts + self._num_local_redundant_experts
        )
        self._layers = {
            layer_id: self._validate_storage(layer_id, storage)
            for layer_id, storage in layer_storages.items()
        }
        reference = next(iter(self._layers.values()))
        self._validate_consistent_layout(reference)

        self._manager: Manager = Manager(
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
        self._placement_ready_events = {
            layer_id: torch.cuda.Event() for layer_id in self._layers
        }
        self._last_copy_event: torch.cuda.Event | None = None
        self._closed = False

    @property
    def nvl_domain_size(self) -> int:
        """Number of EP ranks in one NVLink placement domain."""

        return self._manager.nvl_domain_size

    @property
    def logical_loads_per_rank(self) -> torch.Tensor:
        """UltraEP-owned NVSHMEM result buffer for load collection."""

        return self._manager.logical_loads_per_rank

    @property
    def communication_stream(self) -> torch.cuda.Stream:
        """Stream shared by load collection, MLB placement, and transfer."""

        return self._manager.get_comm_stream()

    def collect_topk_loads_async(
        self, logical_topk_ids: torch.Tensor
    ) -> CollectedExpertLoads:
        """Adapt native router IDs and enqueue UltraEP's fused load collection."""

        collection_topk_ids = logical_topk_ids.to(dtype=torch.int64).contiguous()
        with torch.cuda.nvtx.range("UltraEP NVSHMEM load fcollect"):
            return self._manager.collect_topk_loads(collection_topk_ids)

    def record_placement_ready(self, layer_id: int) -> ExpertTransferEvent:
        """Record the boundary between MLB placement and expert transfer."""

        try:
            event = self._placement_ready_events[layer_id]
        except KeyError as exc:
            raise KeyError(
                f"No expert storage registered for layer {layer_id}."
            ) from exc
        event.record(self.communication_stream)
        return ExpertTransferEvent(event)

    def apply_async(self, placement: L3Placement) -> ExpertTransferEvent:
        """Enqueue transfer and materialization; return its consumer event."""

        try:
            copy_pairs = self._copy_pairs[placement.layer_id]
        except KeyError as exc:
            raise KeyError(
                f"No expert storage registered for layer {placement.layer_id}."
            ) from exc

        current_stream = torch.cuda.current_stream(device=self._copy_stream.device)
        if self._last_copy_event is not None:
            current_stream.wait_event(self._last_copy_event)

        completion_scope = (
            "outgoing" if self._overlap_transfer_with_dispatch else "global"
        )
        transferred = self._manager.transfer(
            layer_id=placement.layer_id,
            physical_to_logical_map=placement.physical_to_logical_map,
            logical_to_physical_map=placement.logical_to_physical_map,
            logical_replica_counts=placement.logical_replica_counts,
            completion_scope=completion_scope,
        )
        return ExpertTransferEvent(
            self._materialize(
                transferred,
                copy_pairs,
                wait_for_incoming=self._overlap_transfer_with_dispatch,
            )
        )

    def close(self) -> None:
        """Release the standalone UltraEP communication runtime once."""

        if self._closed:
            return
        if self._last_copy_event is not None:
            self._last_copy_event.synchronize()
        self._manager.close()
        self._closed = True

    def _materialize(
        self,
        transfer_event,
        copy_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        wait_for_incoming: bool,
    ) -> torch.cuda.Event:
        """Materialize staging rows asynchronously on the copy stream."""

        with torch.cuda.stream(self._copy_stream):
            if wait_for_incoming:
                transfer_event.current_stream_wait_for_incoming()
            else:
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
            self._manager.register_master_tensors(
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
        staging = self._manager.replica_staging_buffers
        replica_fc1 = staging.fc1_weight.view(
            self._num_local_redundant_experts, *storage.fc1.shape[1:]
        )
        replica_fc2 = staging.fc2_weight.view(
            self._num_local_redundant_experts, *storage.fc2.shape[1:]
        )
        replica_fc1_scale = replica_fc2_scale = None
        if storage.fc1_scale is not None:
            replica_fc1_scale = staging.fc1_weight_scale.view(
                self._num_local_redundant_experts,
                *storage.fc1_scale.shape[1:],
            )
            replica_fc2_scale = staging.fc2_weight_scale.view(
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
