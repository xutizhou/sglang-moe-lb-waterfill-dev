import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb.ultraep_expert_transfer import (
    UltraEPExpertTransfer,
    _LayerStorage,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_layout_signature_tracks_weight_and_scale_contract():
    storage = _LayerStorage(
        fc1=torch.empty((3, 4), dtype=torch.uint8),
        fc2=torch.empty((3, 2), dtype=torch.uint8),
        fc1_scale=torch.empty((3, 2), dtype=torch.float32),
        fc2_scale=torch.empty((3, 1), dtype=torch.float32),
    )

    signature = UltraEPExpertTransfer._layout_signature(storage)

    assert signature[:4] == (4, 2, torch.uint8, torch.uint8)
    assert signature[5:] == (2, 1, torch.float32, torch.float32)


def test_storage_rejects_auxiliary_expert_tensors_before_runtime_init():
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._num_local_physical_experts = 3
    storage = type(
        "Storage",
        (),
        {
            "w13_weight": torch.empty((3, 4)),
            "w2_weight": torch.empty((3, 2)),
            "w13_weight_scale": None,
            "w2_weight_scale": None,
            "auxiliary_tensors": (("bias", torch.empty((3, 1))),),
        },
    )()

    with pytest.raises(ValueError, match="auxiliary expert tensors"):
        transfer._validate_storage(0, storage)


def test_copy_pairs_treat_quantized_weights_as_opaque_bytes():
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._num_local_master_experts = 2
    transfer._num_local_redundant_experts = 1
    transfer.manager = SimpleNamespace(
        local_replica_fc1_weight_buffer=torch.empty((1, 4), dtype=torch.float16),
        local_replica_fc2_weight_buffer=torch.empty((1, 2), dtype=torch.float16),
        local_replica_fc1_weight_scale_buffer=torch.empty((1, 2)),
        local_replica_fc2_weight_scale_buffer=torch.empty((1, 1)),
    )
    storage = _LayerStorage(
        fc1=torch.empty((3, 2, 2), dtype=torch.float16),
        fc2=torch.empty((3, 2), dtype=torch.float16),
        fc1_scale=torch.empty((3, 2)),
        fc2_scale=torch.empty((3, 1)),
    )

    pairs = transfer._build_copy_pairs(storage)

    assert pairs[0][0].dtype is torch.uint8
    assert pairs[0][1].dtype is torch.uint8
    assert pairs[1][0].dtype is torch.uint8
    assert pairs[1][1].dtype is torch.uint8
    assert pairs[2][0].dtype is torch.float32
    assert pairs[3][0].dtype is torch.float32


def test_copy_pairs_split_strided_ultraep_storage_into_contiguous_rows():
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._num_local_master_experts = 2
    transfer._num_local_redundant_experts = 2
    weight_buffer = torch.empty((2, 6), dtype=torch.float16)
    scale_buffer = torch.empty((2, 3), dtype=torch.float32)
    transfer.manager = SimpleNamespace(
        local_replica_fc1_weight_buffer=weight_buffer[:, :4],
        local_replica_fc2_weight_buffer=weight_buffer[:, 4:],
        local_replica_fc1_weight_scale_buffer=scale_buffer[:, :2],
        local_replica_fc2_weight_scale_buffer=scale_buffer[:, 2:],
    )
    storage = _LayerStorage(
        fc1=torch.empty((4, 2, 2), dtype=torch.float16),
        fc2=torch.empty((4, 2), dtype=torch.float16),
        fc1_scale=torch.empty((4, 2), dtype=torch.float32),
        fc2_scale=torch.empty((4, 1), dtype=torch.float32),
    )

    pairs = transfer._build_copy_pairs(storage)

    assert len(pairs) == 8
    assert all(destination.is_contiguous() for destination, _ in pairs)
    assert all(source.is_contiguous() for _, source in pairs)


def test_decode_reuse_skips_weight_transfer():
    transfer = object.__new__(UltraEPExpertTransfer)
    placement = SimpleNamespace(metadata={"placement_refreshed": False})

    assert transfer.apply_async(placement) is None


def test_local_materialization_is_scheduled_by_apply_async(monkeypatch):
    calls = []

    class _FakeComputeStream:
        device = "cuda:0"

        def wait_event(self, event):
            calls.append(("wait_previous", event))

    class _FakeCopyStream:
        device = "cuda:0"

        def record_event(self):
            calls.append(("record_materialized",))
            return materialized_event

    class _FakeTransferEvent:
        def current_stream_wait(self):
            calls.append(("wait_transfer",))

    class _FakeMaterializedEvent:
        def wait(self, stream):
            calls.append(("wait_materialized", stream))

    class _FakeDestination:
        def copy_(self, source):
            calls.append(("copy", source))

    compute_stream = _FakeComputeStream()
    materialized_event = _FakeMaterializedEvent()
    transfer_event = _FakeTransferEvent()
    source = object()
    manager = SimpleNamespace()

    def weight_sync_from_placement(*args, **kwargs):
        calls.append(("weight_sync", args, kwargs))
        return transfer_event

    manager.weight_sync_from_placement = weight_sync_from_placement
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer.manager = manager
    transfer._copy_stream = _FakeCopyStream()
    transfer._copy_pairs = {3: ((_FakeDestination(), source),)}
    transfer._last_copy_event = None
    placement = SimpleNamespace(
        layer_id=3,
        physical_to_logical_map="p2l",
        logical_to_physical_map="l2p",
        logical_replica_counts="counts",
        metadata={"placement_refreshed": True},
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args, **kwargs: compute_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    ready = transfer.apply_async(placement)

    assert [call[0] for call in calls] == [
        "weight_sync",
        "wait_transfer",
        "copy",
        "record_materialized",
    ]
    ready.current_stream_wait()
    ready.current_stream_wait()
    assert [call[0] for call in calls] == [
        "weight_sync",
        "wait_transfer",
        "copy",
        "record_materialized",
        "wait_materialized",
        "wait_materialized",
    ]
    assert transfer._last_copy_event is materialized_event


def test_next_transfer_waits_for_previous_materialization(monkeypatch):
    calls = []

    class _FakeComputeStream:
        device = "cuda:0"

        def wait_event(self, event):
            calls.append(("wait_previous", event))

    class _FakeCopyStream:
        device = "cuda:0"

        def record_event(self):
            return materialized_event

    class _FakeTransferEvent:
        def current_stream_wait(self):
            calls.append(("wait_transfer",))

    class _FakeMaterializedEvent:
        def wait(self, stream):
            calls.append(("wait_materialized", stream))

    class _FakeDestination:
        def copy_(self, source):
            calls.append(("copy",))

    compute_stream = _FakeComputeStream()
    materialized_event = _FakeMaterializedEvent()
    manager = SimpleNamespace()

    def weight_sync_from_placement(*args, **kwargs):
        calls.append(("weight_sync",))
        return _FakeTransferEvent()

    manager.weight_sync_from_placement = weight_sync_from_placement
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer.manager = manager
    transfer._copy_stream = _FakeCopyStream()
    transfer._copy_pairs = {0: ((_FakeDestination(), object()),)}
    transfer._last_copy_event = None
    placement = SimpleNamespace(
        layer_id=0,
        physical_to_logical_map="p2l",
        logical_to_physical_map="l2p",
        logical_replica_counts="counts",
        metadata={"placement_refreshed": True},
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args, **kwargs: compute_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    first = transfer.apply_async(placement)
    first.current_stream_wait()
    transfer.apply_async(placement)

    assert [call[0] for call in calls] == [
        "weight_sync",
        "wait_transfer",
        "copy",
        "wait_materialized",
        "wait_previous",
        "weight_sync",
        "wait_transfer",
        "copy",
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
