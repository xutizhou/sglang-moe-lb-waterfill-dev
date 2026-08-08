import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb.ultraep_expert_transfer import (
    UltraEPExpertTransfer,
    _LayerStorage,
    _require_external_transfer_api,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_external_transfer_api_is_checked_before_manager_construction():
    class _UpstreamManagerWithoutExternalTransfer:
        def close(self):
            pass

    incompatible_ultraep = SimpleNamespace(
        EXTERNAL_PLACEMENT_TRANSFER_API_VERSION=2,
        Manager=_UpstreamManagerWithoutExternalTransfer,
    )
    with pytest.raises(ImportError, match="external-placement transfer API") as exc:
        _require_external_transfer_api(incompatible_ultraep)

    message = str(exc.value)
    assert "collect_topk_loads" in message
    assert "get_comm_stream" in message
    assert "register_master_tensors" in message
    assert "transfer" in message


@pytest.mark.parametrize("version", [None, 1])
def test_external_transfer_api_rejects_old_capability_versions(version):
    module = SimpleNamespace(Manager=object)
    if version is not None:
        module.EXTERNAL_PLACEMENT_TRANSFER_API_VERSION = version

    with pytest.raises(ImportError, match="version >= 2"):
        _require_external_transfer_api(module)


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
    transfer._manager = SimpleNamespace(
        replica_staging_buffers=SimpleNamespace(
            fc1_weight=torch.empty((1, 4), dtype=torch.float16),
            fc2_weight=torch.empty((1, 2), dtype=torch.float16),
            fc1_weight_scale=torch.empty((1, 2)),
            fc2_weight_scale=torch.empty((1, 1)),
        )
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
    transfer._manager = SimpleNamespace(
        replica_staging_buffers=SimpleNamespace(
            fc1_weight=weight_buffer[:, :4],
            fc2_weight=weight_buffer[:, 4:],
            fc1_weight_scale=scale_buffer[:, :2],
            fc2_weight_scale=scale_buffer[:, 2:],
        )
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


def test_load_collection_delegates_to_standalone_ultraep():
    calls = []
    loads = torch.arange(6, dtype=torch.int32).view(2, 3)

    class _Collected:
        loads_per_rank = loads

    comm_stream = object()
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._manager = SimpleNamespace(
        logical_loads_per_rank=loads,
        get_comm_stream=lambda: comm_stream,
        collect_topk_loads=lambda topk: calls.append(("collect", topk)) or _Collected(),
    )
    topk = torch.tensor([[0, 1]], dtype=torch.int32)

    assert transfer.logical_loads_per_rank is loads
    assert transfer.communication_stream is comm_stream
    collected = transfer.collect_topk_loads_async(topk)
    assert collected.loads_per_rank is loads
    assert len(calls) == 1
    assert calls[0][0] == "collect"
    assert calls[0][1].dtype is torch.int64
    assert calls[0][1].tolist() == topk.tolist()


def test_placement_ready_event_is_persistent_per_layer(monkeypatch):
    calls = []

    class _Event:
        def record(self, stream):
            calls.append(("record", stream))

        def wait(self, stream):
            calls.append(("wait", stream))

    current_stream = object()
    communication_stream = object()
    event = _Event()
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._placement_ready_events = {3: event}
    transfer._copy_stream = SimpleNamespace(device="cuda:0")
    transfer._manager = SimpleNamespace(get_comm_stream=lambda: communication_stream)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args, **kwargs: current_stream
    )

    ready = transfer.record_placement_ready(3)
    ready.current_stream_wait()

    assert calls == [("record", communication_stream), ("wait", current_stream)]
    with pytest.raises(KeyError, match="layer 4"):
        transfer.record_placement_ready(4)


def test_close_owns_standalone_ultraep_manager_lifecycle():
    calls = []
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._manager = SimpleNamespace(
        nvl_domain_size=8,
        close=lambda: calls.append("close"),
    )
    transfer._closed = False
    transfer._last_copy_event = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize")
    )

    assert transfer.nvl_domain_size == 8
    transfer.close()
    transfer.close()

    assert calls == ["synchronize", "close"]


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

    def transfer_from_placement(*args, **kwargs):
        calls.append(("transfer", args, kwargs))
        return transfer_event

    manager.transfer = transfer_from_placement
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._manager = manager
    transfer._copy_stream = _FakeCopyStream()
    transfer._copy_pairs = {3: ((_FakeDestination(), source),)}
    transfer._last_copy_event = None
    transfer._overlap_transfer_with_dispatch = False
    placement = SimpleNamespace(
        layer_id=3,
        physical_to_logical_map="p2l",
        logical_to_physical_map="l2p",
        logical_replica_counts="counts",
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args, **kwargs: compute_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    ready = transfer.apply_async(placement)

    assert [call[0] for call in calls] == [
        "transfer",
        "wait_transfer",
        "copy",
        "record_materialized",
    ]
    ready.current_stream_wait()
    assert [call[0] for call in calls] == [
        "transfer",
        "wait_transfer",
        "copy",
        "record_materialized",
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

    def transfer_from_placement(*args, **kwargs):
        calls.append(("transfer",))
        return _FakeTransferEvent()

    manager.transfer = transfer_from_placement
    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._manager = manager
    transfer._copy_stream = _FakeCopyStream()
    transfer._copy_pairs = {0: ((_FakeDestination(), object()),)}
    transfer._last_copy_event = None
    transfer._overlap_transfer_with_dispatch = False
    placement = SimpleNamespace(
        layer_id=0,
        physical_to_logical_map="p2l",
        logical_to_physical_map="l2p",
        logical_replica_counts="counts",
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args, **kwargs: compute_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    first = transfer.apply_async(placement)
    first.current_stream_wait()
    transfer.apply_async(placement)

    assert [call[0] for call in calls] == [
        "transfer",
        "wait_transfer",
        "copy",
        "wait_materialized",
        "wait_previous",
        "transfer",
        "wait_transfer",
        "copy",
    ]


def test_incoming_materialization_is_queued_for_dispatch_overlap(monkeypatch):
    calls = []

    class _FakeComputeStream:
        device = "cuda:0"

        def wait_event(self, event):
            calls.append(("wait_previous", event))

    class _FakeTransferred:
        def current_stream_wait(self):
            calls.append(("wait_outgoing",))

        def current_stream_wait_for_incoming(self):
            calls.append(("wait_incoming",))

    class _FakeDestination:
        def copy_(self, source):
            calls.append(("copy", source))

    class _FakeCopyStream:
        device = "cuda:0"

        def record_event(self):
            calls.append(("record_materialized",))
            return materialized_event

    class _FakeMaterializedEvent:
        def wait(self, stream):
            calls.append(("wait_materialized", stream))

    compute_stream = _FakeComputeStream()
    materialized_event = _FakeMaterializedEvent()
    source = object()

    def transfer_from_placement(*args, **kwargs):
        calls.append(("transfer", kwargs["completion_scope"]))
        return _FakeTransferred()

    transfer = object.__new__(UltraEPExpertTransfer)
    transfer._manager = SimpleNamespace(transfer=transfer_from_placement)
    transfer._copy_stream = _FakeCopyStream()
    transfer._copy_pairs = {3: ((_FakeDestination(), source),)}
    transfer._last_copy_event = None
    transfer._overlap_transfer_with_dispatch = True
    placement = SimpleNamespace(
        layer_id=3,
        physical_to_logical_map="p2l",
        logical_to_physical_map="l2p",
        logical_replica_counts="counts",
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *args, **kwargs: compute_stream
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    ready = transfer.apply_async(placement)

    assert calls == [
        ("transfer", "outgoing"),
        ("wait_incoming",),
        ("copy", source),
        ("record_materialized",),
    ]
    ready.current_stream_wait()
    assert calls[-1] == ("wait_materialized", compute_stream)
    assert transfer._last_copy_event is materialized_event


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
