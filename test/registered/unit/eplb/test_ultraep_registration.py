import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.eplb import moe_load_balancer_glue as glue
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _QuantMethod:
    def __init__(self, view):
        self.view = view

    def get_expert_replication_tensor_view(self, experts):
        return self.view


class _FakePolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRouter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTransfer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.nvl_domain_size = 2
        self.logical_loads_per_rank = torch.empty(
            (2, kwargs["num_logical_experts"]), dtype=torch.int32
        )


class _LoadBalancer:
    def __init__(self):
        self.l3_policy = None
        self.routing_policies = {}
        self.registrations = []

    def register_l3_policy(self, policy):
        self.l3_policy = policy
        self.registrations.append(("l3", policy))

    def register_routing_policy(self, name, policy):
        self.routing_policies[name] = policy
        self.registrations.append(("l2", name, policy))


def test_ultraep_attaches_independent_l3_l2_and_transfer(monkeypatch):
    profiler_args = {}

    class _FakeLoadProfiler:
        def __init__(self, **kwargs):
            profiler_args.update(kwargs)
            self.enabled = kwargs["group_rank"] == 0

    w13 = torch.zeros((2, 4))
    w2 = torch.zeros((2, 4))
    view = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=None,
        w2_weight_scale=None,
        auxiliary_tensors=(),
    )
    experts = SimpleNamespace(quant_method=_QuantMethod(view))

    def make_moe_module(layer_id):
        return SimpleNamespace(
            forward_deepep=lambda: None,
            experts=experts,
            layer_id=layer_id,
            num_fused_shared_experts=0,
            supports_ultraep_external_transfer=True,
            moe_load_balancer=None,
            expert_transfer=None,
            _ultraep_active_placement=object(),
            _ultraep_refresh_interval=-1,
            _ultraep_refresh_min_tokens=-1,
            _ultraep_prefill_batches_since_refresh=-1,
        )

    moe_module = make_moe_module(3)
    second_moe_module = make_moe_module(4)
    model = SimpleNamespace(modules=lambda: [moe_module, second_moe_module])
    model_config = SimpleNamespace(hf_config=SimpleNamespace(n_routed_experts=2))
    server_args = SimpleNamespace(
        ultraep_num_redundant_experts_per_rank=1,
        ultraep_placement_refresh_interval=16,
        ultraep_placement_refresh_min_tokens=512,
        deepep_mode="normal",
    )
    load_balancer = _LoadBalancer()
    ep_device_group = object()
    ep_group = SimpleNamespace(
        device_group=ep_device_group,
        world_size=2,
        rank_in_group=1,
        rank=9,
        ranks=[8, 9],
        unique_name="moe_ep:7",
        all_reduce=lambda tensor: pytest.fail("profiling must not run during attach"),
    )

    import moe_load_balancer.kernels.ultraep.profiling as profiling
    import moe_load_balancer.policies.l2.ultraep as l2
    import moe_load_balancer.policies.l3 as l3

    import sglang.srt.eplb.ultraep_expert_transfer as transfer

    monkeypatch.setattr(l2, "UltraEPL2Router", _FakeRouter)
    monkeypatch.setattr(l3, "UltraEPL3Policy", _FakePolicy)
    monkeypatch.setattr(transfer, "UltraEPExpertTransfer", _FakeTransfer)
    monkeypatch.setattr(
        profiling,
        "load_profile_config",
        lambda: SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr(profiling, "ExpertLoadProfiler", _FakeLoadProfiler)
    monkeypatch.setattr(
        glue,
        "get_moe_ep_group",
        lambda: ep_group,
    )

    num_layers, expert_transfer = glue.attach_ultraep(
        model=model,
        model_config=model_config,
        server_args=server_args,
        moe_load_balancer=load_balancer,
    )

    assert num_layers == 2
    for module in (moe_module, second_moe_module):
        assert module.moe_load_balancer is load_balancer
        assert module.expert_transfer is expert_transfer
        assert module._ultraep_active_placement is None
        assert module._ultraep_refresh_interval == 16
        assert module._ultraep_refresh_min_tokens == 512
        assert module._ultraep_prefill_batches_since_refresh == 0
    assert isinstance(expert_transfer, _FakeTransfer)
    assert moe_module._ultraep_load_buffers is second_moe_module._ultraep_load_buffers
    load_buffers = moe_module._ultraep_load_buffers
    assert load_buffers.logical_loads_per_rank.shape == (2, 2)
    assert load_buffers.profile_global_logical_loads.shape == (2,)
    assert load_buffers.profile_physical_loads.shape == (4,)
    assert (
        load_buffers.logical_loads_per_rank.data_ptr()
        == expert_transfer.logical_loads_per_rank.data_ptr()
    )
    assert isinstance(load_balancer.l3_policy, _FakePolicy)
    assert isinstance(load_balancer.routing_policies["ultraep"], _FakeRouter)
    assert load_balancer.l3_policy.kwargs["layer_ids"] == (3, 4)
    assert load_balancer.l3_policy.kwargs["rank"] == 1
    assert load_balancer.l3_policy.kwargs["num_nvl_ranks"] == 2
    assert "manager" not in load_balancer.l3_policy.kwargs
    assert load_balancer.routing_policies["ultraep"].kwargs == {}
    assert load_balancer.registrations == [
        ("l3", load_balancer.l3_policy),
        ("l2", "ultraep", load_balancer.routing_policies["ultraep"]),
    ]
    assert expert_transfer.kwargs["layer_storages"] == {3: view, 4: view}
    assert expert_transfer.kwargs["group"] is ep_device_group
    assert expert_transfer.kwargs["overlap_transfer_with_dispatch"] is True
    assert (
        moe_module._ultraep_balance_profiler
        is second_moe_module._ultraep_balance_profiler
    )
    assert profiler_args["group_rank"] == 1
    assert profiler_args["global_rank"] == 9
    metadata = profiler_args["metadata"]
    assert metadata["framework"] == "sglang"
    assert metadata["ep_group_id"] == "moe_ep-7_r8-9"
    assert metadata["ep_group_unique_name"] == "moe_ep:7"
    assert metadata["ep_group_ranks"] == [8, 9]
    assert metadata["global_rank"] == 9
    assert metadata["ep_rank"] == 1


def test_ultraep_rejects_unmarked_deepep_moe_module():
    unsupported = SimpleNamespace(
        forward_deepep=lambda: None,
        experts=SimpleNamespace(),
        layer_id=0,
    )
    model = SimpleNamespace(modules=lambda: [unsupported])
    model_config = SimpleNamespace(hf_config=SimpleNamespace(n_routed_experts=2))
    server_args = SimpleNamespace(ultraep_num_redundant_experts_per_rank=1)

    with pytest.raises(ValueError, match="external-transfer protocol"):
        glue.attach_ultraep(
            model=model,
            model_config=model_config,
            server_args=server_args,
            moe_load_balancer=_LoadBalancer(),
        )


def test_disabled_balance_profile_has_no_buffers_or_collective(monkeypatch):
    collective_calls = []
    monkeypatch.setattr(
        glue.torch.distributed,
        "all_reduce",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    buffers = glue.UltraEPLoadBuffers.allocate(
        ep_size=2,
        num_logical_experts=3,
        device=torch.device("cpu"),
    )
    ep_group = SimpleNamespace()

    adapter = glue._create_ultraep_balance_profiler(
        config=SimpleNamespace(enabled=False),
        ep_group=ep_group,
        buffers=buffers,
        num_layers=2,
        num_logical_experts=3,
        num_redundant_experts_per_rank=1,
        nvl_domain_size=2,
    )

    assert adapter is None
    assert buffers.profile_global_logical_loads is None
    assert buffers.profile_physical_loads is None
    assert collective_calls == []


def test_balance_profile_records_actual_rerouted_physical_loads(monkeypatch):
    records = []
    collective_data_ptrs = []

    class _Profiler:
        enabled = True

        def stage_pre(self, layer_id, real_layer_id, loads):
            records.append(("pre", layer_id, real_layer_id, loads.clone()))

        def record_post(self, layer_id, loads, placement):
            records.append(("post", layer_id, loads.clone(), placement))

    ep_device_group = object()

    def all_reduce(loads, *, group):
        assert group is ep_device_group
        collective_data_ptrs.append(loads.data_ptr())
        loads.add_(torch.tensor([1, 0, 0, 1, 0], dtype=torch.int32))

    monkeypatch.setattr(glue.torch.distributed, "all_reduce", all_reduce)

    buffers = glue.UltraEPLoadBuffers.allocate(
        ep_size=2,
        num_logical_experts=3,
        device=torch.device("cpu"),
        profile_num_physical_experts=5,
    )
    buffers.logical_loads_per_rank.copy_(
        torch.tensor([[2, 1, 0], [1, 0, 2]], dtype=torch.int32)
    )
    adapter = glue.UltraEPBalanceProfiler(
        profiler=_Profiler(),
        ep_group=SimpleNamespace(device_group=ep_device_group),
        buffers=buffers,
    )
    physical_to_logical_map = torch.tensor([0, 1, 2, 0, 2], dtype=torch.int32)
    placement = SimpleNamespace(
        physical_to_logical_map=physical_to_logical_map,
    )

    adapter.record_refresh(
        layer_id=3,
        routed_physical_topk_ids=torch.tensor([[4, 4], [1, -1]]),
        placement=placement,
    )

    assert collective_data_ptrs == [buffers.profile_physical_loads.data_ptr()]
    assert records[0][:3] == ("pre", 3, 3)
    assert records[0][3].tolist() == [3, 1, 2]
    assert records[1][:2] == ("post", 3)
    assert records[1][2].tolist() == [1, 1, 0, 1, 2]
    assert records[1][3] is physical_to_logical_map


@pytest.mark.parametrize(
    ("forward_mode", "stage"),
    [
        ("DRAFT_EXTEND_V2", "speculative"),
        ("SPLIT_PREFILL", "prefill"),
        ("DLLM_EXTEND", "prefill"),
    ],
)
def test_stage_from_forward_batch_covers_extend_variants(forward_mode, stage):
    forward_batch = SimpleNamespace(forward_mode=SimpleNamespace(name=forward_mode))

    assert glue.stage_from_forward_batch(forward_batch) == stage


def test_ultraep_load_gather_reuses_ultraep_owned_result_buffer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        glue,
        "count_logical_experts",
        lambda *args, **kwargs: pytest.fail(
            "SGLang must not launch a separate local-count kernel"
        ),
    )
    buffers = glue.UltraEPLoadBuffers.allocate(
        ep_size=2,
        num_logical_experts=3,
        device=torch.device("cpu"),
    )
    per_rank_data_ptr = buffers.logical_loads_per_rank.data_ptr()

    class _Collected:
        def __init__(self, loads_per_rank):
            self.loads_per_rank = loads_per_rank

    class _FakeTransfer:
        def collect_topk_loads_async(self, logical_topk_ids):
            calls.append(("collect",))
            assert logical_topk_ids is logical_ids
            buffers.logical_loads_per_rank.copy_(
                torch.tensor([[2, 1, 0], [3, 2, 1]], dtype=torch.int32)
            )
            return _Collected(buffers.logical_loads_per_rank)

    transfer = _FakeTransfer()
    logical_ids = torch.tensor([[0, 1], [0, -1]], dtype=torch.int64)

    first_collected = glue.collect_ultraep_logical_loads(
        logical_topk_ids=logical_ids,
        buffers=buffers,
        expert_transfer=transfer,
    )
    second_collected = glue.collect_ultraep_logical_loads(
        logical_topk_ids=logical_ids,
        buffers=buffers,
        expert_transfer=transfer,
    )

    assert first_collected.loads_per_rank is buffers.logical_loads_per_rank
    assert second_collected.loads_per_rank is buffers.logical_loads_per_rank
    assert first_collected.loads_per_rank.data_ptr() == per_rank_data_ptr
    assert second_collected.loads_per_rank.data_ptr() == per_rank_data_ptr
    assert second_collected.loads_per_rank.dtype is torch.int32
    assert second_collected.loads_per_rank.is_contiguous()
    assert second_collected.loads_per_rank.tolist() == [[2, 1, 0], [3, 2, 1]]
    assert calls == [
        ("collect",),
        ("collect",),
    ]


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("update_weights_from_disk", ("unused", "dummy")),
        ("update_weights_from_distributed", ([], [], [], "unused")),
        ("update_weights_from_tensor", ([],)),
        ("update_weights_from_ipc", (object(),)),
    ],
)
def test_ultraep_rejects_online_weight_update_entry_points(method_name, args):
    from sglang.srt.model_executor.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(enable_ultraep=True)
    runner.ultraep_expert_transfer = object()

    succeeded, message = getattr(runner, method_name)(*args)

    assert succeeded is False
    assert "registered expert storage must remain stable" in message


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
