from __future__ import annotations

import sys
from collections import namedtuple
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from moe_load_balancer import L3Placement, RoutingDecision

from sglang.srt.eplb import moe_load_balancer_glue as glue
from sglang.srt.layers.moe.token_dispatcher.base import _PostDispatchHooks
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_TopKOutput = namedtuple("_TopKOutput", ("topk_weights", "topk_ids", "router_logits"))


class _LoadBalancer:
    def __init__(self, events):
        self.events = events
        self.placement_request = None
        self.routing_request = None
        self.placement = None
        self.placements = []

    def compute_placement(self, request):
        self.events.append("placement")
        self.placement_request = request
        generation = len(self.placements) + 1
        if generation % 2:
            physical_to_logical_map = torch.tensor([0, 1])
        else:
            physical_to_logical_map = torch.tensor([1, 0])
        self.placement = L3Placement(
            layer_id=3,
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=torch.tensor([[0], [1]]),
            logical_replica_counts=torch.tensor([1, 1]),
            metadata={"placement_generation": generation},
        )
        self.placements.append(self.placement)
        return self.placement

    def route_tokens(self, request):
        self.events.append("routing")
        self.routing_request = request
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids + 10,
            topk_weights=request.topk_weights,
        )


class _ReadyEvent:
    def __init__(self, events, label="wait"):
        self.events = events
        self.label = label

    def current_stream_wait(self):
        self.events.append(self.label)


class _Transfer:
    def __init__(self, events):
        self.events = events
        self.placement = None
        self.communication_stream = self._CommunicationStream(events)

    class _CommunicationStream:
        def __init__(self, events):
            self.events = events

        def wait_stream(self, stream):
            self.events.append(("wait_consumer", stream))

    def record_placement_ready(self, layer_id):
        self.events.append(("placement_ready", layer_id))
        return _ReadyEvent(self.events, "wait_placement")

    def apply_async(self, placement):
        self.events.append("transfer")
        self.placement = placement
        return _ReadyEvent(self.events, "wait_materialized")


class _BalanceProfiler:
    def __init__(self, events):
        self.events = events
        self.kwargs = None

    def record_refresh(self, **kwargs):
        self.events.append("profile")
        self.kwargs = kwargs


class _Dispatcher:
    def __init__(self, events):
        self.events = events
        self.post_dispatch_hooks = []

    def register_post_dispatch_hook(self, hook):
        self.post_dispatch_hooks.append(hook)

        class _Handle:
            def remove(inner_self):
                self.post_dispatch_hooks.remove(hook)

        return _Handle()

    def run_post_dispatch_hooks(self, dispatch_output):
        for hook in tuple(self.post_dispatch_hooks):
            hook(self, dispatch_output)

    def combine(self, *, combine_input):
        self.events.append("combine")
        return combine_input


class _Experts:
    should_fuse_routed_scaling_factor_in_topk = True

    def __init__(self, events):
        self.events = events
        self.dispatcher = _Dispatcher(events)
        self.dispatched_topk = None

    def dispatch(self, hidden_states, topk_output):
        self.events.append("dispatch")
        self.dispatched_topk = topk_output
        self.dispatcher.run_post_dispatch_hooks(hidden_states)
        return hidden_states

    def run_moe_core(self, dispatch_output):
        self.events.append("moe")
        return dispatch_output

    def __call__(self, *, hidden_states, topk_output):
        self.events.append("experts_forward")
        dispatch_output = self.dispatch(hidden_states, topk_output)
        combine_input = self.run_moe_core(dispatch_output)
        return self.dispatcher.combine(combine_input=combine_input)


class _TopK:
    def __init__(self, output):
        self.output = output

    def __call__(self, *args, **kwargs):
        return self.output


def test_ultraep_forward_routes_through_l2_with_the_transfer_placement(monkeypatch):
    monkeypatch.setattr(
        deepseek_v2.envs,
        "SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO",
        SimpleNamespace(get=lambda: False),
    )
    events = []
    collect_calls = []
    load_buffers = object()
    logical_loads_per_rank = torch.tensor([[1, 1], [0, 0]], dtype=torch.int32)

    def collect_loads(**kwargs):
        collect_calls.append(kwargs)
        events.append("loads")
        return SimpleNamespace(
            loads_per_rank=logical_loads_per_rank,
            current_stream_wait=lambda: events.append("wait_loads"),
        )

    monkeypatch.setattr(
        glue,
        "collect_ultraep_logical_loads",
        collect_loads,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    logical_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4]])
    load_balancer = _LoadBalancer(events)
    transfer = _Transfer(events)
    balance_profiler = _BalanceProfiler(events)
    experts = _Experts(events)
    module = SimpleNamespace(
        _fuse_shared_experts_inside_sbo=False,
        is_nextn=False,
        num_fused_shared_experts=0,
        alt_stream=None,
        is_hash=False,
        layer_id=3,
        config=SimpleNamespace(n_routed_experts=2),
        routed_scaling_factor=1.0,
        gate=lambda *args, **kwargs: torch.zeros((1, 2)),
        topk=_TopK(_TopKOutput(weights, logical_ids, None)),
        _forward_shared_experts=lambda hidden_states: None,
        moe_load_balancer=load_balancer,
        expert_transfer=transfer,
        _ultraep_load_buffers=load_buffers,
        _ultraep_balance_profiler=balance_profiler,
        _ultraep_active_placement=None,
        _ultraep_refresh_interval=16,
        _ultraep_refresh_min_tokens=1,
        _ultraep_prefill_batches_since_refresh=0,
        experts=experts,
    )
    forward_batch = SimpleNamespace(
        num_token_non_padded=None,
        forward_mode=SimpleNamespace(name="EXTEND"),
    )
    hidden_states = torch.ones((1, 4))

    output = deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )

    placement_request = load_balancer.placement_request
    routing_request = load_balancer.routing_request
    assert placement_request.logical_topk_ids.dtype == torch.int32
    assert placement_request.logical_topk_ids.is_contiguous()
    assert routing_request.logical_topk_ids is placement_request.logical_topk_ids
    assert placement_request.logical_loads_per_rank is logical_loads_per_rank
    assert len(collect_calls) == 1
    assert collect_calls[0]["buffers"] is load_buffers
    assert collect_calls[0]["expert_transfer"] is transfer
    assert routing_request.transient_placement is load_balancer.placement
    assert transfer.placement is load_balancer.placement
    assert balance_profiler.kwargs["placement"] is load_balancer.placement
    assert balance_profiler.kwargs["routed_physical_topk_ids"].tolist() == [[10, 11]]
    assert tuple(policy.name for policy in routing_request.policies) == ("ultraep",)
    assert placement_request.stage == "prefill"
    assert routing_request.stage == "prefill"
    assert module._ultraep_active_placement is load_balancer.placement
    assert module._ultraep_prefill_batches_since_refresh == 0
    assert experts.dispatched_topk.topk_ids.tolist() == [[10, 11]]
    assert events == [
        "loads",
        "placement",
        ("placement_ready", 3),
        "transfer",
        "wait_placement",
        "routing",
        "wait_loads",
        "profile",
        "experts_forward",
        "dispatch",
        "wait_materialized",
        "moe",
        "combine",
    ]
    assert output is hidden_states


def test_ultraep_refresh_interval_reuses_then_refreshes_current_batch(monkeypatch):
    monkeypatch.setattr(
        deepseek_v2.envs,
        "SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO",
        SimpleNamespace(get=lambda: False),
    )
    events = []
    loads = torch.tensor([[1, 1], [0, 0]], dtype=torch.int32)

    def collect_loads(**kwargs):
        events.append("loads")
        return SimpleNamespace(loads_per_rank=loads)

    monkeypatch.setattr(
        glue,
        "collect_ultraep_logical_loads",
        collect_loads,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    logical_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4]])
    load_balancer = _LoadBalancer(events)
    transfer = _Transfer(events)
    experts = _Experts(events)
    module = SimpleNamespace(
        _fuse_shared_experts_inside_sbo=False,
        is_nextn=False,
        num_fused_shared_experts=0,
        alt_stream=None,
        is_hash=False,
        layer_id=3,
        config=SimpleNamespace(n_routed_experts=2),
        routed_scaling_factor=1.0,
        gate=lambda *args, **kwargs: torch.zeros((1, 2)),
        topk=_TopK(_TopKOutput(weights, logical_ids, None)),
        _forward_shared_experts=lambda hidden_states: None,
        moe_load_balancer=load_balancer,
        expert_transfer=transfer,
        _ultraep_load_buffers=object(),
        _ultraep_balance_profiler=None,
        _ultraep_active_placement=None,
        _ultraep_refresh_interval=2,
        _ultraep_refresh_min_tokens=512,
        _ultraep_prefill_batches_since_refresh=0,
        experts=experts,
    )
    forward_batch = SimpleNamespace(
        num_token_non_padded=None,
        num_token_non_padded_cpu=1024,
        original_global_num_tokens_cpu=None,
        is_extend_in_batch=False,
        forward_mode=SimpleNamespace(name="EXTEND"),
    )
    hidden_states = torch.ones((1, 4))

    # Bootstrap generation 1 exactly from the current batch.
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    generation_1 = module._ultraep_active_placement
    events.clear()

    # Reuse generation 1 until the configured prefill interval expires.
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    assert module._ultraep_active_placement is generation_1
    assert events == [
        "routing",
        "experts_forward",
        "dispatch",
        "moe",
        "combine",
    ]
    assert module._ultraep_prefill_batches_since_refresh == 1
    events.clear()

    # The due refresh solves from this batch before L2 and transfers only once.
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    generation_2 = module._ultraep_active_placement
    assert generation_2 is not None and generation_2 is not generation_1
    assert generation_2.metadata["placement_generation"] == 2
    assert module._ultraep_prefill_batches_since_refresh == 0
    assert events == [
        "loads",
        "placement",
        ("placement_ready", 3),
        "transfer",
        "wait_placement",
        "routing",
        "experts_forward",
        "dispatch",
        "wait_materialized",
        "moe",
        "combine",
    ]

    # Once due, a tiny scheduler split keeps the cadence saturated and reuses
    # the active placement rather than replacing it with a noisy sample.
    events.clear()
    module._ultraep_refresh_interval = 1
    forward_batch.num_token_non_padded_cpu = 64
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    assert module._ultraep_active_placement is generation_2
    assert module._ultraep_prefill_batches_since_refresh == 1
    assert events == [
        "routing",
        "experts_forward",
        "dispatch",
        "moe",
        "combine",
    ]

    # DP-attention publishes rank-identical global metadata. Its real-token sum
    # controls the collective even when this rank's local stage is idle.
    events.clear()
    forward_batch.original_global_num_tokens_cpu = [1024, 0]
    forward_batch.is_extend_in_batch = True
    forward_batch.forward_mode.name = "IDLE"
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    generation_3 = module._ultraep_active_placement
    assert generation_3 is not generation_2
    assert generation_3.metadata["placement_generation"] == 3
    assert module._ultraep_prefill_batches_since_refresh == 0
    assert events[:5] == [
        "loads",
        "placement",
        ("placement_ready", 3),
        "transfer",
        "wait_placement",
    ]

    # Every refresh keeps solve -> transfer asynchronous. A CUDA tensor equality
    # check here would synchronize the CPU merely to skip a transfer, while real
    # workload placements almost always change.
    events.clear()
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    generation_4 = module._ultraep_active_placement
    assert generation_4 is not generation_3
    assert generation_4.metadata["placement_generation"] == 4
    assert events == [
        "loads",
        "placement",
        ("placement_ready", 3),
        "transfer",
        "wait_placement",
        "routing",
        "experts_forward",
        "dispatch",
        "wait_materialized",
        "moe",
        "combine",
    ]


def test_ultraep_tiny_bootstrap_primes_next_global_prefill_refresh(monkeypatch):
    monkeypatch.setattr(
        deepseek_v2.envs,
        "SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO",
        SimpleNamespace(get=lambda: False),
    )
    events = []
    loads = torch.tensor([[1, 1], [0, 0]], dtype=torch.int32)

    def collect_loads(**kwargs):
        events.append("loads")
        return SimpleNamespace(loads_per_rank=loads)

    monkeypatch.setattr(
        glue,
        "collect_ultraep_logical_loads",
        collect_loads,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    logical_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    weights = torch.tensor([[0.6, 0.4]])
    load_balancer = _LoadBalancer(events)
    module = SimpleNamespace(
        _fuse_shared_experts_inside_sbo=False,
        is_nextn=False,
        num_fused_shared_experts=0,
        alt_stream=None,
        is_hash=False,
        layer_id=3,
        config=SimpleNamespace(n_routed_experts=2),
        routed_scaling_factor=1.0,
        gate=lambda *args, **kwargs: torch.zeros((1, 2)),
        topk=_TopK(_TopKOutput(weights, logical_ids, None)),
        _forward_shared_experts=lambda hidden_states: None,
        moe_load_balancer=load_balancer,
        expert_transfer=_Transfer(events),
        _ultraep_load_buffers=object(),
        _ultraep_balance_profiler=None,
        _ultraep_active_placement=None,
        _ultraep_refresh_interval=2,
        _ultraep_refresh_min_tokens=512,
        _ultraep_prefill_batches_since_refresh=0,
        experts=_Experts(events),
    )
    forward_batch = SimpleNamespace(
        num_token_non_padded=None,
        num_token_non_padded_cpu=64,
        original_global_num_tokens_cpu=[64, 0],
        is_extend_in_batch=False,
        forward_mode=SimpleNamespace(name="IDLE"),
    )
    hidden_states = torch.ones((1, 4))

    # Bootstrap remains exact even for an unrepresentative idle slice.
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    bootstrap = module._ultraep_active_placement
    assert bootstrap is not None
    assert module._ultraep_prefill_batches_since_refresh == 2

    # A second idle slice reuses placement and preserves the saturated cadence.
    events.clear()
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    assert module._ultraep_active_placement is bootstrap
    assert module._ultraep_prefill_batches_since_refresh == 2
    assert "loads" not in events

    # Rank-identical global metadata makes every EP rank refresh together even
    # though this rank's local stage remains idle.
    events.clear()
    forward_batch.original_global_num_tokens_cpu = [1024, 0]
    forward_batch.is_extend_in_batch = True
    deepseek_v2.DeepseekV2MoE.forward_deepep(
        module,
        hidden_states,
        forward_batch,
    )
    refreshed = module._ultraep_active_placement
    assert refreshed is not bootstrap
    assert refreshed.metadata["placement_generation"] == 2
    assert module._ultraep_prefill_batches_since_refresh == 0
    assert events[:5] == [
        "loads",
        "placement",
        ("placement_ready", 3),
        "transfer",
        "wait_placement",
    ]


def test_post_dispatch_hooks_allow_multiple_one_shot_hooks():
    hooks = _PostDispatchHooks()
    events = []
    handles = {}

    def first(_dispatcher, dispatch_output):
        events.append("first")
        handles["first"].remove()
        return dispatch_output

    def second(_dispatcher, dispatch_output):
        events.append("second")
        handles["second"].remove()
        return dispatch_output

    handles["first"] = hooks.register_hook(first)
    handles["second"] = hooks.register_hook(second)
    dispatch_output = object()

    assert hooks(None, dispatch_output) is dispatch_output
    assert events == ["first", "second"]
    assert len(hooks.hook_dict) == 0

    assert hooks(None, dispatch_output) is dispatch_output
    assert events == ["first", "second"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
