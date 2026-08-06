from __future__ import annotations

import sys
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from moe_load_balancer import L3Placement, RoutingDecision
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_TopKOutput = namedtuple("_TopKOutput", ("topk_weights", "topk_ids", "router_logits"))


class _LoadBalancer:
    def __init__(self, events):
        self.events = events
        self.placement_request = None
        self.routing_request = None
        self.placement = L3Placement(
            layer_id=3,
            physical_to_logical_map=torch.tensor([0, 1]),
            logical_to_physical_map=torch.tensor([[0], [1]]),
            logical_replica_counts=torch.tensor([1, 1]),
        )

    def compute_placement(self, request):
        self.events.append("placement")
        self.placement_request = request
        return self.placement

    def route_tokens(self, request):
        self.events.append("routing")
        self.routing_request = request
        return RoutingDecision(
            routed_physical_topk_ids=request.logical_topk_ids + 10,
            topk_weights=request.topk_weights,
        )


class _ReadyEvent:
    def __init__(self, events):
        self.events = events

    def current_stream_wait(self):
        self.events.append("wait")


class _Transfer:
    def __init__(self, events):
        self.events = events
        self.placement = None

    def apply_async(self, placement):
        self.events.append("transfer")
        self.placement = placement
        return _ReadyEvent(self.events)


class _Dispatcher:
    def __init__(self, events):
        self.events = events

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
        return hidden_states

    def run_moe_core(self, dispatch_output):
        self.events.append("moe")
        return dispatch_output


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
        routed_scaling_factor=1.0,
        gate=lambda *args, **kwargs: torch.zeros((1, 2)),
        topk=_TopK(_TopKOutput(weights, logical_ids, None)),
        _forward_shared_experts=lambda hidden_states: None,
        moe_load_balancer=load_balancer,
        expert_transfer=transfer,
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
    assert placement_request.logical_topk_ids.dtype == torch.int64
    assert placement_request.logical_topk_ids.is_contiguous()
    assert routing_request.logical_topk_ids is placement_request.logical_topk_ids
    assert routing_request.transient_placement is load_balancer.placement
    assert transfer.placement is load_balancer.placement
    assert tuple(policy.name for policy in routing_request.policies) == ("ultraep",)
    assert placement_request.stage == "prefill"
    assert routing_request.stage == "prefill"
    assert experts.dispatched_topk.topk_ids.tolist() == [[10, 11]]
    assert events == [
        "placement",
        "transfer",
        "routing",
        "dispatch",
        "wait",
        "moe",
        "combine",
    ]
    assert output is hidden_states


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
