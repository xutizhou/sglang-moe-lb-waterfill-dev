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


class _LoadBalancer:
    def __init__(self):
        self.policy = None

    def register_l3_policy(self, policy):
        self.policy = policy


def test_ultraep_registers_l3_on_existing_orchestrator(monkeypatch):
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
    moe_module = SimpleNamespace(
        forward_deepep=lambda: None,
        experts=experts,
        layer_id=3,
        num_fused_shared_experts=0,
        moe_load_balancer=None,
    )
    model = SimpleNamespace(modules=lambda: [moe_module])
    model_config = SimpleNamespace(hf_config=SimpleNamespace(n_routed_experts=2))
    server_args = SimpleNamespace(ultraep_num_redundant_experts_per_rank=1)
    load_balancer = _LoadBalancer()

    import moe_load_balancer.policies.l3 as l3

    monkeypatch.setattr(l3, "UltraEPL3Policy", _FakePolicy)
    monkeypatch.setattr(
        glue,
        "get_moe_ep_group",
        lambda: SimpleNamespace(device_group=object(), world_size=2),
    )

    num_layers = glue.register_ultraep_l3(
        model=model,
        model_config=model_config,
        server_args=server_args,
        moe_load_balancer=load_balancer,
    )

    assert num_layers == 1
    assert moe_module.moe_load_balancer is load_balancer
    assert isinstance(load_balancer.policy, _FakePolicy)
    layer = load_balancer.policy.kwargs["layers"][0]
    assert layer.fc1.weight.data_ptr() == w13.data_ptr()
    assert layer.fc2.weight.data_ptr() == w2.data_ptr()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
