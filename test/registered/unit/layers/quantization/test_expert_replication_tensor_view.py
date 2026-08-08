import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _SemanticQuantMethod(FusedMoEMethodBase):
    def __init__(self, quant_info):
        self.quant_info = quant_info

    def apply(self, layer, dispatch_output):
        raise NotImplementedError

    def get_triton_quant_info(self, layer):
        return self.quant_info


def test_replication_view_uses_quant_semantics_without_copying_tensors():
    w13 = torch.zeros((2, 4))
    w2 = torch.zeros((2, 4))
    s13 = torch.ones((2, 1))
    s2 = torch.ones((2, 1))
    method = _SemanticQuantMethod(
        SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=s13,
            w2_scale=s2,
            a2_scale=torch.tensor(1.0),
        )
    )
    layer = SimpleNamespace(
        num_local_experts=2,
        named_per_expert_tensors=lambda _: [
            ("backend_specific_fc1_scale_name", s13),
            ("backend_specific_fc2_scale_name", s2),
        ],
    )

    view = method.get_expert_replication_tensor_view(layer)

    assert view.w13_weight is w13
    assert view.w2_weight is w2
    assert view.w13_weight_scale is s13
    assert view.w2_weight_scale is s2
    assert view.auxiliary_tensors == ()


def test_replication_view_reports_unsupported_per_expert_state():
    zero_points = torch.zeros((2, 1))
    activation_scales = torch.ones((2, 1))
    method = _SemanticQuantMethod(
        SimpleNamespace(
            w13_weight=torch.zeros((2, 4)),
            w2_weight=torch.zeros((2, 4)),
            a13_scale=activation_scales,
            w13_zp=zero_points,
        )
    )

    view = method.get_expert_replication_tensor_view(
        SimpleNamespace(num_local_experts=2)
    )

    assert view.auxiliary_tensors == (
        ("a13_scale", activation_scales),
        ("w13_zp", zero_points),
    )


def test_replication_view_does_not_drop_unmodeled_side_tensors():
    extra = torch.zeros((2, 1))
    method = _SemanticQuantMethod(
        SimpleNamespace(
            w13_weight=torch.zeros((2, 4)),
            w2_weight=torch.zeros((2, 4)),
        )
    )
    layer = SimpleNamespace(
        num_local_experts=2,
        named_per_expert_tensors=lambda _: [("backend_alpha", extra)],
    )

    view = method.get_expert_replication_tensor_view(layer)

    assert view.auxiliary_tensors == (("backend_alpha", extra),)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
