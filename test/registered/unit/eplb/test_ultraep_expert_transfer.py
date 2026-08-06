import sys
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


def test_decode_reuse_skips_weight_transfer():
    transfer = object.__new__(UltraEPExpertTransfer)
    placement = SimpleNamespace(metadata={"weights_updated": False})

    assert transfer.apply_async(placement) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
