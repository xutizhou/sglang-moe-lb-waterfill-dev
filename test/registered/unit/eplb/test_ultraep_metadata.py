from types import SimpleNamespace

import torch

from sglang.srt.eplb.expert_distribution import (
    _convert_global_physical_count_to_logical_count,
)
from sglang.srt.eplb.expert_location import ExpertLocationMetadata


def test_ultraep_metadata_normalizes_kernel_layout_once_at_commit():
    metadata = ExpertLocationMetadata._init_raw(
        server_args=SimpleNamespace(
            eplb_algorithm="ultraep",
            ep_dispatch_algorithm="ultraep",
        ),
        ep_size=2,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
        logical_to_all_physical_map=torch.tensor([[[0, 2], [1, 3]]]),
        rank_quota_prefix=torch.ones((1, 2, 2), dtype=torch.int32),
    )

    assert metadata.physical_to_logical_map.dtype == torch.int32
    assert metadata.logical_to_all_physical_map.dtype == torch.int32
    assert metadata.logical_to_all_physical_map_num_valid.dtype == torch.int32
    assert metadata.logical_to_all_physical_map.shape == (1, 2, 4)
    assert metadata.rank_quota_prefix.shape == (1, 2, 4)
    assert metadata.logical_to_all_physical_map.is_contiguous()
    assert metadata.rank_quota_prefix.is_contiguous()


def test_logical_count_ignores_unassigned_physical_slots():
    logical_count = _convert_global_physical_count_to_logical_count(
        global_physical_count=torch.tensor([[[3, 5, 7, 11]]], dtype=torch.int64),
        physical_to_logical_map=torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
        num_layers=1,
        num_logical_experts=2,
    )

    torch.testing.assert_close(
        logical_count,
        torch.tensor([[[3, 5]]], dtype=torch.int64),
    )
