from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.expert_transfer.ultraep import UltraEPExpertTransfer


def test_transfer_uses_only_current_layer_tensors():
    group = SimpleNamespace(world_size=2, device_group=object())
    manager = MagicMock()
    manager.transfer_from_placement.return_value = object()
    gathered = torch.empty((2, 4), dtype=torch.int32)
    gather_event = MagicMock()
    manager.all_gather_loads.return_value = (gathered, gather_event)
    manager_type = MagicMock(return_value=manager)
    get_quant_info = MagicMock(
        return_value=SimpleNamespace(
            w13_weight=torch.empty((3, 4)),
            w2_weight=torch.empty((3, 2)),
            w13_scale=None,
            w2_scale=None,
            b13=None,
            b2=None,
            w13_zp=None,
            w2_zp=None,
        )
    )
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(get_triton_quant_info=get_quant_info)
    )
    placement = SimpleNamespace(
        physical_to_logical=torch.tensor([0, 1, 0], dtype=torch.int32),
        logical_to_physical=torch.tensor([[0, 2], [1, -1]], dtype=torch.int32),
        replica_counts=torch.tensor([2, 1], dtype=torch.int32),
    )
    with (
        patch(
            "sglang.srt.expert_transfer.ultraep.get_moe_ep_group",
            return_value=group,
        ),
        patch("torch.cuda.Stream") as stream_type,
        patch("torch.cuda.current_stream", return_value=object()),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch.dict(
            "sys.modules",
            {
                "ultra_ep": SimpleNamespace(
                    Manager=manager_type,
                    init_runtime=MagicMock(return_value=2),
                )
            },
        ),
    ):
        transfer = UltraEPExpertTransfer(
            num_layers=4,
            num_logical_experts=4,
            num_redundant_per_rank=1,
        )
        result = transfer.transfer(3, experts, placement)
        gathered_result = transfer.all_gather_loads(torch.empty(4, dtype=torch.int32))

    assert result is manager.transfer_from_placement.return_value
    get_quant_info.assert_called_once_with(experts)
    stream_type.return_value.wait_stream.assert_called_once()
    manager.transfer_from_placement.assert_called_once()
    assert gathered_result is gathered
    manager.all_gather_loads.assert_called_once()
    gather_event.current_stream_wait.assert_called_once()
