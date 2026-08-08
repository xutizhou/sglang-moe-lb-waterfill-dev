"""SGLang-owned expert transfer services."""

from __future__ import annotations

_expert_transfer = None


def get_expert_transfer():
    return _expert_transfer


def set_expert_transfer(value) -> None:
    global _expert_transfer
    _expert_transfer = value


def start_expert_transfer(layer_id: int, experts):
    if _expert_transfer is None:
        return None
    from sglang.srt.eplb.expert_placement_state import (
        get_global_expert_placement_state,
    )

    placement = get_global_expert_placement_state().pending(layer_id)
    if placement is None:
        return None
    return _expert_transfer.transfer(layer_id, experts, placement)


def finish_expert_transfer(layer_id: int, event) -> None:
    if event is None:
        return
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.eplb.expert_placement_state import (
        get_global_expert_placement_state,
    )

    event.current_stream_wait()
    get_global_expert_placement_state().commit(
        layer_id,
        get_global_expert_location_metadata(),
    )


def close_expert_transfer() -> None:
    global _expert_transfer
    if _expert_transfer is not None:
        _expert_transfer.close()
        _expert_transfer = None


__all__ = [
    "close_expert_transfer",
    "finish_expert_transfer",
    "get_expert_transfer",
    "set_expert_transfer",
    "start_expert_transfer",
]
