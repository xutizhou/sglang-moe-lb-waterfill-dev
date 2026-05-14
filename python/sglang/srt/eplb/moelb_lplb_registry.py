# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Global per-layer registry for LPLB runtimes.

When ``--ep-dispatch-algorithm=lp`` is enabled, the SGLang model runner
builds one
:class:`moe_load_balancer.adapters.sglang.lplb.runtime.LPLBRuntime` per MoE
layer and registers it here. ``ExpertLocationDispatchInfo.init_new`` looks
the runtime up by ``layer_id`` so the per-layer TopK hook can route through
the moe_load_balancer SDK without an explicit ``runtime`` argument
threading through every call site.

Kept structurally separate from the SGLang-internal kernel modules so the
glue branch stays additive: nothing in the LPLB-free baseline references
this file, and existing dispatch paths (``static``, ``dynamic``, ``fake``)
are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from moe_load_balancer.adapters.sglang.lplb.runtime import LPLBRuntime

_GLOBAL_LPLB_RUNTIMES: dict[int, "LPLBRuntime"] = {}


def get_global_lplb_runtime(layer_id: int) -> Optional["LPLBRuntime"]:
    """Return the registered LPLB runtime for ``layer_id`` or ``None``."""
    return _GLOBAL_LPLB_RUNTIMES.get(layer_id)


def set_global_lplb_runtime(layer_id: int, runtime: "LPLBRuntime") -> None:
    """Register a runtime for ``layer_id``.

    Called by ``ModelRunner._init_lplb_runtimes`` once per layer at startup,
    and re-called after EPLB rebalances if the placement adapter needs to
    be rebuilt (e.g. when the placement metadata is replaced rather than
    mutated in place).
    """
    _GLOBAL_LPLB_RUNTIMES[layer_id] = runtime


def clear_global_lplb_runtimes() -> None:
    """Clear every registered runtime.

    Intended for full re-init paths (model reload, EPLB rebalance that
    swaps in a new ``ExpertLocationMetadata`` object).
    """
    _GLOBAL_LPLB_RUNTIMES.clear()
