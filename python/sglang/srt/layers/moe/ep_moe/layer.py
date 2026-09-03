from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe import (
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    moe_forward_piecewise_cuda_graph_impl,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        DispatchOutput,
    )

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


logger = logging.getLogger(__name__)


class DeepEPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    Mooncake EP shares the same class, as they expose the same interface.
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )
        is_humming = (
            get_moe_runner_backend().is_humming()
            or get_moe_runner_backend().is_auto()
            and quant_config is not None
            and quant_config.get_name() == "humming"
        )
        if is_humming:
            self.deprecate_flag = True
        elif _use_aiter:
            self.deprecate_flag = True
        elif _is_npu:
            self.deprecate_flag = True
        elif deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and isinstance(
            quant_config, Fp8Config
        ):
            self.deprecate_flag = True
        elif (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            self.deprecate_flag = True
        elif (
            get_moe_runner_backend().is_flashinfer_cutedsl()
            and quant_config is not None
            and quant_config.get_name() in ("modelopt_fp4", "modelopt_mixed")
        ):
            self.deprecate_flag = True
        elif (
            quant_config is None
            and self.w13_weight.dtype == torch.bfloat16
            and get_moe_runner_backend().is_deep_gemm()
            and get_moe_a2a_backend().is_deepep()
            and get_deepep_mode().enable_low_latency()
            and not _is_npu
            and not _is_hip
        ):
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), "Unquantized DeepEP low-latency MoE requires DeepGEMM BF16"
            self.deprecate_flag = True
        else:
            self.deprecate_flag = False

        if self.deprecate_flag:
            return

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.use_w4afp8 = False
        elif isinstance(quant_config, W4AFp8Config):
            self.use_w4afp8 = True
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
        else:
            self.use_w4afp8 = False
            self.use_fp8_w8a8 = False
            self.use_block_quant = False

        self.deepep_mode = get_deepep_mode()
        if (
            self.deepep_mode.enable_low_latency()
            and not _is_npu
            and not _is_hip
            and quant_config is not None
        ):
            # AMD HIP and NPU support low_latency DeepEP without DeepGEMM.
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        if is_in_tc_piecewise_cuda_graph():
            assert TopKOutputChecker.format_is_standard(
                topk_output
            ), "Only standard topk output is supported for piecewise cuda graph"
            return moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.topk_weights,
                topk_output.topk_ids,
                topk_output.router_logits,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

        lplb_solver = get_global_lplb_solver(self.layer_id)
        if (
            lplb_solver is not None
            and lplb_solver.consume_metro_allgather_decode()
        ):
            return self._forward_metro_allgather(
                hidden_states, topk_output, lplb_solver
            )

        if self.deprecate_flag:
            return super().forward_impl(
                hidden_states,
                topk_output,
            )

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )
        combine_input = self.run_moe_core(dispatch_output)
        return self.dispatcher.combine(combine_input=combine_input)

    def _forward_metro_allgather(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        lplb_solver,
    ) -> torch.Tensor:
        """Paper path: all-gather inputs, local experts, reduce-scatter output."""
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.layers.dp_attention import (
            get_dp_global_num_tokens,
            get_local_dp_buffer,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatcher,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        assert TopKOutputChecker.format_is_standard(topk_output)
        # metro_allgather deliberately leaves TopK IDs logical.  Replica
        # selection happens once, below, from the global active set.
        logical_ids = topk_output.topk_ids
        # StandardDispatcher may consume/reshape its input storage in-place.
        # Preserve the model width before dispatch so the reduce-scatter result
        # is not sliced with a mutated trailing dimension.
        hidden_size = hidden_states.shape[-1]

        group = get_tp_group()
        sizes = get_dp_global_num_tokens()
        if hidden_states.shape[0] == sum(sizes):
            # The paper path gathers hidden states before top-k in the model
            # layer, so weights and logical IDs are already global here.
            global_weights = topk_output.topk_weights
            global_logical_ids = logical_ids
            global_hidden = hidden_states
        else:
            # Fallback for direct DeepEPMoE callers used by unit tests.
            global_weights, global_logical_ids, global_hidden = group.all_gatherv(
                [topk_output.topk_weights, logical_ids, hidden_states],
                sizes=sizes,
            )
        global_physical_ids = lplb_solver.route_decode_metro_global(
            global_logical_ids
        )
        global_topk = StandardTopKOutput(
            topk_weights=global_weights,
            topk_ids=global_physical_ids,
            router_logits=topk_output.router_logits,
        )

        if not hasattr(self, "_metro_standard_dispatcher"):
            self._metro_standard_dispatcher = StandardDispatcher(
                self.moe_runner_config
            )
        dispatch_output = self._metro_standard_dispatcher.dispatch(
            global_hidden, global_topk
        )
        combine_input = self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
        )
        global_output = self._metro_standard_dispatcher.combine(combine_input)
        local_output = get_local_dp_buffer(group)
        group.reduce_scatterv(global_output, output=local_output, sizes=sizes)
        return local_output[..., :hidden_size].contiguous()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        return self.dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

    def run_moe_core(
        self,
        dispatch_output: DispatchOutput,
    ):

        if self.deprecate_flag:
            return super().run_moe_core(dispatch_output)

        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            if self.quant_config is None:
                raise NotImplementedError(
                    "Unquantized DeepEP MoE currently supports low_latency mode only"
                )
            elif self.use_w4afp8:
                output = self.forward_cutlass_w4afp8(dispatch_output)
            else:
                assert False, "forward_deepgemm_contiguous is deprecated"
        elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            if self.use_w4afp8:
                output = self.forward_cutlass_w4afp8_masked(dispatch_output)
            else:
                assert False, "forward_deepgemm_masked is deprecated"

        combine_input_wrapper = (
            DeepEPNormalCombineInput
            if DispatchOutputChecker.format_is_deepep_normal(dispatch_output)
            else DeepEPLLCombineInput
        )

        return combine_input_wrapper(
            hidden_states=output,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[Dict[str, Any]] = None,
    ):
        return self.dispatcher.combine(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )

    def forward_cutlass_w4afp8(
        self,
        dispatch_output: DeepEPNormalDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_normal(
            layer=self,
            dispatch_output=dispatch_output,
        )

    def forward_cutlass_w4afp8_masked(
        self,
        dispatch_output: DeepEPLLDispatchOutput,
    ):
        assert self.moe_runner_config.activation == "silu"
        assert isinstance(self.quant_method, W4AFp8MoEMethod)
        return self.quant_method.apply_deepep_ll(
            layer=self,
            dispatch_output=dispatch_output,
        )


def get_moe_impl_class(quant_config: Optional[QuantizationConfig]):
    # [TODO] kk, temporary solution
    if (
        get_moe_a2a_backend().is_mori()
        or get_moe_a2a_backend().is_deepep()
        or get_moe_a2a_backend().is_mooncake()
        or get_moe_a2a_backend().is_nixl()
    ):
        return DeepEPMoE
    return FusedMoE
