import logging
import time
from typing import TYPE_CHECKING, List

import torch.cuda

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def on_forward_pass_end(self):
        next(self._main_generator)

    def reset_generator(self):
        self._main_generator = self._entrypoint()

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.get_device_module().synchronize()
            time_start = time.time()

        # Phase 2.B: drain + skip-gate in one MLB call. MLB owns the
        # threshold (RebalancePolicyConfig.min_utilization_threshold);
        # sglang provides the utilization rate as a plain float from its
        # own _UtilizationRateAccumulatorMixin.
        from moe_load_balancer.adapters.sglang.eplb import get_default_runtime

        mlb_runtime = get_default_runtime()

        # Pull windowed utilization from sglang's recorder (computed by
        # the mixin during per-forward append). Returns None if metric is
        # disabled or threshold is the 1.0 short-circuit -- in that case
        # MLB degrades to "always rebalance".
        observed_utilization = (
            get_global_expert_distribution_recorder().get_average_utilization_rate()
        )

        drained = mlb_runtime.drain_for_rebalance(
            observed_utilization_rate=observed_utilization,
        )
        if drained is None:
            logger.info(
                "[EPLBManager] rebalance skipped (no data or utilization "
                f"{observed_utilization} above threshold)"
            )
            return

        # Cross-rank reduction (distributed comm is rightly sglang's job).
        logical_count = drained["logical_count"]
        torch.distributed.all_reduce(logical_count, op=torch.distributed.ReduceOp.SUM)

        # Compute new placement (MLB) + sglang-shape writeback.
        from sglang.srt.eplb.expert_location import (
            ExpertLocationMetadata as _ELM,
            _mlb_eplb_active_ranks,
        )

        common = _ELM._init_common(self._server_args, self._model_runner.model_config)
        if common is None:
            return

        plan = mlb_runtime.compute_placement(
            logical_count=logical_count,
            num_physical_experts=common["num_physical_experts"],
            num_local_physical_experts=(
                common["num_physical_experts"] // common["ep_size"]
            ),
            num_groups=common["model_config_for_expert_location"].num_groups,
            num_nodes=self._server_args.nnodes,
            algorithm=self._server_args.eplb_algorithm,
            active_ranks=_mlb_eplb_active_ranks(self._server_args),
        )
        if plan is None:
            return

        expert_location_metadata = _ELM._init_raw(
            server_args=self._server_args,
            ep_size=common["ep_size"],
            physical_to_logical_map=plan["physical_to_logical_map"].to(
                self._server_args.device
            ),
            logical_to_all_physical_map=plan["logical_to_all_physical_map"].to(
                self._server_args.device
            ),
        )

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.get_device_module().synchronize()
            time_end = time.time()
            msg += f" time={time_end - time_start:.3f}s"
        logger.info(msg)

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
