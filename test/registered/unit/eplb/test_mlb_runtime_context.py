"""MLB bootstrap and placement commit integration with RuntimeContext."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.eplb.eplb_manager import (
    EPLBManager,
    update_expert_location_with_recovery,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.runtime_context import ParallelContext, RuntimeContext
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMLBRuntimeContext(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("moe_load_balancer"),
        "moe_load_balancer is not installed",
    )
    def test_bootstrap_uses_bound_context_before_metadata_exists(self):
        from moe_load_balancer import MoELoadBalancer

        context = RuntimeContext(ParallelContext())
        model_config = object()
        with (
            context.override_server_args(
                moe_load_balancer_algorithm="static",
                ep_size=2,
                ep_num_redundant_experts=2,
                _model_config=model_config,
            ),
            context.parallel.override(moe_ep_rank=1),
            patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=ModelConfigForExpertLocation(1, 2, 1),
            ) as extract,
        ):
            self.assertIsNone(context.resources.expert_location_metadata)
            mlb = MoELoadBalancer.from_sglang_context(context)
            self.assertIs(type(mlb), MoELoadBalancer)
            self.assertIs(mlb.runtime_context, context)
            layout = ExpertLocationMetadata._init_common(model_config, context=context)
            self.assertEqual(layout["num_physical_experts"], 4)
            self.assertEqual(layout["num_local_physical_experts"], 2)
            extract.assert_called_with(model_config)

    def test_topk_modules_share_the_load_balancer_and_commit_their_layers(self):
        from sglang.srt.layers.moe.hash_topk import HashTopK
        from sglang.srt.layers.moe.topk import TopK
        from sglang.srt.model_executor.model_runner import ModelRunner

        topk = Mock(spec=TopK, layer_id=2)
        hash_topk = Mock(spec=HashTopK, layer_id=3)
        mlb = Mock()
        runner = SimpleNamespace(
            moe_load_balancer=mlb,
            model=SimpleNamespace(modules=lambda: [object(), topk, hash_topk]),
        )
        ModelRunner._prepare_moe_topk(runner)
        self.assertIs(topk.moe_load_balancer, mlb)
        self.assertIs(hash_topk.moe_load_balancer, mlb)
        mlb.commit_placement.assert_called_once_with([2, 3])

    def test_rebalance_commits_each_chunk_to_the_same_load_balancer(self):
        order = []
        mlb = Mock()
        mlb.commit_placement.side_effect = lambda ids: order.append(("commit", ids))
        updater = Mock()
        updater.update.side_effect = lambda *a, **kw: (
            order.append(("move", kw["update_layer_ids"])) or {}
        )
        manager = Mock(
            _rebalance_disabled_reason=None,
            _rebalance_layers_per_chunk=1,
            _moe_load_balancer=mlb,
            _ps=SimpleNamespace(tp_rank=0),
        )
        manager._compute_update_layer_ids_chunks.return_value = [[2], [3]]
        manager._check_rebalance_needed.return_value = True
        manager._get_expert_location_updater.return_value = updater
        recorder = Mock()
        recorder.dump_record.return_value = {
            "logical_count": object(),
            "average_utilization_rate_over_window": 0.5,
        }
        module = "sglang.srt.eplb.eplb_manager"
        with (
            patch(f"{module}.ElasticEPStateManager.instance", return_value=None),
            patch(
                f"{module}.get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch(f"{module}.get_parallel", return_value=SimpleNamespace(nnodes=1)),
            patch(
                f"{module}.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_dispatch_algorithm=None)
                ),
            ),
        ):
            list(EPLBManager.rebalance(manager))
        self.assertEqual(
            order, [("move", [2]), ("commit", [2]), ("move", [3]), ("commit", [3])]
        )

    def test_commit_follows_weight_update_and_recovery(self):
        order = []
        updater = Mock()
        updater.update.side_effect = lambda *a, **kw: order.append("move") or {0: [1]}
        backup = Mock(use_backup=True)
        backup.update_weights.side_effect = lambda *a: order.append("recover")
        model = SimpleNamespace(
            routed_experts_weights_of_layer={},
            generate_weight_name_filter=lambda missing: missing,
        )
        committed = Mock(side_effect=lambda layers: order.append("commit"))
        update_expert_location_with_recovery(
            expert_location_updater=updater,
            model=model,
            new_expert_location_metadata=object(),
            update_layer_ids=[0],
            nnodes=1,
            tp_rank=0,
            expert_backup_client=backup,
            update_weights_from_disk_callable=Mock(),
            ep_dispatch_algorithm=None,
            init_lplb_solvers_callable=Mock(),
            on_placement_committed=committed,
        )
        self.assertEqual(order, ["move", "recover", "commit"])
        committed.assert_called_once_with([0])

    def test_failed_recovery_does_not_commit(self):
        updater = Mock()
        updater.update.return_value = {0: [1]}
        backup = Mock(use_backup=True)
        backup.update_weights.side_effect = RuntimeError("recovery failed")
        committed = Mock()
        with self.assertRaisesRegex(RuntimeError, "recovery failed"):
            update_expert_location_with_recovery(
                expert_location_updater=updater,
                model=SimpleNamespace(routed_experts_weights_of_layer={}),
                new_expert_location_metadata=object(),
                update_layer_ids=[0],
                nnodes=1,
                tp_rank=0,
                expert_backup_client=backup,
                update_weights_from_disk_callable=Mock(),
                ep_dispatch_algorithm=None,
                init_lplb_solvers_callable=Mock(),
                on_placement_committed=committed,
            )
        committed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
