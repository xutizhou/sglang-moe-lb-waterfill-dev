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
    def test_bootstrap_uses_explicit_context_before_metadata_exists(self):
        from moe_load_balancer import MoELoadBalancer
        from moe_load_balancer.adapters.sglang import create_load_balancer

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
            mlb = create_load_balancer(context)
            self.assertIs(type(mlb), MoELoadBalancer)
            layout = ExpertLocationMetadata._init_common(model_config, context=context)
            self.assertEqual(layout["num_physical_experts"], 4)
            self.assertEqual(layout["num_local_physical_experts"], 2)
            extract.assert_called_with(model_config)

    @unittest.skipUnless(
        importlib.util.find_spec("moe_load_balancer"),
        "moe_load_balancer is not installed",
    )
    def test_model_runner_creates_core_through_context_adapter(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        context, mlb = object(), object()
        with (
            patch("sglang.srt.runtime_context.get_context", return_value=context),
            patch(
                "sglang.srt.model_executor.model_runner.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_load_balancer_algorithm="static")
                ),
            ),
            patch(
                "moe_load_balancer.adapters.sglang.create_load_balancer",
                return_value=mlb,
            ) as create,
        ):
            result = ModelRunner._create_moe_load_balancer(
                SimpleNamespace(is_draft_worker=False)
            )
        self.assertIs(result, mlb)
        create.assert_called_once_with(context)

    @unittest.skipUnless(
        importlib.util.find_spec("moe_load_balancer"),
        "moe_load_balancer is not installed",
    )
    def test_placement_planning_uses_context_adapter(self):
        import torch
        from moe_load_balancer import MoELoadBalancer

        module = "sglang.srt.eplb.expert_location"
        mlb = Mock(spec=MoELoadBalancer)
        context, model_config = object(), object()
        counts = torch.tensor([[[10, 20]]])
        mapping = torch.tensor([[0, 1, 0, 1]])
        maps = SimpleNamespace(
            physical_to_logical_map=mapping, logical_to_all_physical_map=mapping
        )
        layout = dict(
            ep_size=2,
            num_physical_experts=4,
            model_config_for_expert_location=SimpleNamespace(num_groups=1),
        )
        with (
            patch(f"{module}.get_device", return_value=SimpleNamespace(device="cpu")),
            patch("sglang.srt.runtime_context.get_context", return_value=context),
            patch.object(ExpertLocationMetadata, "_init_common", return_value=layout),
            patch.object(ExpertLocationMetadata, "_init_raw") as initialize,
            patch(
                "sglang.srt.elastic_ep.elastic_ep.ElasticEPStateManager.instance",
                return_value=None,
            ),
            patch(
                "moe_load_balancer.adapters.sglang.plan_placement", return_value=maps
            ) as plan,
        ):
            result = ExpertLocationMetadata.init_by_eplb(
                model_config,
                counts,
                use_flat_topology=True,
                moe_load_balancer=mlb,
            )
        plan.assert_called_once_with(
            mlb, context, counts, use_flat_topology=True, active_ranks=None
        )
        self.assertIs(result, initialize.return_value)
        self.assertIs(initialize.call_args.kwargs["physical_to_logical_map"], mapping)

    @unittest.skipUnless(
        importlib.util.find_spec("moe_load_balancer"),
        "moe_load_balancer is not installed",
    )
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
        context = object()
        with (
            patch("sglang.srt.runtime_context.get_context", return_value=context),
            patch("moe_load_balancer.adapters.sglang.commit_placement") as commit,
        ):
            ModelRunner._prepare_moe_topk(runner)
        self.assertIs(topk.moe_load_balancer, mlb)
        self.assertIs(hash_topk.moe_load_balancer, mlb)
        commit.assert_called_once_with(mlb, context, [2, 3])

    @unittest.skipUnless(
        importlib.util.find_spec("moe_load_balancer"),
        "moe_load_balancer is not installed",
    )
    def test_rebalance_commits_each_chunk_to_the_same_load_balancer(self):
        order = []
        mlb = Mock()
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
            patch(
                "moe_load_balancer.adapters.sglang.commit_placement",
                side_effect=lambda core, context, ids: order.append(("commit", ids)),
            ) as commit,
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
        self.assertTrue(all(call.args[0] is mlb for call in commit.call_args_list))

    @unittest.skipUnless(
        importlib.util.find_spec("moe_load_balancer"),
        "moe_load_balancer is not installed",
    )
    def test_glue_routes_once_and_records_current_resources(self):
        from sglang.srt.eplb import moe_load_balancer_glue as glue

        output = SimpleNamespace(
            topk_ids=object(),
            topk_weights=object(),
            router_logits=object(),
            recorded_physical_topk_ids=object(),
        )
        context = SimpleNamespace(
            resources=SimpleNamespace(
                expert_distribution_recorder=Mock(),
                experts_capturer=Mock(),
            )
        )
        mlb, topk = object(), object()
        with (
            patch.object(glue, "get_context", return_value=context),
            patch(
                "moe_load_balancer.adapters.sglang.route_topk", return_value=output
            ) as route,
        ):
            result = glue.route_topk_with_mlb(
                moe_load_balancer=mlb,
                layer_id=3,
                topk_output=topk,
                num_tokens=2,
                num_token_non_padded=None,
                forward_batch=None,
                routed_scaling_factor=1.0,
            )
        route.assert_called_once_with(
            mlb,
            context,
            layer_id=3,
            topk_output=topk,
            token_count=2,
            stage=None,
            routed_scaling_factor=1.0,
        )
        context.resources.expert_distribution_recorder.on_select_experts.assert_called_once_with(
            topk_ids=output.recorded_physical_topk_ids
        )
        context.resources.experts_capturer.capture.assert_called_once_with(
            layer_id=3, topk_indices=output.recorded_physical_topk_ids
        )
        self.assertIs(result.topk_ids, output.topk_ids)
        self.assertIs(result.topk_weights, output.topk_weights)
        self.assertIs(result.router_logits, output.router_logits)

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
