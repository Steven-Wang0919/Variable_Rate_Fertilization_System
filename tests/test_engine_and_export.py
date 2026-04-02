from __future__ import annotations

import json
import tempfile
import unittest

import pandas as pd

from vrf_system.controller import SimulationController
from vrf_system.defaults import DEFAULT_KAN_ARTIFACT_DIR, DEFAULT_MLP_ARTIFACT_DIR, DEFAULT_SAMPLE_PRESCRIPTION
from vrf_system.domain import MachineConfig


@unittest.skipUnless(
    DEFAULT_KAN_ARTIFACT_DIR.exists() and DEFAULT_MLP_ARTIFACT_DIR.exists() and DEFAULT_SAMPLE_PRESCRIPTION.exists(),
    "默认模型或样例处方不存在，跳过集成测试。",
)
class EngineAndExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = SimulationController()
        self.controller.load_default_models()
        self.controller.load_sample_prescription()

    def test_simulation_should_route_and_export(self) -> None:
        config = MachineConfig(
            row_count=6,
            row_spacing_m=0.6,
            travel_speed_kmh=6.0,
            sample_period_ms=300,
        )
        result = self.controller.run_simulation(config)
        decisions = result.flatten_decisions()
        self.assertGreater(len(result.frames), 0)
        self.assertTrue(any(item.selected_model == "inverse_KAN" for item in decisions))
        self.assertTrue(any(item.selected_model == "inverse_MLP" for item in decisions))
        self.assertTrue(any(item.status == "out_of_field" for item in decisions))
        self.assertTrue(
            any(
                len({round(item.target_speed_r_min, 2) for item in frame.row_decisions if item.status == "ok"}) > 1
                for frame in result.frames
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self.controller.export_last_result(tmp)
            self.assertTrue(artifacts.row_command_timeline.exists())
            self.assertTrue(artifacts.model_routing_trace.exists())
            self.assertTrue(artifacts.simulation_summary.exists())
            self.assertTrue(artifacts.map_overview_png.exists())
            self.assertTrue(artifacts.map_current_frame_png.exists())
            self.assertTrue(artifacts.map_legend_png.exists())

            routing_df = pd.read_csv(artifacts.model_routing_trace)
            self.assertIn("selected_model", routing_df.columns)
            self.assertIn("domain_status", routing_df.columns)

            summary = json.loads(artifacts.simulation_summary.read_text(encoding="utf-8"))
            self.assertGreater(summary["extrapolation_count"], 0)
            self.assertGreater(summary["total_row_decisions"], 0)
            self.assertIn("visual_assets", summary)
