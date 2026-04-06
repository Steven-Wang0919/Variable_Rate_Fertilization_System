from __future__ import annotations

import json
import unittest
from pathlib import Path
import shutil

import pandas as pd

from vrf_system.controller import SimulationController
from vrf_system.defaults import (
    DEFAULT_FORWARD_KAN_ARTIFACT_DIR,
    DEFAULT_FORWARD_MLP_ARTIFACT_DIR,
    DEFAULT_KAN_ARTIFACT_DIR,
    DEFAULT_MLP_ARTIFACT_DIR,
    DEFAULT_SAMPLE_PRESCRIPTION,
)
from vrf_system.domain import MachineConfig
from vrf_system.routing_spec import (
    ABOVE_GLOBAL_SUPPORT,
    BELOW_GLOBAL_SUPPORT,
    EXPERIMENTAL_EXTRAPOLATION,
    IN_GLOBAL_SUPPORT,
    INVERSE_KAN,
    INVERSE_MLP,
    NORMAL_IN_RANGE,
)


ROUTING_COVERAGE_SAMPLE = Path("samples/prescription_grid_routing_coverage.csv")


@unittest.skipUnless(
    DEFAULT_KAN_ARTIFACT_DIR.exists()
    and DEFAULT_MLP_ARTIFACT_DIR.exists()
    and DEFAULT_FORWARD_KAN_ARTIFACT_DIR.exists()
    and DEFAULT_FORWARD_MLP_ARTIFACT_DIR.exists()
    and DEFAULT_SAMPLE_PRESCRIPTION.exists(),
    "默认模型或样例处方不存在，跳过集成测试。",
)
class EngineAndExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = SimulationController()
        self.controller.load_default_models()
        self.controller.load_default_forward_models()
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
        self.assertTrue(any(item.model_route == "inverse_KAN" for item in decisions if item.row_state == "IN_FIELD"))
        self.assertTrue(any(item.row_state == "OUT_OF_FIELD" for item in decisions))
        self.assertTrue(
            all(item.model_route in {"inverse_KAN", "inverse_MLP"} for item in decisions if item.row_state == "IN_FIELD")
        )
        self.assertTrue(
            any(
                len({round(item.speed_r_min_cmd, 2) for item in frame.row_decisions if item.row_state == "IN_FIELD"}) > 1
                for frame in result.frames
            )
        )

        outputs_root = Path.cwd() / "outputs"
        outputs_root.mkdir(exist_ok=True)
        export_root = outputs_root / "test_engine_and_export"
        shutil.rmtree(export_root, ignore_errors=True)
        export_root.mkdir(parents=True)
        try:
            artifacts = self.controller.export_last_result(export_root)
            self.assertTrue(artifacts.row_command_timeline.exists())
            self.assertTrue(artifacts.model_routing_trace.exists())
            self.assertTrue(artifacts.simulation_summary.exists())
            self.assertTrue(artifacts.map_overview_png.exists())
            self.assertTrue(artifacts.map_current_frame_png.exists())
            self.assertTrue(artifacts.map_legend_png.exists())

            routing_df = pd.read_csv(artifacts.model_routing_trace)
            self.assertIn("model_route", routing_df.columns)
            self.assertIn("route_mode", routing_df.columns)
            self.assertIn("speed_r_min_cmd", routing_df.columns)

            summary = json.loads(artifacts.simulation_summary.read_text(encoding="utf-8"))
            self.assertIn("experimental_extrapolation_count", summary)
            self.assertGreater(summary["total_row_decisions"], 0)
            self.assertIn("visual_assets", summary)
        finally:
            shutil.rmtree(export_root, ignore_errors=True)

    def test_routing_coverage_sample_should_cover_all_inverse_route_bands(self) -> None:
        self.controller.load_prescription(ROUTING_COVERAGE_SAMPLE)

        result = self.controller.run_simulation(MachineConfig())
        decisions = [item for item in result.flatten_decisions() if item.row_state == "IN_FIELD"]

        self.assertGreater(len(decisions), 0)
        self.assertEqual(
            {item.model_route for item in decisions},
            {INVERSE_KAN, INVERSE_MLP},
        )
        self.assertEqual(
            {item.mass_state for item in decisions},
            {BELOW_GLOBAL_SUPPORT, IN_GLOBAL_SUPPORT, ABOVE_GLOBAL_SUPPORT},
        )
        self.assertEqual(
            {item.route_mode for item in decisions},
            {NORMAL_IN_RANGE, EXPERIMENTAL_EXTRAPOLATION},
        )
        self.assertEqual(
            {item.opening_mm for item in decisions},
            {20.0, 35.0, 50.0},
        )

        zone_expectations = {
            "EL": (BELOW_GLOBAL_SUPPORT, 20.0, INVERSE_MLP),
            "L": (IN_GLOBAL_SUPPORT, 20.0, INVERSE_KAN),
            "M": (IN_GLOBAL_SUPPORT, 35.0, INVERSE_KAN),
            "H": (IN_GLOBAL_SUPPORT, 50.0, INVERSE_KAN),
            "EH": (ABOVE_GLOBAL_SUPPORT, 50.0, INVERSE_KAN),
        }
        for zone_id, (mass_state, opening_mm, model_route) in zone_expectations.items():
            zone_decisions = [item for item in decisions if item.zone_id == zone_id]
            self.assertGreater(len(zone_decisions), 0, zone_id)
            self.assertTrue(all(item.mass_state == mass_state for item in zone_decisions), zone_id)
            self.assertTrue(all(item.opening_mm == opening_mm for item in zone_decisions), zone_id)
            self.assertTrue(all(item.model_route == model_route for item in zone_decisions), zone_id)

        summary = result.summary
        self.assertEqual(
            set(summary["model_route_counts"].keys()),
            {INVERSE_KAN, INVERSE_MLP},
        )
        self.assertEqual(
            set(summary["mass_state_counts"].keys()),
            {BELOW_GLOBAL_SUPPORT, IN_GLOBAL_SUPPORT, ABOVE_GLOBAL_SUPPORT},
        )
        self.assertEqual(
            set(summary["route_mode_counts"].keys()),
            {NORMAL_IN_RANGE, EXPERIMENTAL_EXTRAPOLATION},
        )
