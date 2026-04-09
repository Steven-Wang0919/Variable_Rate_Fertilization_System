from __future__ import annotations

import json
import tempfile
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
from vrf_system.domain import Bounds, MachineConfig, PrescriptionCell
from vrf_system.exporters import export_simulation_result
from vrf_system.prescription import PrescriptionMap
from vrf_system.routing_spec import (
    ABOVE_GLOBAL_SUPPORT,
    BELOW_GLOBAL_SUPPORT,
    EXPERIMENTAL_EXTRAPOLATION,
    IN_GLOBAL_SUPPORT,
    INVERSE_KAN,
    INVERSE_MLP,
    NORMAL_IN_RANGE,
)
from vrf_system.simulator import FieldSimulator


ROUTING_COVERAGE_SAMPLE = Path("samples/prescription_grid_routing_coverage.csv")


class StubRouter:
    def select_strategy_opening(self, target_mass_g_min: float) -> float:
        return 20.0

    def predict(self, target_mass_g_min: float, opening_mm: float) -> tuple[float, float, str, str, str, str, str]:
        return 30.0, 30.0, INVERSE_KAN, IN_GLOBAL_SUPPORT, NORMAL_IN_RANGE, "", "HIGH"


def build_centerline_regression_map() -> PrescriptionMap:
    cell = PrescriptionCell(
        cell_id="cell-1",
        center_x_m=1.5,
        center_y_m=3.0,
        width_m=3.0,
        height_m=6.0,
        target_rate_kg_ha=260.0,
        zone_id="A",
    )
    return PrescriptionMap(
        cells=[cell],
        bounds=Bounds(min_x=0.0, max_x=3.0, min_y=0.0, max_y=6.0),
        source_path=Path("tests/centerline_regression.csv"),
    )


class FieldAlignmentTests(unittest.TestCase):
    def test_simulation_should_align_first_pass_rows_to_centerlines(self) -> None:
        result = FieldSimulator(StubRouter()).run(
            build_centerline_regression_map(),
            MachineConfig(row_count=6, row_spacing_m=0.6, travel_speed_kmh=6.0, sample_period_ms=300),
        )

        first_frame_rows = [item for item in result.frames[0].row_decisions if item.row_state == "IN_FIELD"]
        self.assertEqual([round(item.y_m, 1) for item in first_frame_rows], [0.3, 0.9, 1.5, 2.1, 2.7, 3.3])
        self.assertTrue(all(item.y_m > result.prescription_cells[0].bottom for item in first_frame_rows))

    def test_export_should_preserve_centerline_row_coordinates(self) -> None:
        result = FieldSimulator(StubRouter()).run(
            build_centerline_regression_map(),
            MachineConfig(row_count=6, row_spacing_m=0.6, travel_speed_kmh=6.0, sample_period_ms=300),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = export_simulation_result(result, output_root=Path(tmp_dir), highlighted_frame_index=0)
            timeline_df = pd.read_csv(artifacts.row_command_timeline)
            routing_df = pd.read_csv(artifacts.model_routing_trace)

        timeline_first_frame = timeline_df.loc[timeline_df["timestamp_ms"] == 0].sort_values("row_index")
        routing_first_frame = routing_df.loc[routing_df["timestamp_ms"] == 0].sort_values("row_index")
        expected_centerlines = [0.3, 0.9, 1.5, 2.1, 2.7, 3.3]

        self.assertEqual([round(value, 1) for value in timeline_first_frame["y_m"].tolist()], expected_centerlines)
        self.assertEqual([round(value, 1) for value in routing_first_frame["y_m"].tolist()], expected_centerlines)


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
        first_frame_rows = [item for item in result.frames[0].row_decisions if item.row_state == "IN_FIELD"]
        expected_centerlines = [0.3, 0.9, 1.5, 2.1, 2.7, 3.3]
        self.assertEqual([round(item.y_m, 1) for item in first_frame_rows], expected_centerlines)
        self.assertTrue(all(item.y_m > self.controller.prescription_map.bounds.min_y for item in first_frame_rows))

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
            timeline_df = pd.read_csv(artifacts.row_command_timeline)
            self.assertIn("model_route", routing_df.columns)
            self.assertIn("route_mode", routing_df.columns)
            self.assertIn("speed_r_min_cmd", routing_df.columns)
            in_field_routing = routing_df.loc[routing_df["row_state"] == "IN_FIELD"]
            self.assertTrue((in_field_routing["raw_speed_r_min"] == in_field_routing["speed_r_min_cmd"]).all())
            routing_first_frame = routing_df.loc[routing_df["timestamp_ms"] == 0].sort_values("row_index")
            timeline_first_frame = timeline_df.loc[timeline_df["timestamp_ms"] == 0].sort_values("row_index")
            self.assertEqual([round(value, 1) for value in routing_first_frame["y_m"].tolist()], expected_centerlines)
            self.assertEqual([round(value, 1) for value in timeline_first_frame["y_m"].tolist()], expected_centerlines)

            summary = json.loads(artifacts.simulation_summary.read_text(encoding="utf-8"))
            self.assertIn("experimental_extrapolation_count", summary)
            self.assertGreater(summary["total_row_decisions"], 0)
            self.assertEqual(summary["speed_clip_state_counts"], {})
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
