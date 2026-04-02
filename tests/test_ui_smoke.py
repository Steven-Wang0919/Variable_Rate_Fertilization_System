from __future__ import annotations

import unittest
from pathlib import Path
from tkinter import TclError

from vrf_system.annotations import OUT_OF_FIELD_TEXT, format_row_annotation
from vrf_system.domain import Bounds, MachineConfig, PrescriptionCell, RowDecision, SimulationFrame, SimulationResult
from vrf_system.prescription import PrescriptionMap
from vrf_system.ui import FertilizerApp


def build_sample_prescription_map() -> PrescriptionMap:
    cells = [
        PrescriptionCell(
            cell_id="cell-1",
            center_x_m=1.5,
            center_y_m=1.0,
            width_m=3.0,
            height_m=2.0,
            target_rate_kg_ha=180.0,
            zone_id="A1",
        ),
        PrescriptionCell(
            cell_id="cell-2",
            center_x_m=4.5,
            center_y_m=1.0,
            width_m=3.0,
            height_m=2.0,
            target_rate_kg_ha=260.0,
            zone_id="B2",
        ),
    ]
    return PrescriptionMap(
        cells=cells,
        bounds=Bounds(min_x=0.0, max_x=6.0, min_y=0.0, max_y=2.0),
        source_path=Path("tests/sample_prescription.csv"),
    )


def build_sample_result() -> SimulationResult:
    prescription = build_sample_prescription_map()
    frames = [
        SimulationFrame(
            timestamp_ms=0,
            pass_id=1,
            machine_center_x_m=1.4,
            machine_center_y_m=1.0,
            direction_sign=1,
            row_decisions=[
                RowDecision(
                    timestamp_ms=0,
                    pass_id=1,
                    row_index=1,
                    x_m=1.0,
                    y_m=0.8,
                    zone_id="A1",
                    target_rate_kg_ha=180.0,
                    target_mass_g_min=108.0,
                    strategy_opening_mm=20.0,
                    target_speed_r_min=298.75,
                    selected_model="inverse_KAN",
                    domain_status="in_domain",
                    status="ok",
                ),
                RowDecision(
                    timestamp_ms=0,
                    pass_id=1,
                    row_index=2,
                    x_m=1.8,
                    y_m=1.2,
                    zone_id="A1",
                    target_rate_kg_ha=180.0,
                    target_mass_g_min=108.0,
                    strategy_opening_mm=35.0,
                    target_speed_r_min=312.40,
                    selected_model="inverse_MLP",
                    domain_status="extrapolated",
                    status="ok",
                ),
            ],
        ),
        SimulationFrame(
            timestamp_ms=500,
            pass_id=1,
            machine_center_x_m=4.2,
            machine_center_y_m=1.0,
            direction_sign=1,
            row_decisions=[
                RowDecision(
                    timestamp_ms=500,
                    pass_id=1,
                    row_index=1,
                    x_m=3.8,
                    y_m=0.8,
                    zone_id="B2",
                    target_rate_kg_ha=260.0,
                    target_mass_g_min=156.0,
                    strategy_opening_mm=35.0,
                    target_speed_r_min=365.25,
                    selected_model="inverse_KAN",
                    domain_status="in_domain",
                    status="ok",
                ),
                RowDecision(
                    timestamp_ms=500,
                    pass_id=1,
                    row_index=2,
                    x_m=4.6,
                    y_m=1.2,
                    zone_id="",
                    target_rate_kg_ha=0.0,
                    target_mass_g_min=0.0,
                    strategy_opening_mm=0.0,
                    target_speed_r_min=0.0,
                    selected_model="none",
                    domain_status="out_of_field",
                    status="out_of_field",
                ),
            ],
        ),
    ]
    return SimulationResult(
        frames=frames,
        machine_config=MachineConfig(row_count=2, row_spacing_m=0.6, travel_speed_kmh=6.0, sample_period_ms=200),
        prescription_path=prescription.source_path,
        prescription_cells=prescription.cells,
        summary={
            "frame_count": 2,
            "pass_count": 1,
            "total_row_decisions": 4,
            "extrapolation_count": 1,
            "selected_model_counts": {
                "inverse_KAN": 2,
                "inverse_MLP": 1,
            },
            "status_counts": {
                "ok": 3,
                "out_of_field": 1,
            },
            "domain_status_counts": {
                "in_domain": 2,
                "extrapolated": 1,
            },
            "average_target_rate_kg_ha": 220.0,
            "average_target_speed_r_min": 325.47,
        },
    )


class UISmokeTests(unittest.TestCase):
    def create_app(self) -> FertilizerApp:
        try:
            app = FertilizerApp(auto_load_models=False)
        except TclError as exc:
            self.skipTest(f"当前环境无法创建 tkinter 窗口：{exc}")
            raise
        app.withdraw()
        app.update_idletasks()
        return app

    def test_app_can_create_notebook_and_destroy(self) -> None:
        app = self.create_app()
        tab_texts = [app.preview_notebook.tab(tab_id, "text") for tab_id in app.preview_notebook.tabs()]
        self.assertEqual(tab_texts, ["总览图", "当前帧细节图", "图例"])
        self.assertEqual(str(app.frame_slider.cget("state")), "disabled")
        self.assertGreaterEqual(int(app.frame_slider.cget("width")), 20)
        self.assertGreaterEqual(int(app.frame_slider.cget("sliderlength")), 30)
        app.destroy()

    def test_preview_tabs_should_show_empty_state_without_prescription_or_result(self) -> None:
        app = self.create_app()
        app._render_preview_tabs()
        app.update_idletasks()

        self.assertIn("请先导入处方图并运行仿真", app.preview_status_vars["overview"].get())
        self.assertIn("请先导入处方图并运行仿真", app.preview_status_vars["current"].get())
        self.assertIn("请先导入处方图并运行仿真", app.preview_status_vars["legend"].get())
        self.assertEqual(app.preview_canvases, {})
        app.destroy()

    def test_loading_prescription_should_render_overview_and_legend_before_simulation(self) -> None:
        app = self.create_app()
        app.controller.prescription_map = build_sample_prescription_map()

        app._render_preview_tabs()
        app.update_idletasks()

        self.assertEqual(sorted(app.preview_canvases.keys()), ["legend", "overview"])
        self.assertIn("运行仿真后可查看当前帧细节", app.preview_status_vars["current"].get())
        app.destroy()

    def test_slider_change_should_live_update_frame_info_and_release_should_refresh_table(self) -> None:
        app = self.create_app()
        prescription = build_sample_prescription_map()
        result = build_sample_result()

        app.controller.prescription_map = prescription
        app.controller.last_result = result
        app.frame_slider.configure(to=len(result.frames) - 1)
        app._set_result_state(has_result=True)
        app._refresh_summary()
        app._refresh_current_frame_details(refresh_table=True)
        app._render_preview_tabs(force=True)
        app.update_idletasks()
        overview_canvas = app.preview_canvases["overview"]
        current_canvas = app.preview_canvases["current"]

        self.assertEqual(app.current_frame_index, 0)
        self.assertEqual(len(app.decision_table.get_children()), 2)
        self.assertEqual(sorted(app.preview_canvases.keys()), ["current", "legend", "overview"])

        app.frame_slider.set(1)
        app._on_slider_changed("1")
        app.update_idletasks()

        self.assertEqual(app.current_frame_index, 1)
        self.assertIn("500 ms", app.frame_info_var.get())
        self.assertEqual(len(app.decision_table.get_children()), len(result.frames[0].row_decisions))
        self.assertTrue(app._table_dirty)
        self.assertEqual(app._pending_live_frame_index, 1)

        app._on_slider_released(None)  # type: ignore[arg-type]
        app.update_idletasks()

        self.assertEqual(len(app.decision_table.get_children()), len(result.frames[1].row_decisions))
        self.assertFalse(app._table_dirty)
        self.assertEqual(app._table_frame_index, 1)
        self.assertEqual(sorted(app.preview_canvases.keys()), ["current", "legend", "overview"])
        self.assertIs(app.preview_canvases["overview"], overview_canvas)
        self.assertIs(app.preview_canvases["current"], current_canvas)
        app.destroy()

    def test_switching_to_dirty_current_tab_should_render_detailed_final_frame(self) -> None:
        app = self.create_app()
        prescription = build_sample_prescription_map()
        result = build_sample_result()

        app.controller.prescription_map = prescription
        app.controller.last_result = result
        app.frame_slider.configure(to=len(result.frames) - 1)
        app._set_result_state(has_result=True)
        app._refresh_summary()
        app._refresh_current_frame_details(refresh_table=True)
        app._render_preview_tabs(force=True)
        app.update_idletasks()

        app.frame_slider.set(1)
        app._on_slider_changed("1")
        self.assertIn("current", app.preview_dirty_keys)

        app.preview_notebook.select(app.preview_tab_ids["current"])
        app._on_preview_tab_changed(None)
        app.update_idletasks()

        current_renderer = app.preview_renderers["current"]
        self.assertEqual(current_renderer.rendered_frame_index, 1)
        self.assertTrue(current_renderer.detailed)
        self.assertNotIn("current", app.preview_dirty_keys)
        app.destroy()

    def test_out_of_field_label_should_use_status_text(self) -> None:
        decision = RowDecision(
            timestamp_ms=0,
            pass_id=1,
            row_index=3,
            x_m=0.0,
            y_m=0.0,
            zone_id="",
            target_rate_kg_ha=0.0,
            target_mass_g_min=0.0,
            strategy_opening_mm=0.0,
            target_speed_r_min=0.0,
            selected_model="none",
            domain_status="out_of_field",
            status="out_of_field",
        )
        self.assertEqual(format_row_annotation(decision), f"R3 | {OUT_OF_FIELD_TEXT}")
