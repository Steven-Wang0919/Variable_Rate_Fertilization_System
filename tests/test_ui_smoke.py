from __future__ import annotations

import unittest
from tkinter import TclError

from vrf_system.annotations import OUT_OF_FIELD_TEXT, format_row_annotation
from vrf_system.domain import RowDecision
from vrf_system.ui import FertilizerApp


class UISmokeTests(unittest.TestCase):
    def _assert_label_within_plot(
        self,
        app: FertilizerApp,
        *,
        row_x: float,
        row_y: float,
        row_index: int,
        label_text: str,
        width: int,
        height: int,
        expected_anchor: str,
    ) -> None:
        left, top, right, bottom = app._plot_area(width, height)
        label_x, label_y, anchor = app._row_label_position(row_x, row_y, row_index, label_text, width, height)
        label_width, label_height = app._measure_text_block(label_text, app.row_annotation_font)
        half_height = label_height / 2.0

        self.assertEqual(anchor, expected_anchor)
        self.assertGreaterEqual(label_y - half_height, top)
        self.assertLessEqual(label_y + half_height, bottom)
        if anchor == "e":
            self.assertGreaterEqual(label_x - label_width, left)
            self.assertLessEqual(label_x, right)
        else:
            self.assertGreaterEqual(label_x, left)
            self.assertLessEqual(label_x + label_width, right)

    def test_app_can_create_and_destroy(self) -> None:
        try:
            app = FertilizerApp(auto_load_models=False)
        except TclError as exc:
            self.skipTest(f"当前环境不可创建 tkinter 窗口：{exc}")
            return
        app.withdraw()
        app.update_idletasks()
        app.destroy()

    def test_top_row_label_should_stay_inside_plot_area(self) -> None:
        try:
            app = FertilizerApp(auto_load_models=False)
        except TclError as exc:
            self.skipTest(f"当前环境不可创建 tkinter 窗口：{exc}")
            return
        app.withdraw()
        width, height = 800, 600
        _, top, right, bottom = app._plot_area(width, height)
        display_x, display_y = app._display_row_point(right - 2, top - 20, 6, width, height)
        self.assertGreaterEqual(display_y, top + 8)
        self.assertLessEqual(display_y, bottom - 8)
        self._assert_label_within_plot(
            app,
            row_x=right - 2,
            row_y=top - 20,
            row_index=6,
            label_text="R6 | 35.0mm\n412.3r/min",
            width=width,
            height=height,
            expected_anchor="e",
        )
        app.destroy()

    def test_bottom_row_label_should_stay_inside_plot_area(self) -> None:
        try:
            app = FertilizerApp(auto_load_models=False)
        except TclError as exc:
            self.skipTest(f"当前环境不可创建 tkinter 窗口：{exc}")
            return
        app.withdraw()
        width, height = 800, 600
        left, top, _, bottom = app._plot_area(width, height)
        display_x, display_y = app._display_row_point(left + 2, bottom + 20, 1, width, height)
        self.assertGreaterEqual(display_y, top + 8)
        self.assertLessEqual(display_y, bottom - 8)
        self._assert_label_within_plot(
            app,
            row_x=left + 2,
            row_y=bottom + 20,
            row_index=1,
            label_text="R1 | 20.0mm\n128.4r/min",
            width=width,
            height=height,
            expected_anchor="w",
        )
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
