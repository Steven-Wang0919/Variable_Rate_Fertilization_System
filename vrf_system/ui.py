from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .annotations import format_row_annotation
from .controller import SimulationController
from .defaults import (
    DEFAULT_KAN_ARTIFACT_DIR,
    DEFAULT_MLP_ARTIFACT_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLE_PRESCRIPTION,
)
from .domain import Bounds, MachineConfig


KAN_COLOR = "#059669"
MLP_COLOR = "#d97706"
OUT_COLOR = "#6b7280"
PATH_COLOR = "#2563eb"
CENTER_COLOR = "#111827"


class FertilizerApp(tk.Tk):
    def __init__(self, auto_load_models: bool = True) -> None:
        super().__init__()
        self.title("玉米精量播种机变量施肥决策系统")
        self.geometry("1600x900")
        self.minsize(1360, 780)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.row_annotation_font = ("Microsoft YaHei UI", 8)
        self.pass_label_font = ("Microsoft YaHei UI", 8, "bold")

        self.controller = SimulationController()
        self.current_frame_index = 0
        self.last_export_dir: Path | None = None

        self.kan_dir_var = tk.StringVar(value=str(DEFAULT_KAN_ARTIFACT_DIR))
        self.mlp_dir_var = tk.StringVar(value=str(DEFAULT_MLP_ARTIFACT_DIR))
        self.prescription_path_var = tk.StringVar(value=str(DEFAULT_SAMPLE_PRESCRIPTION))
        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.row_count_var = tk.StringVar(value="6")
        self.row_spacing_var = tk.StringVar(value="0.6")
        self.travel_speed_var = tk.StringVar(value="6.0")
        self.sample_period_var = tk.StringVar(value="200")
        self.longitudinal_offset_var = tk.StringVar(value="0.0")
        self.row_offsets_var = tk.StringVar(value="")
        self.frame_info_var = tk.StringVar(value="当前还没有仿真结果。")
        self.summary_var = tk.StringVar(value="请先加载模型并导入处方图。")

        self._build_layout()
        self.canvas.bind("<Configure>", lambda _event: self._redraw_map())
        self.legend_canvas.bind("<Configure>", lambda _event: self._redraw_legend())

        if auto_load_models:
            self._load_default_models(show_message=False)

    def _build_layout(self) -> None:
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main_pane, padding=12)
        center = ttk.Frame(main_pane, padding=12)
        right = ttk.Frame(main_pane, padding=12)
        main_pane.add(left, weight=28)
        main_pane.add(center, weight=42)
        main_pane.add(right, weight=30)

        self._build_left_panel(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        model_box = ttk.LabelFrame(parent, text="模型包管理", padding=10)
        model_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(model_box, text="KAN 模型目录").pack(anchor=tk.W)
        ttk.Entry(model_box, textvariable=self.kan_dir_var).pack(fill=tk.X, pady=(2, 6))
        ttk.Button(model_box, text="选择 KAN 目录", command=self._choose_kan_dir).pack(fill=tk.X)
        ttk.Label(model_box, text="MLP 模型目录").pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(model_box, textvariable=self.mlp_dir_var).pack(fill=tk.X, pady=(2, 6))
        ttk.Button(model_box, text="选择 MLP 目录", command=self._choose_mlp_dir).pack(fill=tk.X)
        ttk.Button(model_box, text="按当前目录加载模型", command=self._load_models_from_current_inputs).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(model_box, text="恢复默认模型目录", command=self._load_default_models).pack(fill=tk.X, pady=(6, 0))

        prescription_box = ttk.LabelFrame(parent, text="处方图", padding=10)
        prescription_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(prescription_box, text="处方图 CSV 路径").pack(anchor=tk.W)
        ttk.Entry(prescription_box, textvariable=self.prescription_path_var).pack(fill=tk.X, pady=(2, 6))
        ttk.Button(prescription_box, text="导入处方图 CSV", command=self._choose_prescription_csv).pack(fill=tk.X)
        ttk.Button(prescription_box, text="加载示例处方图", command=self._load_sample_prescription).pack(fill=tk.X, pady=(6, 0))

        machine_box = ttk.LabelFrame(parent, text="机器参数", padding=10)
        machine_box.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(machine_box, "行数", self.row_count_var)
        self._add_labeled_entry(machine_box, "行距 (m)", self.row_spacing_var)
        self._add_labeled_entry(machine_box, "作业速度 (km/h)", self.travel_speed_var)
        self._add_labeled_entry(machine_box, "采样周期 (ms)", self.sample_period_var)
        self._add_labeled_entry(machine_box, "纵向偏移 (m)", self.longitudinal_offset_var)
        self._add_labeled_entry(
            machine_box,
            "排位偏移列表",
            self.row_offsets_var,
            note="留空则自动按行距居中生成，例如：-1.5,-0.9,-0.3,0.3,0.9,1.5",
        )
        ttk.Button(machine_box, text="恢复默认机器参数", command=self._reset_machine_defaults).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(machine_box, text="运行仿真", command=self._run_simulation).pack(fill=tk.X, pady=(6, 0))

        export_box = ttk.LabelFrame(parent, text="导出设置", padding=10)
        export_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(export_box, text="导出根目录").pack(anchor=tk.W)
        ttk.Entry(export_box, textvariable=self.output_root_var).pack(fill=tk.X, pady=(2, 6))
        ttk.Button(export_box, text="选择导出目录", command=self._choose_output_root).pack(fill=tk.X)

        log_box = ttk.LabelFrame(parent, text="运行日志", padding=10)
        log_box.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(log_box, height=18, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="处方热力图与机具轨迹", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="地图区已拆分为绘图区和独立图例区，用于避免轨迹、排号和图例互相干涉。",
            foreground="#64748b",
        ).pack(anchor=tk.W, pady=(2, 8))

        map_frame = ttk.Frame(parent)
        map_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(map_frame, background="#f8fafc", highlightthickness=1, highlightbackground="#d1d5db")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.legend_canvas = tk.Canvas(
            map_frame,
            width=220,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d1d5db",
        )
        self.legend_canvas.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        slider_frame = ttk.Frame(parent)
        slider_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(slider_frame, text="时间步").pack(anchor=tk.W)
        self.frame_slider = ttk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_slider_changed,
        )
        self.frame_slider.pack(fill=tk.X, pady=(4, 6))
        ttk.Label(slider_frame, textvariable=self.frame_info_var, wraplength=760).pack(anchor=tk.W)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        summary_box = ttk.LabelFrame(parent, text="仿真摘要", padding=10)
        summary_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(summary_box, textvariable=self.summary_var, wraplength=420, justify=tk.LEFT).pack(anchor=tk.W)

        actions = ttk.Frame(summary_box)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="导出结果与地图", command=self._export_results).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="打开导出目录", command=self._open_last_export_dir).pack(side=tk.LEFT)

        table_box = ttk.LabelFrame(parent, text="当前时刻单排决策", padding=10)
        table_box.pack(fill=tk.BOTH, expand=True)
        columns = ("row", "zone", "rate", "mass", "opening", "speed", "model", "status")
        self.decision_table = ttk.Treeview(table_box, columns=columns, show="headings", height=20)
        headings = {
            "row": "排号",
            "zone": "分区",
            "rate": "目标量(kg/ha)",
            "mass": "目标排肥量(g/min)",
            "opening": "开度(mm)",
            "speed": "目标转速(r/min)",
            "model": "模型",
            "status": "状态",
        }
        widths = {"row": 56, "zone": 72, "rate": 110, "mass": 126, "opening": 92, "speed": 112, "model": 108, "status": 110}
        for column in columns:
            self.decision_table.heading(column, text=headings[column])
            self.decision_table.column(column, width=widths[column], anchor=tk.CENTER)
        y_scroll = ttk.Scrollbar(table_box, orient=tk.VERTICAL, command=self.decision_table.yview)
        self.decision_table.configure(yscrollcommand=y_scroll.set)
        self.decision_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, note: str | None = None) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if note:
            ttk.Label(parent, text=note, foreground="#64748b", wraplength=280).pack(anchor=tk.W, padx=(2, 0), pady=(0, 2))

    def _choose_kan_dir(self) -> None:
        path = filedialog.askdirectory(title="选择 inverse_KAN 模型目录", initialdir=self.kan_dir_var.get() or str(Path.cwd()))
        if path:
            self.kan_dir_var.set(path)

    def _choose_mlp_dir(self) -> None:
        path = filedialog.askdirectory(title="选择 inverse_MLP 模型目录", initialdir=self.mlp_dir_var.get() or str(Path.cwd()))
        if path:
            self.mlp_dir_var.set(path)

    def _choose_prescription_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="选择处方图 CSV",
            initialdir=str(DEFAULT_SAMPLE_PRESCRIPTION.parent),
            filetypes=[("CSV 文件", "*.csv")],
        )
        if path:
            self.prescription_path_var.set(path)
            self._load_prescription(path)

    def _choose_output_root(self) -> None:
        path = filedialog.askdirectory(title="选择导出根目录", initialdir=self.output_root_var.get() or str(Path.cwd()))
        if path:
            self.output_root_var.set(path)

    def _load_default_models(self, show_message: bool = True) -> None:
        self.kan_dir_var.set(str(DEFAULT_KAN_ARTIFACT_DIR))
        self.mlp_dir_var.set(str(DEFAULT_MLP_ARTIFACT_DIR))
        self._load_models_from_current_inputs(show_message=show_message)

    def _load_models_from_current_inputs(self, show_message: bool = True) -> None:
        try:
            kan_bundle, mlp_bundle = self.controller.load_models_from_dirs(
                self.kan_dir_var.get(),
                self.mlp_dir_var.get(),
            )
            self._log(f"已加载模型：{kan_bundle.config.name} 与 {mlp_bundle.config.name}")
            if show_message:
                messagebox.showinfo("模型加载完成", "已成功加载 KAN 与 MLP 模型包。")
        except Exception as exc:  # noqa: BLE001
            self._log(f"模型加载失败：{exc}")
            if show_message:
                messagebox.showerror("模型加载失败", str(exc))

    def _load_sample_prescription(self) -> None:
        self.prescription_path_var.set(str(DEFAULT_SAMPLE_PRESCRIPTION))
        self._load_prescription(DEFAULT_SAMPLE_PRESCRIPTION)

    def _load_prescription(self, path: str | Path) -> None:
        try:
            prescription = self.controller.load_prescription(path)
            self._log(f"已加载处方图：{prescription.source_path}")
            self.summary_var.set("处方图已导入，可以直接运行仿真。")
            self._redraw_map()
            self._redraw_legend()
        except Exception as exc:  # noqa: BLE001
            self._log(f"处方图导入失败：{exc}")
            messagebox.showerror("处方图导入失败", str(exc))

    def _parse_machine_config(self) -> MachineConfig:
        row_offsets_text = self.row_offsets_var.get().strip()
        row_offsets = []
        if row_offsets_text:
            row_offsets = [float(item.strip()) for item in row_offsets_text.split(",") if item.strip()]
        return MachineConfig(
            row_count=int(self.row_count_var.get()),
            row_spacing_m=float(self.row_spacing_var.get()),
            travel_speed_kmh=float(self.travel_speed_var.get()),
            sample_period_ms=int(self.sample_period_var.get()),
            machine_center_to_row_origin_m=float(self.longitudinal_offset_var.get()),
            row_offsets_m=row_offsets,
        )

    def _reset_machine_defaults(self) -> None:
        self.row_count_var.set("6")
        self.row_spacing_var.set("0.6")
        self.travel_speed_var.set("6.0")
        self.sample_period_var.set("200")
        self.longitudinal_offset_var.set("0.0")
        self.row_offsets_var.set("")
        self._log("已恢复默认机器参数。")

    def _run_simulation(self) -> None:
        try:
            if self.controller.router is None:
                self._load_models_from_current_inputs(show_message=False)
            if self.controller.prescription_map is None and self.prescription_path_var.get():
                self._load_prescription(self.prescription_path_var.get())
            result = self.controller.run_simulation(self._parse_machine_config())
            self.current_frame_index = 0
            self.frame_slider.configure(to=max(len(result.frames) - 1, 0))
            self.frame_slider.set(0)
            self._refresh_summary()
            self._refresh_current_frame()
            self._log("仿真完成。")
        except Exception as exc:  # noqa: BLE001
            self._log(f"仿真失败：{exc}")
            messagebox.showerror("仿真失败", str(exc))

    def _refresh_summary(self) -> None:
        result = self.controller.last_result
        if result is None:
            self.summary_var.set("当前没有仿真结果。")
            return
        summary = result.summary
        model_counts = summary.get("selected_model_counts", {})
        status_counts = summary.get("status_counts", {})
        domain_counts = summary.get("domain_status_counts", {})
        self.summary_var.set(
            f"总时间步：{summary.get('frame_count', 0)}\n"
            f"总作业往返：{summary.get('pass_count', 0)}\n"
            f"总单排决策数：{summary.get('total_row_decisions', 0)}\n"
            f"外推次数：{summary.get('extrapolation_count', 0)}\n"
            f"KAN 调用：{model_counts.get('inverse_KAN', 0)}\n"
            f"MLP 调用：{model_counts.get('inverse_MLP', 0)}\n"
            f"地块外点数：{status_counts.get('out_of_field', 0)}\n"
            f"域内点数：{domain_counts.get('in_domain', 0)}\n"
            f"平均目标施肥量：{summary.get('average_target_rate_kg_ha', 0)} kg/ha\n"
            f"平均目标转速：{summary.get('average_target_speed_r_min', 0)} r/min"
        )

    def _refresh_current_frame(self) -> None:
        result = self.controller.last_result
        if result is None or not result.frames:
            self.frame_info_var.set("当前还没有仿真结果。")
            return
        frame = result.frames[self.current_frame_index]
        self.frame_info_var.set(
            f"时间戳：{frame.timestamp_ms} ms，作业趟次：第 {frame.pass_id} 趟，"
            f"机具中心：({frame.machine_center_x_m:.2f}, {frame.machine_center_y_m:.2f}) m"
        )
        for item in self.decision_table.get_children():
            self.decision_table.delete(item)
        for decision in frame.row_decisions:
            self.decision_table.insert(
                "",
                tk.END,
                values=(
                    decision.row_index,
                    decision.zone_id or "-",
                    f"{decision.target_rate_kg_ha:.1f}",
                    f"{decision.target_mass_g_min:.1f}",
                    f"{decision.strategy_opening_mm:.1f}",
                    f"{decision.target_speed_r_min:.2f}",
                    decision.selected_model,
                    decision.status,
                ),
            )
        self._redraw_map()
        self._redraw_legend()

    def _on_slider_changed(self, _value: str) -> None:
        result = self.controller.last_result
        if result is None or not result.frames:
            return
        index = int(round(float(self.frame_slider.get())))
        index = max(0, min(index, len(result.frames) - 1))
        self.current_frame_index = index
        self._refresh_current_frame()

    def _map_layout(self, width: int, height: int) -> dict[str, int]:
        return {"left": 70, "right": 30, "top": 26, "bottom": 68}

    def _plot_area(self, width: int, height: int) -> tuple[int, int, int, int]:
        layout = self._map_layout(width, height)
        return (
            layout["left"],
            layout["top"],
            width - layout["right"],
            height - layout["bottom"],
        )

    def _rate_to_color(self, rate: float, min_rate: float, max_rate: float) -> str:
        if max_rate <= min_rate:
            normalized = 0.5
        else:
            normalized = (rate - min_rate) / (max_rate - min_rate)
        normalized = max(0.0, min(normalized, 1.0))
        red = int(24 + normalized * 210)
        green = int(130 + (1.0 - normalized) * 80)
        blue = int(190 - normalized * 120)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _transform_point(self, x: float, y: float, width: int, height: int) -> tuple[float, float]:
        prescription = self.controller.prescription_map
        if prescription is None:
            return x, y
        bounds = prescription.bounds
        left, top, right, bottom = self._plot_area(width, height)
        usable_width = max(right - left, 1)
        usable_height = max(bottom - top, 1)
        scale_x = usable_width / max(bounds.width, 1e-6)
        scale_y = usable_height / max(bounds.height, 1e-6)
        scale = min(scale_x, scale_y)
        canvas_x = left + (x - bounds.min_x) * scale
        canvas_y = bottom - (y - bounds.min_y) * scale
        return canvas_x, canvas_y

    def _draw_axes(self, width: int, height: int, bounds: Bounds) -> None:
        left, top, right, bottom = self._plot_area(width, height)
        self.canvas.create_rectangle(left, top, right, bottom, outline="#cbd5e1", width=1)
        tick_count = 5
        for index in range(tick_count + 1):
            ratio = index / tick_count
            x = left + (right - left) * ratio
            y = bottom - (bottom - top) * ratio
            self.canvas.create_line(x, bottom, x, bottom + 6, fill="#64748b")
            self.canvas.create_line(left - 6, y, left, y, fill="#64748b")
            x_value = bounds.min_x + bounds.width * ratio
            y_value = bounds.min_y + bounds.height * ratio
            self.canvas.create_text(x, bottom + 18, text=f"{x_value:.1f}", fill="#334155", font=("Microsoft YaHei UI", 9))
            self.canvas.create_text(left - 28, y, text=f"{y_value:.1f}", fill="#334155", font=("Microsoft YaHei UI", 9))

        self.canvas.create_text((left + right) / 2, height - 20, text="X 坐标 / m", fill="#0f172a", font=("Microsoft YaHei UI", 10, "bold"))
        self.canvas.create_text(22, (top + bottom) / 2, text="Y\n坐\n标\n/\nm", fill="#0f172a", font=("Microsoft YaHei UI", 10, "bold"))

    def _measure_text_block(self, text: str, font: tuple) -> tuple[int, int]:
        font_obj = tkfont.Font(font=font)
        lines = text.splitlines() or [text]
        width = max(font_obj.measure(line) for line in lines)
        height = font_obj.metrics("linespace") * len(lines)
        return width, height

    def _draw_text_with_bg(
        self,
        x: float,
        y: float,
        text: str,
        *,
        anchor: str,
        fill: str,
        font: tuple,
        justify: str | None = None,
    ) -> None:
        options = {"text": text, "anchor": anchor, "fill": fill, "font": font}
        if justify is not None:
            options["justify"] = justify
        text_id = self.canvas.create_text(x, y, **options)
        bbox = self.canvas.bbox(text_id)
        if bbox:
            rect_id = self.canvas.create_rectangle(
                bbox[0] - 2,
                bbox[1] - 1,
                bbox[2] + 2,
                bbox[3] + 1,
                fill="#f8fafc",
                outline="",
            )
            self.canvas.tag_lower(rect_id, text_id)

    def _draw_cell_label(self, cell, left: float, top: float, right: float, bottom: float) -> None:
        if (right - left) < 54 or (bottom - top) < 30:
            return
        self.canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text=f"{cell.zone_id}\n{cell.target_rate_kg_ha:.0f}",
            fill="#0f172a",
            font=("Microsoft YaHei UI", 8),
        )

    def _display_row_point(
        self,
        row_x: float,
        row_y: float,
        row_index: int,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        left, top, right, bottom = self._plot_area(width, height)
        display_x = max(left + 6, min(right - 6, row_x))
        display_y = row_y

        if row_y < top + 8:
            display_y = top + 10 + ((row_index - 1) % 3) * 14
        elif row_y > bottom - 8:
            display_y = bottom - 10 - ((row_index - 1) % 3) * 14

        return display_x, display_y

    def _row_label_position(
        self,
        row_x: float,
        row_y: float,
        row_index: int,
        label_text: str,
        width: int,
        height: int,
    ) -> tuple[float, float, str]:
        left, top, right, bottom = self._plot_area(width, height)
        display_x, display_y = self._display_row_point(row_x, row_y, row_index, width, height)
        label_width, label_height = self._measure_text_block(label_text, self.row_annotation_font)
        vertical_gap = 6
        half_height = label_height / 2.0
        horizontal_gap = 10
        inner_padding = 6

        if display_x <= left + label_width + horizontal_gap:
            anchor = "w"
            label_x = display_x + horizontal_gap
        elif display_x >= right - label_width - horizontal_gap:
            anchor = "e"
            label_x = display_x - horizontal_gap
        elif row_index % 2 == 0:
            anchor = "w"
            label_x = display_x + horizontal_gap
        else:
            anchor = "e"
            label_x = display_x - horizontal_gap

        if anchor == "w":
            label_x = min(label_x, right - label_width - inner_padding)
            label_x = max(label_x, left + inner_padding)
        else:
            label_x = max(label_x, left + label_width + inner_padding)
            label_x = min(label_x, right - inner_padding)

        if display_y <= top + half_height + vertical_gap:
            label_y = display_y + half_height + vertical_gap
        elif display_y >= bottom - half_height - vertical_gap:
            label_y = display_y - half_height - vertical_gap
        else:
            label_y = display_y - half_height - vertical_gap

        label_y = max(top + half_height + inner_padding, min(bottom - half_height - inner_padding, label_y))
        return label_x, label_y, anchor

    def _redraw_map(self) -> None:
        self.canvas.delete("all")
        prescription = self.controller.prescription_map
        width = max(self.canvas.winfo_width(), 720)
        height = max(self.canvas.winfo_height(), 560)

        if prescription is None:
            self.canvas.create_text(width / 2, height / 2, text="请先导入处方图。", fill="#475569", font=("Microsoft YaHei UI", 12))
            return

        min_rate, max_rate = prescription.rate_range()
        for cell in prescription.cells:
            left, top = self._transform_point(cell.left, cell.top, width, height)
            right, bottom = self._transform_point(cell.right, cell.bottom, width, height)
            fill = self._rate_to_color(cell.target_rate_kg_ha, min_rate, max_rate)
            self.canvas.create_rectangle(left, top, right, bottom, fill=fill, outline="#ffffff")
            self._draw_cell_label(cell, left, top, right, bottom)

        self._draw_axes(width, height, prescription.bounds)

        result = self.controller.last_result
        if result is None or not result.frames:
            return

        left, top, right, bottom = self._plot_area(width, height)
        pass_points: dict[int, list[tuple[float, float]]] = {}
        for frame in result.frames:
            pass_points.setdefault(frame.pass_id, []).append(
                self._transform_point(frame.machine_center_x_m, frame.machine_center_y_m, width, height)
            )
        for pass_id, points in pass_points.items():
            if len(points) > 1:
                flattened = [value for point in points for value in point]
                self.canvas.create_line(*flattened, fill=PATH_COLOR, width=2, smooth=True, stipple="gray50")
                max_x_point = max(points, key=lambda point: point[0])
                label_y = max(top + 10, min(bottom - 10, max_x_point[1] - 10))
                self._draw_text_with_bg(right - 6, label_y, f"P{pass_id}", anchor="e", fill=PATH_COLOR, font=self.pass_label_font)

        frame = result.frames[self.current_frame_index]
        center_x, center_y = self._transform_point(frame.machine_center_x_m, frame.machine_center_y_m, width, height)
        ordered_rows = sorted(frame.row_decisions, key=lambda item: item.row_index)
        valid_rows = [item for item in ordered_rows if item.status != "out_of_field"]
        if len(valid_rows) >= 2:
            line_points = [self._transform_point(item.x_m, item.y_m, width, height) for item in valid_rows]
            flattened = [value for point in line_points for value in point]
            self.canvas.create_line(*flattened, fill=CENTER_COLOR, width=3)

        self.canvas.create_oval(center_x - 6, center_y - 6, center_x + 6, center_y + 6, fill=CENTER_COLOR, outline="")
        arrow_length = 26
        arrow_x = max(left + 6, min(right - 6, center_x + arrow_length * frame.direction_sign))
        self.canvas.create_line(center_x, center_y, arrow_x, center_y, fill=CENTER_COLOR, width=2, arrow=tk.LAST)

        for decision in ordered_rows:
            row_x, row_y = self._transform_point(decision.x_m, decision.y_m, width, height)
            display_x, display_y = self._display_row_point(row_x, row_y, decision.row_index, width, height)
            if decision.status == "out_of_field":
                self.canvas.create_line(display_x - 5, display_y - 5, display_x + 5, display_y + 5, fill=OUT_COLOR, width=2)
                self.canvas.create_line(display_x + 5, display_y - 5, display_x - 5, display_y + 5, fill=OUT_COLOR, width=2)
                label_color = OUT_COLOR
            else:
                if decision.selected_model == "inverse_KAN":
                    self.canvas.create_oval(display_x - 5, display_y - 5, display_x + 5, display_y + 5, fill=KAN_COLOR, outline="")
                    label_color = KAN_COLOR
                else:
                    self.canvas.create_rectangle(display_x - 5, display_y - 5, display_x + 5, display_y + 5, fill=MLP_COLOR, outline="")
                    label_color = MLP_COLOR

            label_text = format_row_annotation(decision)
            label_x, label_y, anchor = self._row_label_position(
                row_x,
                row_y,
                decision.row_index,
                label_text,
                width,
                height,
            )
            justify = tk.RIGHT if anchor == "e" else tk.LEFT
            self._draw_text_with_bg(
                label_x,
                label_y,
                label_text,
                anchor=anchor,
                fill=label_color,
                font=self.row_annotation_font,
                justify=justify,
            )

    def _redraw_legend(self) -> None:
        self.legend_canvas.delete("all")
        width = max(self.legend_canvas.winfo_width(), 180)
        height = max(self.legend_canvas.winfo_height(), 560)
        self.legend_canvas.create_text(16, 16, anchor=tk.NW, text="地图图例", fill="#0f172a", font=("Microsoft YaHei UI", 11, "bold"))

        prescription = self.controller.prescription_map
        if prescription is None:
            self.legend_canvas.create_text(width / 2, height / 2, text="请先导入处方图。", fill="#64748b")
            return

        min_rate, max_rate = prescription.rate_range()
        self.legend_canvas.create_text(16, 42, anchor=tk.NW, text="目标施肥量", fill="#334155", font=("Microsoft YaHei UI", 9))
        self.legend_canvas.create_text(16, 58, anchor=tk.NW, text="kg/ha", fill="#64748b", font=("Microsoft YaHei UI", 9))

        bar_left = 40
        bar_top = 88
        bar_right = 62
        bar_bottom = min(height - 210, 390)
        steps = 48
        height_step = (bar_bottom - bar_top) / steps
        for index in range(steps):
            ratio = index / max(steps - 1, 1)
            rate = max_rate - (max_rate - min_rate) * ratio
            y0 = bar_top + index * height_step
            y1 = y0 + height_step + 1
            color = self._rate_to_color(rate, min_rate, max_rate)
            self.legend_canvas.create_rectangle(bar_left, y0, bar_right, y1, outline="", fill=color)
        self.legend_canvas.create_rectangle(bar_left, bar_top, bar_right, bar_bottom, outline="#94a3b8")
        self.legend_canvas.create_text(bar_right + 18, bar_top, anchor=tk.W, text=f"{max_rate:.0f}", fill="#0f172a", font=("Microsoft YaHei UI", 9))
        self.legend_canvas.create_text(bar_right + 18, (bar_top + bar_bottom) / 2, anchor=tk.W, text=f"{(min_rate + max_rate) / 2:.0f}", fill="#334155", font=("Microsoft YaHei UI", 9))
        self.legend_canvas.create_text(bar_right + 18, bar_bottom, anchor=tk.W, text=f"{min_rate:.0f}", fill="#0f172a", font=("Microsoft YaHei UI", 9))

        legend_top = bar_bottom + 28
        self.legend_canvas.create_text(16, legend_top, anchor=tk.NW, text="标记说明", fill="#0f172a", font=("Microsoft YaHei UI", 10, "bold"))
        items = [
            ("oval", KAN_COLOR, "KAN 域内决策"),
            ("rect", MLP_COLOR, "MLP 外推决策"),
            ("cross", OUT_COLOR, "地块外排点"),
            ("line", CENTER_COLOR, "机具中心 / 跨排连线"),
            ("path", PATH_COLOR, "历史轨迹"),
        ]
        y_cursor = legend_top + 26
        for shape, color, text in items:
            if shape == "oval":
                self.legend_canvas.create_oval(18, y_cursor, 28, y_cursor + 10, fill=color, outline="")
            elif shape == "rect":
                self.legend_canvas.create_rectangle(18, y_cursor, 28, y_cursor + 10, fill=color, outline="")
            elif shape == "cross":
                self.legend_canvas.create_line(18, y_cursor, 28, y_cursor + 10, fill=color, width=2)
                self.legend_canvas.create_line(28, y_cursor, 18, y_cursor + 10, fill=color, width=2)
            elif shape == "line":
                self.legend_canvas.create_line(18, y_cursor + 5, 34, y_cursor + 5, fill=color, width=2)
                self.legend_canvas.create_oval(24, y_cursor + 1, 28, y_cursor + 9, fill=color, outline="")
            else:
                self.legend_canvas.create_line(18, y_cursor + 5, 34, y_cursor + 5, fill=color, width=2, stipple="gray50")
            self.legend_canvas.create_text(44, y_cursor + 5, anchor=tk.W, text=text, fill="#334155", font=("Microsoft YaHei UI", 9))
            y_cursor += 24

        result = self.controller.last_result
        if result is not None:
            summary = result.summary
            extra_top = y_cursor + 12
            self.legend_canvas.create_text(16, extra_top, anchor=tk.NW, text="当前统计", fill="#0f172a", font=("Microsoft YaHei UI", 10, "bold"))
            self.legend_canvas.create_text(
                16,
                extra_top + 22,
                anchor=tk.NW,
                text=(
                    f"外推次数：{summary.get('extrapolation_count', 0)}\n"
                    f"总单排决策：{summary.get('total_row_decisions', 0)}"
                ),
                fill="#334155",
                font=("Microsoft YaHei UI", 9),
            )

    def _export_results(self) -> None:
        try:
            artifacts = self.controller.export_last_result(
                self.output_root_var.get(),
                highlighted_frame_index=self.current_frame_index,
            )
            self.last_export_dir = artifacts.output_dir
            self._log(f"导出完成：{artifacts.output_dir}")
            messagebox.showinfo(
                "导出完成",
                "已生成以下文件：\n"
                f"{artifacts.row_command_timeline}\n"
                f"{artifacts.model_routing_trace}\n"
                f"{artifacts.simulation_summary}\n"
                f"{artifacts.map_overview_png}\n"
                f"{artifacts.map_current_frame_png}\n"
                f"{artifacts.map_legend_png}",
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"导出失败：{exc}")
            messagebox.showerror("导出失败", str(exc))

    def _open_last_export_dir(self) -> None:
        if self.last_export_dir is None:
            messagebox.showwarning("未导出", "请先导出一次结果。")
            return
        if self.last_export_dir.exists():
            os.startfile(self.last_export_dir)  # type: ignore[attr-defined]

    def _log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def launch_app() -> None:
    app = FertilizerApp(auto_load_models=True)
    app.mainloop()
