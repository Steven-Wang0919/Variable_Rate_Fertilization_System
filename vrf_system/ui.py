from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import sv_ttk
except ImportError:  # pragma: no cover - optional dependency fallback
    sv_ttk = None

from .controller import SimulationController
from .defaults import (
    DEFAULT_FORWARD_KAN_ARTIFACT_DIR,
    DEFAULT_KAN_ARTIFACT_DIR,
    DEFAULT_MLP_ARTIFACT_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLE_PRESCRIPTION,
)
from .domain import ForwardPredictionResult, MachineConfig
from .visualization import (
    CurrentFramePreviewState,
    OverviewPreviewState,
    create_current_preview_state,
    create_legend_figure,
    create_overview_preview_state,
    create_prescription_legend_figure,
    create_prescription_overview_figure,
)


APP_BG = "#eef3f9"
SURFACE_BG = "#ffffff"
BORDER_COLOR = "#d6dee8"
TEXT_PRIMARY = "#0f172a"
TEXT_MUTED = "#64748b"
TEXT_SOFT = "#94a3b8"
PRIMARY_COLOR = "#0f766e"
PRIMARY_COLOR_ACTIVE = "#115e59"
ACCENT_COLOR = "#0ea5a4"
SLIDER_TROUGH_ACTIVE = "#99f6e4"
SLIDER_TROUGH_DISABLED = "#dbe4ee"
SLIDER_HANDLE_ACTIVE = "#0f766e"
SLIDER_HANDLE_DISABLED = "#94a3b8"
LEFT_PANEL_WEIGHT = 30
CENTER_PANEL_WEIGHT = 50
RIGHT_PANEL_WEIGHT = 20
MIN_SIDE_PANEL_WIDTH = 260
MIN_CENTER_PANEL_WIDTH = 640


class FertilizerApp(tk.Tk):
    PREVIEW_TITLES = {
        "overview": "总览图",
        "current": "当前帧细节图",
        "legend": "图例",
    }
    INTERACTIVE_PREVIEW_KEYS = ("overview", "current")
    LIVE_PREVIEW_INTERVAL_MS = 16

    def __init__(self, auto_load_models: bool = True) -> None:
        super().__init__()
        self.title("玉米精量播种机变量施肥决策系统")
        self.geometry("1800x1040")
        self.minsize(1480, 860)
        self.configure(bg=APP_BG)
        self.option_add("*Font", ("Microsoft YaHei UI", 10))

        self.controller = SimulationController()
        self.current_frame_index = 0
        self.last_export_dir: Path | None = None

        self.kan_dir_var = tk.StringVar(value=str(DEFAULT_KAN_ARTIFACT_DIR))
        self.mlp_dir_var = tk.StringVar(value=str(DEFAULT_MLP_ARTIFACT_DIR))
        self.forward_kan_dir_var = tk.StringVar(value=str(DEFAULT_FORWARD_KAN_ARTIFACT_DIR))
        self.prescription_path_var = tk.StringVar(value=str(DEFAULT_SAMPLE_PRESCRIPTION))
        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.row_count_var = tk.StringVar(value="6")
        self.row_spacing_var = tk.StringVar(value="0.6")
        self.travel_speed_var = tk.StringVar(value="6.0")
        self.sample_period_var = tk.StringVar(value="200")
        self.longitudinal_offset_var = tk.StringVar(value="0.0")
        self.row_offsets_var = tk.StringVar(value="")
        self.forward_opening_var = tk.StringVar(value="")
        self.forward_speed_var = tk.StringVar(value="")
        self.frame_info_var = tk.StringVar(value="当前还没有仿真结果。")
        self.summary_var = tk.StringVar(value="请先加载模型并导入处方图。")
        self.forward_prediction_var = tk.StringVar(value="请输入开度与转速后执行预测。")

        self.left_toggle_text = tk.StringVar(value="收起左栏")
        self.right_toggle_text = tk.StringVar(value="收起右栏")

        self.main_pane: ttk.PanedWindow | None = None
        self.left_panel: ttk.Frame | None = None
        self.center_panel: ttk.Frame | None = None
        self.right_panel: ttk.Frame | None = None
        self.left_form_canvas: tk.Canvas | None = None
        self.left_form_scrollbar: ttk.Scrollbar | None = None
        self.left_form_content: ttk.Frame | None = None
        self.left_form_window_id: int | None = None
        self.left_log_box: ttk.LabelFrame | None = None
        self.left_toggle_button: ttk.Button | None = None
        self.right_toggle_button: ttk.Button | None = None
        self.forward_predict_button: ttk.Button | None = None
        self.left_collapsed = False
        self.right_collapsed = False
        self.saved_left_width: int | None = None
        self.saved_right_width: int | None = None

        self.preview_frames: dict[str, ttk.Frame] = {}
        self.preview_hosts: dict[str, ttk.Frame] = {}
        self.preview_placeholders: dict[str, tk.Label] = {}
        self.preview_status_vars: dict[str, tk.StringVar] = {}
        self.preview_canvases: dict[str, FigureCanvasTkAgg] = {}
        self.preview_figures: dict[str, object] = {}
        self.preview_renderers: dict[str, OverviewPreviewState | CurrentFramePreviewState] = {}
        self.preview_tab_keys: dict[str, str] = {}
        self.preview_tab_ids: dict[str, str] = {}
        self.preview_dirty_keys: set[str] = set(self.PREVIEW_TITLES)

        self._preview_result_token: int | None = None
        self._slider_events_suspended = False
        self._live_preview_after_id: str | None = None
        self._pending_live_frame_index: int | None = None
        self._live_preview_key: str | None = None
        self._table_dirty = False
        self._table_frame_index: int | None = None

        self._configure_styles()
        self._build_layout()
        self._set_result_state(has_result=False)
        self._render_preview_tabs(force=True)

        if auto_load_models:
            self._load_default_models(show_message=False)

    def _configure_styles(self) -> None:
        if sv_ttk is not None:
            try:
                sv_ttk.set_theme("light")
            except Exception:
                pass

        style = ttk.Style(self)
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("TLabelframe", background=SURFACE_BG, borderwidth=1)
        style.configure("TLabelframe.Label", background=SURFACE_BG, foreground=TEXT_PRIMARY, font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=SURFACE_BG)
        style.configure("Section.TLabel", background=APP_BG, foreground=TEXT_PRIMARY, font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Hint.TLabel", background=APP_BG, foreground=TEXT_MUTED)
        style.configure("Note.TLabel", background=SURFACE_BG, foreground=TEXT_SOFT)
        style.configure("Primary.TButton", padding=(10, 8))
        style.configure("Secondary.TButton", padding=(10, 8))
        style.configure("Treeview", rowheight=28)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_layout(self) -> None:
        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.left_panel = ttk.Frame(self.main_pane, style="Card.TFrame", padding=14)
        self.center_panel = ttk.Frame(self.main_pane, style="Card.TFrame", padding=14)
        self.right_panel = ttk.Frame(self.main_pane, style="Card.TFrame", padding=14)
        self.main_pane.add(self.left_panel, weight=LEFT_PANEL_WEIGHT)
        self.main_pane.add(self.center_panel, weight=CENTER_PANEL_WEIGHT)
        self.main_pane.add(self.right_panel, weight=RIGHT_PANEL_WEIGHT)

        self._build_left_panel(self.left_panel)
        self._build_center_panel(self.center_panel)
        self._build_right_panel(self.right_panel)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        scroll_box = ttk.Frame(parent, style="Card.TFrame")
        scroll_box.grid(row=0, column=0, sticky="nsew")
        scroll_box.columnconfigure(0, weight=1)
        scroll_box.rowconfigure(0, weight=1)

        self.left_form_canvas = tk.Canvas(
            scroll_box,
            bg=SURFACE_BG,
            borderwidth=0,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        self.left_form_canvas.grid(row=0, column=0, sticky="nsew")
        self.left_form_scrollbar = ttk.Scrollbar(scroll_box, orient=tk.VERTICAL, command=self.left_form_canvas.yview)
        self.left_form_scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.left_form_canvas.configure(yscrollcommand=self.left_form_scrollbar.set)

        self.left_form_content = ttk.Frame(self.left_form_canvas, style="Card.TFrame", padding=(0, 0, 4, 0))
        self.left_form_window_id = self.left_form_canvas.create_window((0, 0), window=self.left_form_content, anchor="nw")
        self.left_form_content.bind("<Configure>", self._on_left_form_content_configure)
        self.left_form_canvas.bind("<Configure>", self._on_left_form_canvas_configure)

        self._build_left_form_content(self.left_form_content)
        self._bind_left_form_mousewheel_widgets(self.left_form_content)
        self.after_idle(self._refresh_left_form_scrollregion)

        self.left_log_box = ttk.LabelFrame(parent, text="运行日志", padding=12)
        self.left_log_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.left_log_box.columnconfigure(0, weight=1)
        self.left_log_box.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(self.left_log_box, height=12, wrap=tk.WORD, borderwidth=0, relief=tk.FLAT)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state=tk.DISABLED)

    def _build_left_form_content(self, parent: ttk.Frame) -> None:
        model_box = ttk.LabelFrame(parent, text="模型包管理", padding=12)
        model_box.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(model_box, text="KAN 模型目录").pack(anchor=tk.W)
        ttk.Entry(model_box, textvariable=self.kan_dir_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(model_box, text="选择 KAN 目录", command=self._choose_kan_dir, style="Secondary.TButton").pack(fill=tk.X)
        ttk.Label(model_box, text="MLP 模型目录").pack(anchor=tk.W, pady=(12, 0))
        ttk.Entry(model_box, textvariable=self.mlp_dir_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(model_box, text="选择 MLP 目录", command=self._choose_mlp_dir, style="Secondary.TButton").pack(fill=tk.X)
        ttk.Button(
            model_box,
            text="按当前目录加载模型",
            command=self._load_models_from_current_inputs,
            style="Secondary.TButton",
        ).pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            model_box,
            text="恢复默认模型目录",
            command=self._load_default_models,
            style="Secondary.TButton",
        ).pack(fill=tk.X, pady=(8, 0))

        forward_prediction_box = ttk.LabelFrame(parent, text="正向预测", padding=12)
        forward_prediction_box.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(forward_prediction_box, text="前向 KAN 模型目录").pack(anchor=tk.W)
        ttk.Entry(forward_prediction_box, textvariable=self.forward_kan_dir_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(
            forward_prediction_box,
            text="选择前向 KAN 目录",
            command=self._choose_forward_kan_dir,
            style="Secondary.TButton",
        ).pack(fill=tk.X)
        ttk.Label(
            forward_prediction_box,
            text="默认使用论文主结果中的最佳前向模型 KAN。",
            style="Note.TLabel",
            wraplength=300,
        ).pack(anchor=tk.W, pady=(8, 8))
        self._add_labeled_entry(forward_prediction_box, "开度 (mm)", self.forward_opening_var)
        self._add_labeled_entry(forward_prediction_box, "转速 (r/min)", self.forward_speed_var)
        self.forward_predict_button = ttk.Button(
            forward_prediction_box,
            text="执行正向预测",
            command=self._run_forward_prediction,
            style="Primary.TButton",
        )
        self.forward_predict_button.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            forward_prediction_box,
            textvariable=self.forward_prediction_var,
            wraplength=300,
            justify=tk.LEFT,
            bg=SURFACE_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor=tk.W, pady=(10, 0))

        prescription_box = ttk.LabelFrame(parent, text="处方图", padding=12)
        prescription_box.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(prescription_box, text="处方图 CSV 路径").pack(anchor=tk.W)
        ttk.Entry(prescription_box, textvariable=self.prescription_path_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(
            prescription_box,
            text="导入处方图 CSV",
            command=self._choose_prescription_csv,
            style="Secondary.TButton",
        ).pack(fill=tk.X)
        ttk.Button(
            prescription_box,
            text="加载示例处方图",
            command=self._load_sample_prescription,
            style="Secondary.TButton",
        ).pack(fill=tk.X, pady=(8, 0))

        machine_box = ttk.LabelFrame(parent, text="机器参数", padding=12)
        machine_box.pack(fill=tk.X, pady=(0, 12))
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
        ttk.Button(
            machine_box,
            text="恢复默认机器参数",
            command=self._reset_machine_defaults,
            style="Secondary.TButton",
        ).pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(
            machine_box,
            text="运行仿真",
            command=self._run_simulation,
            style="Primary.TButton",
        )
        self.run_button.pack(fill=tk.X, pady=(8, 0))

        export_box = ttk.LabelFrame(parent, text="导出设置", padding=12)
        export_box.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(export_box, text="导出根目录").pack(anchor=tk.W)
        ttk.Entry(export_box, textvariable=self.output_root_var).pack(fill=tk.X, pady=(4, 8))
        ttk.Button(
            export_box,
            text="选择导出目录",
            command=self._choose_output_root,
            style="Secondary.TButton",
        ).pack(fill=tk.X)

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=0)
        parent.columnconfigure(2, weight=0)
        parent.rowconfigure(2, weight=1)

        self.left_toggle_button = ttk.Button(
            parent,
            textvariable=self.left_toggle_text,
            command=self._toggle_left_panel,
            style="Secondary.TButton",
        )
        self.left_toggle_button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.right_toggle_button = ttk.Button(
            parent,
            textvariable=self.right_toggle_text,
            command=self._toggle_right_panel,
            style="Secondary.TButton",
        )
        self.right_toggle_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

        ttk.Label(parent, text="处方热力图与机具轨迹", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            parent,
            text="统一使用 Matplotlib 三视图预览，保持界面预览与导出图像的一致性。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 8))

        self.preview_notebook = ttk.Notebook(parent)
        self.preview_notebook.grid(row=2, column=0, columnspan=3, sticky="nsew")
        self.preview_notebook.bind("<<NotebookTabChanged>>", self._on_preview_tab_changed)

        for key, title in self.PREVIEW_TITLES.items():
            preview_frame = ttk.Frame(self.preview_notebook, style="Card.TFrame", padding=6)
            preview_frame.pack_propagate(False)
            status_var = tk.StringVar(value="")
            host = ttk.Frame(preview_frame, style="Card.TFrame")
            placeholder = tk.Label(
                preview_frame,
                textvariable=status_var,
                justify=tk.CENTER,
                anchor="center",
                fg=TEXT_MUTED,
                bg=SURFACE_BG,
                font=("Microsoft YaHei UI", 12),
            )
            placeholder.pack(fill=tk.BOTH, expand=True)

            self.preview_notebook.add(preview_frame, text=title)
            tab_id = self.preview_notebook.tabs()[-1]
            self.preview_frames[key] = preview_frame
            self.preview_hosts[key] = host
            self.preview_placeholders[key] = placeholder
            self.preview_status_vars[key] = status_var
            self.preview_tab_keys[tab_id] = key
            self.preview_tab_ids[key] = tab_id

        slider_frame = ttk.Frame(parent, style="Card.TFrame")
        slider_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        slider_frame.columnconfigure(0, weight=1)
        tk.Label(slider_frame, text="时间步", bg=SURFACE_BG, fg=TEXT_PRIMARY, font=("Microsoft YaHei UI", 10)).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.frame_slider = tk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=self._on_slider_changed,
            width=24,
            sliderlength=36,
            borderwidth=0,
            highlightthickness=0,
            relief=tk.FLAT,
            bg=SURFACE_BG,
            fg=TEXT_PRIMARY,
            activebackground=SLIDER_HANDLE_DISABLED,
            troughcolor=SLIDER_TROUGH_DISABLED,
        )
        self.frame_slider.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        self.frame_slider.bind("<ButtonRelease-1>", self._on_slider_released)
        self.frame_slider.bind("<KeyRelease>", self._on_slider_released)
        tk.Label(
            slider_frame,
            textvariable=self.frame_info_var,
            wraplength=920,
            bg=SURFACE_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=2, column=0, sticky="w")

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        summary_box = ttk.LabelFrame(parent, text="仿真摘要", padding=12)
        summary_box.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            summary_box,
            textvariable=self.summary_var,
            wraplength=320,
            justify=tk.LEFT,
            bg=SURFACE_BG,
            fg=TEXT_PRIMARY,
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor=tk.W)

        actions = ttk.Frame(summary_box, style="Card.TFrame")
        actions.pack(fill=tk.X, pady=(12, 0))
        self.export_button = ttk.Button(
            actions,
            text="导出结果与地图",
            command=self._export_results,
            style="Primary.TButton",
        )
        self.export_button.pack(side=tk.LEFT, padx=(0, 8))
        self.open_export_button = ttk.Button(
            actions,
            text="打开导出目录",
            command=self._open_last_export_dir,
            style="Secondary.TButton",
        )
        self.open_export_button.pack(side=tk.LEFT)

        table_box = ttk.LabelFrame(parent, text="当前时刻单排行决策", padding=12)
        table_box.pack(fill=tk.BOTH, expand=True)
        columns = ("row", "zone", "rate", "mass", "opening", "speed", "model", "status")
        self.decision_table = ttk.Treeview(table_box, columns=columns, show="headings", height=18)
        headings = {
            "row": "行号",
            "zone": "分区",
            "rate": "目标量 (kg/ha)",
            "mass": "目标排肥量 (g/min)",
            "opening": "开度 (mm)",
            "speed": "目标转速 (r/min)",
            "model": "模型",
            "status": "状态",
        }
        widths = {
            "row": 56,
            "zone": 74,
            "rate": 116,
            "mass": 138,
            "opening": 96,
            "speed": 122,
            "model": 96,
            "status": 108,
        }
        numeric_columns = {"rate", "mass", "opening", "speed"}
        for column in columns:
            anchor = tk.E if column in numeric_columns else tk.CENTER
            self.decision_table.heading(column, text=headings[column], anchor=anchor)
            self.decision_table.column(column, width=widths[column], anchor=anchor, stretch=column != "row")

        self.decision_table.tag_configure("oddrow", background="#ffffff")
        self.decision_table.tag_configure("evenrow", background="#f8fafc")

        y_scroll = ttk.Scrollbar(table_box, orient=tk.VERTICAL, command=self.decision_table.yview)
        self.decision_table.configure(yscrollcommand=y_scroll.set)
        self.decision_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _refresh_panel_toggle_texts(self) -> None:
        self.left_toggle_text.set("展开左栏" if self.left_collapsed else "收起左栏")
        self.right_toggle_text.set("展开右栏" if self.right_collapsed else "收起右栏")

    def _remember_visible_panel_widths(self) -> None:
        if self.main_pane is None:
            return

        self.update_idletasks()
        pane_names = tuple(self.main_pane.panes())
        total_width = self.main_pane.winfo_width()
        if self.left_panel is not None and not self.left_collapsed:
            left_width = self.left_panel.winfo_width()
            if left_width <= 1 and len(pane_names) >= 2:
                left_width = self.main_pane.sashpos(0)
            if left_width > 1:
                self.saved_left_width = int(left_width)
        if self.right_panel is not None and not self.right_collapsed:
            right_width = self.right_panel.winfo_width()
            if right_width <= 1 and len(pane_names) >= 2:
                if not self.left_collapsed and len(pane_names) >= 3:
                    right_width = total_width - self.main_pane.sashpos(1)
                else:
                    right_width = total_width - self.main_pane.sashpos(0)
            if right_width > 1:
                self.saved_right_width = int(right_width)

    def _default_panel_width(self, side: str) -> int:
        total_width = self.main_pane.winfo_width() if self.main_pane is not None else 1800
        ratio = 0.30 if side == "left" else 0.20
        return max(int(total_width * ratio), MIN_SIDE_PANEL_WIDTH)

    def _target_panel_width(self, side: str) -> int:
        if side == "left":
            saved_width = self.saved_left_width
        else:
            saved_width = self.saved_right_width
        if saved_width is not None and saved_width > 1:
            return saved_width
        return self._default_panel_width(side)

    def _clamp_side_panel_width(self, target_width: int, total_width: int, reserved_other: int = 0) -> int:
        max_width = total_width - MIN_CENTER_PANEL_WIDTH - reserved_other
        if max_width < MIN_SIDE_PANEL_WIDTH:
            max_width = max(MIN_SIDE_PANEL_WIDTH, total_width // 2)
        return max(MIN_SIDE_PANEL_WIDTH, min(int(target_width), max_width))

    def _restore_side_panel_widths(self) -> None:
        if self.main_pane is None:
            return

        self.update_idletasks()
        total_width = self.main_pane.winfo_width()
        if total_width <= 1:
            return

        sash_positions: list[tuple[int, int]] = []
        if not self.left_collapsed and not self.right_collapsed:
            left_target = self._clamp_side_panel_width(self._target_panel_width("left"), total_width, MIN_SIDE_PANEL_WIDTH)
            right_target = self._clamp_side_panel_width(self._target_panel_width("right"), total_width, left_target)
            sash_positions = [(0, left_target), (1, total_width - right_target)]
        elif not self.left_collapsed:
            left_target = self._clamp_side_panel_width(self._target_panel_width("left"), total_width)
            sash_positions = [(0, left_target)]
        elif not self.right_collapsed:
            right_target = self._clamp_side_panel_width(self._target_panel_width("right"), total_width)
            sash_positions = [(0, total_width - right_target)]

        for index, position in sash_positions:
            self.main_pane.sashpos(index, position)

        if sash_positions:
            self.update_idletasks()
            for index, position in sash_positions:
                self.main_pane.sashpos(index, position)
        self.after_idle(self._refresh_left_form_scrollregion)

    def _toggle_left_panel(self) -> None:
        if self.main_pane is None or self.left_panel is None:
            return

        if self.left_collapsed:
            self.main_pane.insert(0, self.left_panel, weight=LEFT_PANEL_WEIGHT)
            self.left_collapsed = False
            self.after_idle(self._restore_side_panel_widths)
            self.after_idle(self._refresh_left_form_scrollregion)
        else:
            self._remember_visible_panel_widths()
            self.main_pane.forget(self.left_panel)
            self.left_collapsed = True
        self._refresh_panel_toggle_texts()

    def _toggle_right_panel(self) -> None:
        if self.main_pane is None or self.right_panel is None:
            return

        if self.right_collapsed:
            self.main_pane.add(self.right_panel, weight=RIGHT_PANEL_WEIGHT)
            self.right_collapsed = False
            self.after_idle(self._restore_side_panel_widths)
        else:
            self._remember_visible_panel_widths()
            self.main_pane.forget(self.right_panel)
            self.right_collapsed = True
        self._refresh_panel_toggle_texts()

    def _on_left_form_content_configure(self, _event=None) -> None:
        self._refresh_left_form_scrollregion()

    def _on_left_form_canvas_configure(self, event) -> None:
        if self.left_form_canvas is None or self.left_form_window_id is None:
            return
        self.left_form_canvas.itemconfigure(self.left_form_window_id, width=max(int(event.width), 1))
        self._refresh_left_form_scrollregion()

    def _refresh_left_form_scrollregion(self) -> None:
        if self.left_form_canvas is None:
            return
        self.left_form_canvas.update_idletasks()
        bbox = self.left_form_canvas.bbox("all")
        if bbox is None:
            self.left_form_canvas.configure(scrollregion=(0, 0, 0, 0))
            return
        self.left_form_canvas.configure(scrollregion=bbox)

    def _bind_left_form_mousewheel_widgets(self, root: tk.Misc) -> None:
        sequences = ("<MouseWheel>", "<Button-4>", "<Button-5>")
        for widget in self._iter_widget_tree(root):
            for sequence in sequences:
                widget.bind(sequence, self._on_left_form_mousewheel, add="+")

    def _iter_widget_tree(self, widget: tk.Misc):
        yield widget
        for child in widget.winfo_children():
            yield from self._iter_widget_tree(child)

    def _on_left_form_mousewheel(self, event) -> str:
        if self.left_form_canvas is None:
            return "break"

        delta = 0
        if getattr(event, "delta", 0):
            delta = -int(event.delta / 120)
            if delta == 0:
                delta = -1 if event.delta > 0 else 1
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1

        if delta != 0:
            self.left_form_canvas.yview_scroll(delta, "units")
        return "break"

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, note: str | None = None) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(0, 8))
        tk.Label(row, text=label, width=14, bg=SURFACE_BG, fg=TEXT_PRIMARY, font=("Microsoft YaHei UI", 10)).pack(
            side=tk.LEFT
        )
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if note:
            ttk.Label(parent, text=note, style="Note.TLabel", wraplength=300).pack(anchor=tk.W, pady=(0, 6))

    def _choose_kan_dir(self) -> None:
        path = filedialog.askdirectory(title="选择 inverse_KAN 模型目录", initialdir=self.kan_dir_var.get() or str(Path.cwd()))
        if path:
            self.kan_dir_var.set(path)

    def _choose_mlp_dir(self) -> None:
        path = filedialog.askdirectory(title="选择 inverse_MLP 模型目录", initialdir=self.mlp_dir_var.get() or str(Path.cwd()))
        if path:
            self.mlp_dir_var.set(path)

    def _choose_forward_kan_dir(self) -> None:
        path = filedialog.askdirectory(
            title="选择 forward_KAN 模型目录",
            initialdir=self.forward_kan_dir_var.get() or str(Path.cwd()),
        )
        if path:
            self.forward_kan_dir_var.set(path)

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
        self.forward_kan_dir_var.set(str(DEFAULT_FORWARD_KAN_ARTIFACT_DIR))
        self._load_models_from_current_inputs(show_message=show_message)

    def _load_inverse_models_from_current_inputs(self, *, log_success: bool = True):
        kan_bundle, mlp_bundle = self.controller.load_models_from_dirs(self.kan_dir_var.get(), self.mlp_dir_var.get())
        if log_success:
            self._log(f"已加载决策模型：{kan_bundle.config.name}、{mlp_bundle.config.name}")
        return kan_bundle, mlp_bundle

    def _load_forward_model_from_current_input(self, *, log_success: bool = True):
        forward_bundle = self.controller.load_forward_model_from_dir(self.forward_kan_dir_var.get())
        if log_success:
            self._log(f"已加载预测模型：{forward_bundle.config.name}")
        return forward_bundle

    def _load_models_from_current_inputs(self, show_message: bool = True) -> None:
        try:
            kan_bundle, mlp_bundle = self._load_inverse_models_from_current_inputs(log_success=False)
            forward_bundle = self._load_forward_model_from_current_input(log_success=False)
            self._log(
                "已加载决策模型："
                f"{kan_bundle.config.name}、{mlp_bundle.config.name}；"
                f"已加载预测模型：{forward_bundle.config.name}"
            )
            if show_message:
                messagebox.showinfo(
                    "模型加载完成",
                    "已成功加载决策模型（inverse_KAN、inverse_MLP）和预测模型（forward_KAN）。",
                )
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
            self._reset_result_views(summary_message="处方图已导入，运行仿真后即可查看三视图预览。")
        except Exception as exc:  # noqa: BLE001
            self._log(f"处方图导入失败：{exc}")
            messagebox.showerror("处方图导入失败", str(exc))

    def _parse_machine_config(self) -> MachineConfig:
        row_offsets_text = self.row_offsets_var.get().strip()
        row_offsets = [float(item.strip()) for item in row_offsets_text.split(",") if item.strip()] if row_offsets_text else []
        return MachineConfig(
            row_count=int(self.row_count_var.get()),
            row_spacing_m=float(self.row_spacing_var.get()),
            travel_speed_kmh=float(self.travel_speed_var.get()),
            sample_period_ms=int(self.sample_period_var.get()),
            machine_center_to_row_origin_m=float(self.longitudinal_offset_var.get()),
            row_offsets_m=row_offsets,
        )

    def _parse_forward_prediction_inputs(self) -> tuple[float, float]:
        opening_text = self.forward_opening_var.get().strip()
        speed_text = self.forward_speed_var.get().strip()
        if not opening_text or not speed_text:
            raise ValueError("请输入开度和转速。")
        try:
            opening_mm = float(opening_text)
            speed_r_min = float(speed_text)
        except ValueError as exc:
            raise ValueError("开度和转速必须为数字。") from exc
        if opening_mm < 0 or speed_r_min < 0:
            raise ValueError("开度和转速不能为负数。")
        return opening_mm, speed_r_min

    def _forward_rate_context(self) -> tuple[float | None, float | None]:
        try:
            row_spacing_m = float(self.row_spacing_var.get())
            travel_speed_kmh = float(self.travel_speed_var.get())
        except ValueError:
            return None, None
        if row_spacing_m <= 0 or travel_speed_kmh <= 0:
            return None, None
        return row_spacing_m, travel_speed_kmh

    def _forward_domain_status_text(self, domain_status: str) -> str:
        mapping = {
            "in_domain": "训练域内",
            "opening_extrapolation": "开度外推",
            "speed_extrapolation": "转速外推",
            "opening_and_speed_extrapolation": "开度与转速外推",
        }
        return mapping.get(domain_status, domain_status)

    def _forward_prediction_status_text(self, status: str) -> str:
        mapping = {
            "ok": "正常",
            "clamped_low": "预测结果低于 0，已钳制到 0",
        }
        return mapping.get(status, status)

    def _format_forward_prediction(self, result: ForwardPredictionResult) -> str:
        lines = [
            f"输入：开度 {result.opening_mm:.1f} mm，转速 {result.speed_r_min:.2f} r/min",
            f"预测排肥量：{result.predicted_mass_g_min:.2f} g/min",
        ]
        if result.equivalent_rate_kg_ha is None:
            lines.append("等效施肥量：需填写有效行距和作业速度后显示")
        else:
            lines.append(f"等效施肥量：{result.equivalent_rate_kg_ha:.2f} kg/ha")
        lines.extend(
            [
                f"模型：{result.selected_model}",
                f"训练域状态：{self._forward_domain_status_text(result.domain_status)}",
                f"预测状态：{self._forward_prediction_status_text(result.status)}",
            ]
        )
        return "\n".join(lines)

    def _reset_machine_defaults(self) -> None:
        self.row_count_var.set("6")
        self.row_spacing_var.set("0.6")
        self.travel_speed_var.set("6.0")
        self.sample_period_var.set("200")
        self.longitudinal_offset_var.set("0.0")
        self.row_offsets_var.set("")
        self._log("已恢复默认机器参数。")

    def _run_forward_prediction(self) -> None:
        try:
            if self.controller.forward_kan_bundle is None:
                self._load_forward_model_from_current_input(log_success=False)

            opening_mm, speed_r_min = self._parse_forward_prediction_inputs()
            row_spacing_m, travel_speed_kmh = self._forward_rate_context()
            result = self.controller.predict_forward_mass(
                opening_mm,
                speed_r_min,
                row_spacing_m=row_spacing_m,
                travel_speed_kmh=travel_speed_kmh,
            )
            self.forward_prediction_var.set(self._format_forward_prediction(result))
            self._log(
                "前向预测："
                f"开度 {result.opening_mm:.1f} mm，"
                f"转速 {result.speed_r_min:.2f} r/min，"
                f"排肥量 {result.predicted_mass_g_min:.2f} g/min，"
                f"训练域状态 {self._forward_domain_status_text(result.domain_status)}"
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"前向预测失败：{exc}")
            messagebox.showerror("前向预测失败", str(exc))

    def _run_simulation(self) -> None:
        try:
            if self.controller.router is None:
                self._load_inverse_models_from_current_inputs(log_success=False)
            if self.controller.prescription_map is None:
                if self.prescription_path_var.get():
                    self._load_prescription(self.prescription_path_var.get())
                if self.controller.prescription_map is None:
                    raise ValueError("请先导入处方图。")

            result = self.controller.run_simulation(self._parse_machine_config())
            self.current_frame_index = 0
            self.last_export_dir = None
            self._preview_result_token = None
            self._cancel_live_preview_update()
            self.frame_slider.configure(to=max(len(result.frames) - 1, 0))
            self._set_slider_value(0)
            self._table_dirty = False
            self._table_frame_index = None
            self._set_result_state(has_result=True)
            self._refresh_summary()
            self._refresh_current_frame_details(refresh_table=True)
            self._initialize_simulation_previews(result)
            self._render_preview_tabs(force=True)
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

    def _refresh_current_frame_details(self, refresh_table: bool = True) -> None:
        result = self.controller.last_result
        if result is None or not result.frames:
            self.frame_info_var.set("当前还没有仿真结果。")
            if refresh_table:
                self._clear_decision_table()
                self._table_dirty = False
                self._table_frame_index = None
            return

        self.current_frame_index = self._normalized_slider_index(self.current_frame_index)
        frame = result.frames[self.current_frame_index]
        self.frame_info_var.set(
            f"时间戳：{frame.timestamp_ms} ms，作业趟次：第 {frame.pass_id} 趟，"
            f"机具中心：({frame.machine_center_x_m:.2f}, {frame.machine_center_y_m:.2f}) m"
        )

        if not refresh_table:
            return

        self._clear_decision_table()
        for index, decision in enumerate(frame.row_decisions):
            tag = "evenrow" if index % 2 else "oddrow"
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
                tags=(tag,),
            )

        self._table_dirty = False
        self._table_frame_index = self.current_frame_index

    def _clear_decision_table(self) -> None:
        for item in self.decision_table.get_children():
            self.decision_table.delete(item)

    def _on_slider_changed(self, value: str) -> None:
        if self._slider_events_suspended:
            return
        result = self.controller.last_result
        if result is None or not result.frames:
            return

        index = self._normalized_slider_index(value)
        if index == self.current_frame_index and self._pending_live_frame_index is None:
            return

        self.current_frame_index = index
        self._pending_live_frame_index = index
        self._table_dirty = True
        self._refresh_current_frame_details(refresh_table=False)
        self._mark_previews_dirty(self.INTERACTIVE_PREVIEW_KEYS)

        selected = self._selected_preview_key()
        if selected in self.INTERACTIVE_PREVIEW_KEYS:
            self._schedule_live_preview_update(selected)

    def _on_slider_released(self, _event) -> None:
        result = self.controller.last_result
        if result is None or not result.frames:
            return

        self._cancel_live_preview_update()
        self.current_frame_index = self._normalized_slider_index()
        self._pending_live_frame_index = None
        self._refresh_current_frame_details(refresh_table=True)

        selected = self._selected_preview_key()
        if selected == "overview":
            self._render_simulation_preview("overview", detailed=False, immediate=True)
            self._mark_previews_dirty(("current",))
        elif selected == "current":
            self._render_simulation_preview("current", detailed=True, immediate=True)
            self._mark_previews_dirty(("overview",))
        else:
            self._mark_previews_dirty(self.INTERACTIVE_PREVIEW_KEYS)

    def _normalized_slider_index(self, value: str | int | None = None) -> int:
        result = self.controller.last_result
        if result is None or not result.frames:
            return 0

        raw_value = self.frame_slider.get() if value is None else value
        try:
            index = int(round(float(raw_value)))
        except (TypeError, ValueError):
            index = 0
        return max(0, min(index, len(result.frames) - 1))

    def _set_slider_value(self, index: int) -> None:
        self._slider_events_suspended = True
        try:
            self.frame_slider.set(index)
        finally:
            self._slider_events_suspended = False

    def _set_result_state(self, has_result: bool) -> None:
        slider_state = tk.NORMAL if has_result else tk.DISABLED
        self.frame_slider.configure(
            state=slider_state,
            troughcolor=SLIDER_TROUGH_ACTIVE if has_result else SLIDER_TROUGH_DISABLED,
            activebackground=SLIDER_HANDLE_ACTIVE if has_result else SLIDER_HANDLE_DISABLED,
        )
        self.export_button.configure(state=tk.NORMAL if has_result else tk.DISABLED)
        can_open_export = bool(has_result and self.last_export_dir and self.last_export_dir.exists())
        self.open_export_button.configure(state=tk.NORMAL if can_open_export else tk.DISABLED)

    def _mark_previews_dirty(self, preview_keys: tuple[str, ...] | list[str] | set[str] | None = None) -> None:
        keys = tuple(preview_keys or self.PREVIEW_TITLES.keys())
        self.preview_dirty_keys.update(keys)

    def _schedule_live_preview_update(self, preview_key: str) -> None:
        if preview_key not in self.INTERACTIVE_PREVIEW_KEYS:
            return
        self._live_preview_key = preview_key
        if self._live_preview_after_id is None:
            self._live_preview_after_id = self.after(self.LIVE_PREVIEW_INTERVAL_MS, self._process_live_preview_update)

    def _cancel_live_preview_update(self) -> None:
        if self._live_preview_after_id is not None:
            try:
                self.after_cancel(self._live_preview_after_id)
            except tk.TclError:
                pass
        self._live_preview_after_id = None
        self._live_preview_key = None

    def _cancel_scheduled_preview_refresh(self) -> None:
        self._cancel_live_preview_update()

    def _process_live_preview_update(self) -> None:
        self._live_preview_after_id = None
        result = self.controller.last_result
        preview_key = self._selected_preview_key()
        pending_index = self._pending_live_frame_index
        if result is None or not result.frames or pending_index is None or preview_key not in self.INTERACTIVE_PREVIEW_KEYS:
            return

        self._pending_live_frame_index = None
        if preview_key == "overview":
            self._render_simulation_preview("overview", detailed=False, immediate=False)
        else:
            self._render_simulation_preview("current", detailed=False, immediate=False)

        if self._pending_live_frame_index is not None:
            self._schedule_live_preview_update(preview_key)

    def _refresh_preview_tabs(self, force: bool = False, preview_keys: tuple[str, ...] | None = None) -> None:
        self._render_preview_tabs(force=force, preview_keys=preview_keys)

    def _render_pending_preview_tabs(self) -> None:
        preview_keys = tuple(self.preview_dirty_keys) if self.preview_dirty_keys else (self._selected_preview_key(),)
        self._render_preview_tabs(force=True, preview_keys=preview_keys)

    def _render_preview_tabs(self, force: bool = False, preview_keys: tuple[str, ...] | None = None) -> None:
        del force
        keys = preview_keys or tuple(self.PREVIEW_TITLES.keys())
        prescription = self.controller.prescription_map
        result = self.controller.last_result

        if prescription is None:
            for key in keys:
                self._show_preview_placeholder(key, "请先导入处方图并运行仿真。")
            return

        if result is None or not result.frames:
            self._preview_result_token = None
            for key in keys:
                if key == "overview":
                    figure = create_prescription_overview_figure(prescription)
                    self._show_preview_figure("overview", figure)
                elif key == "legend":
                    figure = create_prescription_legend_figure(prescription)
                    self._show_preview_figure("legend", figure)
                else:
                    self._show_preview_placeholder("current", "运行仿真后可查看当前帧细节图。")
            return

        self._initialize_simulation_previews(result)
        for key in keys:
            if key == "legend":
                self.preview_dirty_keys.discard("legend")
                continue
            if key == "current":
                self._render_simulation_preview("current", detailed=True, immediate=True)
            else:
                self._render_simulation_preview("overview", detailed=False, immediate=True)

    def _initialize_simulation_previews(self, result) -> None:
        if (
            self._preview_result_token == id(result)
            and "overview" in self.preview_renderers
            and "current" in self.preview_renderers
            and "legend" in self.preview_canvases
        ):
            return

        for key in self.PREVIEW_TITLES:
            self._clear_preview(key)

        overview_renderer = create_overview_preview_state(result, frame_index=self.current_frame_index)
        current_renderer = create_current_preview_state(result, frame_index=self.current_frame_index, detailed=True)
        legend_figure = create_legend_figure(result)

        self._mount_preview_figure("overview", overview_renderer.figure, renderer=overview_renderer)
        self._mount_preview_figure("current", current_renderer.figure, renderer=current_renderer)
        self._mount_preview_figure("legend", legend_figure)

        self.preview_canvases["overview"].draw()
        self.preview_canvases["current"].draw()
        self.preview_canvases["legend"].draw()
        self.preview_dirty_keys.clear()
        self._preview_result_token = id(result)

    def _render_simulation_preview(self, preview_key: str, *, detailed: bool, immediate: bool) -> None:
        if preview_key not in self.preview_renderers:
            result = self.controller.last_result
            if result is None or not result.frames:
                return
            self._initialize_simulation_previews(result)

        renderer = self.preview_renderers.get(preview_key)
        canvas = self.preview_canvases.get(preview_key)
        if renderer is None or canvas is None:
            return

        if preview_key == "overview":
            renderer.update_frame(self.current_frame_index)
        else:
            renderer.update_frame(self.current_frame_index, detailed=detailed)

        self._show_preview_host(preview_key)
        if immediate:
            canvas.draw()
        else:
            canvas.draw_idle()

        if preview_key == "current" and not detailed:
            self.preview_dirty_keys.add("current")
        else:
            self.preview_dirty_keys.discard(preview_key)

    def _selected_preview_key(self) -> str:
        tab_id = self.preview_notebook.select()
        return self.preview_tab_keys.get(tab_id, "overview")

    def _on_preview_tab_changed(self, _event) -> None:
        selected = self._selected_preview_key()
        if selected not in self.PREVIEW_TITLES:
            return

        result = self.controller.last_result
        if result is None or not result.frames:
            self._render_preview_tabs(force=True, preview_keys=(selected,))
            return

        self._initialize_simulation_previews(result)
        if selected in self.preview_dirty_keys or selected not in self.preview_canvases:
            if selected == "current":
                self._render_simulation_preview("current", detailed=True, immediate=True)
            elif selected == "overview":
                self._render_simulation_preview("overview", detailed=False, immediate=True)

        if self._pending_live_frame_index is not None and selected in self.INTERACTIVE_PREVIEW_KEYS:
            self._schedule_live_preview_update(selected)

    def _show_preview_placeholder(self, preview_key: str, message: str) -> None:
        self.preview_status_vars[preview_key].set(message)
        self._clear_preview(preview_key)
        host = self.preview_hosts[preview_key]
        placeholder = self.preview_placeholders[preview_key]
        host.pack_forget()
        placeholder.pack(fill=tk.BOTH, expand=True)

    def _show_preview_host(self, preview_key: str) -> None:
        host = self.preview_hosts[preview_key]
        placeholder = self.preview_placeholders[preview_key]
        placeholder.pack_forget()
        if not host.winfo_manager():
            host.pack(fill=tk.BOTH, expand=True)

    def _show_preview_figure(self, preview_key: str, figure: object) -> None:
        self._mount_preview_figure(preview_key, figure)
        canvas = self.preview_canvases[preview_key]
        canvas.draw()

    def _mount_preview_figure(
        self,
        preview_key: str,
        figure: object,
        *,
        renderer: OverviewPreviewState | CurrentFramePreviewState | None = None,
    ) -> None:
        if self.preview_figures.get(preview_key) is figure and preview_key in self.preview_canvases:
            if renderer is not None:
                self.preview_renderers[preview_key] = renderer
            self._show_preview_host(preview_key)
            return

        self._clear_preview(preview_key)
        host = self.preview_hosts[preview_key]
        canvas = FigureCanvasTkAgg(figure, master=host)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.preview_canvases[preview_key] = canvas
        self.preview_figures[preview_key] = figure
        if renderer is not None:
            self.preview_renderers[preview_key] = renderer
        self._show_preview_host(preview_key)

    def _clear_preview(self, preview_key: str) -> None:
        canvas = self.preview_canvases.pop(preview_key, None)
        if canvas is not None:
            try:
                canvas.get_tk_widget().destroy()
            except tk.TclError:
                pass

        figure = self.preview_figures.pop(preview_key, None)
        if figure is not None:
            plt.close(figure)

        self.preview_renderers.pop(preview_key, None)
        host = self.preview_hosts.get(preview_key)
        if host is not None:
            for child in host.winfo_children():
                child.destroy()

    def _reset_result_views(self, summary_message: str = "当前没有仿真结果。") -> None:
        self._cancel_live_preview_update()
        self.controller.last_result = None
        self.current_frame_index = 0
        self.last_export_dir = None
        self._preview_result_token = None
        self._pending_live_frame_index = None
        self._live_preview_key = None
        self._table_dirty = False
        self._table_frame_index = None
        self.frame_slider.configure(to=0)
        self._set_slider_value(0)
        self.frame_info_var.set("当前还没有仿真结果。")
        self.summary_var.set(summary_message)
        self._clear_decision_table()
        for key in self.PREVIEW_TITLES:
            self._clear_preview(key)
        self.preview_dirty_keys = set(self.PREVIEW_TITLES)
        self._set_result_state(has_result=False)
        self._render_preview_tabs(force=True)

    def _export_results(self) -> None:
        try:
            artifacts = self.controller.export_last_result(
                self.output_root_var.get(),
                highlighted_frame_index=self.current_frame_index,
            )
            self.last_export_dir = artifacts.output_dir
            self._set_result_state(has_result=self.controller.last_result is not None)
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
        if self.last_export_dir is None or not self.last_export_dir.exists():
            messagebox.showwarning("未导出", "请先导出一次结果。")
            return
        if hasattr(os, "startfile"):
            os.startfile(self.last_export_dir)  # type: ignore[attr-defined]

    def _log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def destroy(self) -> None:
        self._cancel_live_preview_update()
        for key in tuple(self.PREVIEW_TITLES.keys()):
            self._clear_preview(key)
        super().destroy()


def launch_app() -> None:
    app = FertilizerApp(auto_load_models=True)
    if sv_ttk is not None:
        try:
            sv_ttk.set_theme("light")
        except Exception:
            pass
    app.mainloop()
