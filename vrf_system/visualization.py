from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .annotations import format_row_annotation
from .domain import PrescriptionCell, SimulationFrame, SimulationResult
from .prescription import PrescriptionMap


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MAP_CMAP = cm.get_cmap("YlGnBu")
KAN_COLOR = "#059669"
MLP_COLOR = "#d97706"
OUT_COLOR = "#6b7280"
PATH_COLOR = "#2563eb"
SPAN_COLOR = "#111827"
EMPTY_OFFSETS = np.empty((0, 2))


@dataclass(slots=True)
class OverviewPreviewState:
    result: SimulationResult
    figure: object
    ax: object
    center_artist: object
    span_artist: object
    kan_artist: object
    mlp_artist: object
    out_artist: object
    annotation_artists: list[object]
    rendered_frame_index: int = 0

    def update_frame(self, frame_index: int) -> None:
        index = _clamp_frame_index(self.result, frame_index)
        frame = self.result.frames[index]
        valid_rows, kan_points, mlp_points, out_points = _group_row_points(frame)

        if valid_rows:
            self.span_artist.set_data(
                [decision.x_m for decision in valid_rows],
                [decision.y_m for decision in valid_rows],
            )
        else:
            self.span_artist.set_data([], [])

        self.center_artist.set_offsets(_points_array([(frame.machine_center_x_m, frame.machine_center_y_m)]))
        self.kan_artist.set_offsets(_points_array(kan_points))
        self.mlp_artist.set_offsets(_points_array(mlp_points))
        self.out_artist.set_offsets(_points_array(out_points))
        self._update_annotations(frame)
        self.rendered_frame_index = index

    def _update_annotations(self, frame: SimulationFrame) -> None:
        xlim = tuple(float(value) for value in self.ax.get_xlim())
        ylim = tuple(float(value) for value in self.ax.get_ylim())

        for annotation, decision in zip(self.annotation_artists, frame.row_decisions):
            label_x, label_y, ha, va = _overview_annotation_position(
                decision,
                row_spacing_m=self.result.machine_config.row_spacing_m,
                xlim=xlim,
                ylim=ylim,
            )
            annotation.set_position((label_x, label_y))
            annotation.set_text(format_row_annotation(decision))
            annotation.set_color(_decision_color(decision))
            annotation.set_ha(ha)
            annotation.set_va(va)
            annotation.set_visible(True)

        for annotation in self.annotation_artists[len(frame.row_decisions) :]:
            annotation.set_visible(False)


@dataclass(slots=True)
class CurrentFramePreviewState:
    result: SimulationResult
    figure: object
    ax: object
    cells: list[PrescriptionCell]
    cell_patches: list[Rectangle]
    cell_labels: list[object]
    center_artist: object
    span_artist: object
    kan_artist: object
    mlp_artist: object
    out_artist: object
    annotation_artists: list[object]
    rendered_frame_index: int = 0
    detailed: bool = True

    def update_frame(self, frame_index: int, *, detailed: bool) -> None:
        index = _clamp_frame_index(self.result, frame_index)
        frame = self.result.frames[index]
        xlim, ylim = _current_frame_window(self.result, frame)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_title(f"当前时刻地图细节图 | 第 {frame.pass_id} 趟 | {frame.timestamp_ms} ms", fontsize=13)

        for cell, patch, label in zip(self.cells, self.cell_patches, self.cell_labels):
            visible = _cell_intersects_window(cell, xlim, ylim)
            patch.set_visible(visible)
            label.set_visible(visible)

        valid_rows, kan_points, mlp_points, out_points = _group_row_points(frame)
        if valid_rows:
            self.span_artist.set_data(
                [decision.x_m for decision in valid_rows],
                [decision.y_m for decision in valid_rows],
            )
        else:
            self.span_artist.set_data([], [])

        self.center_artist.set_offsets(_points_array([(frame.machine_center_x_m, frame.machine_center_y_m)]))
        self.kan_artist.set_offsets(_points_array(kan_points))
        self.mlp_artist.set_offsets(_points_array(mlp_points))
        self.out_artist.set_offsets(_points_array(out_points))
        self._update_annotations(frame, xlim, ylim, detailed=detailed)

        self.rendered_frame_index = index
        self.detailed = detailed

    def _update_annotations(
        self,
        frame: SimulationFrame,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        *,
        detailed: bool,
    ) -> None:
        if not detailed:
            for annotation in self.annotation_artists:
                annotation.set_visible(False)
            return

        for annotation, decision in zip(self.annotation_artists, frame.row_decisions):
            label_x, label_y, ha, va = _annotation_position(
                decision,
                row_spacing_m=self.result.machine_config.row_spacing_m,
                xlim=xlim,
                ylim=ylim,
            )
            color = _decision_color(decision)
            annotation.set_position((label_x, label_y))
            annotation.set_text(format_row_annotation(decision))
            annotation.set_color(color)
            annotation.set_ha(ha)
            annotation.set_va(va)
            annotation.set_visible(True)

        for annotation in self.annotation_artists[len(frame.row_decisions) :]:
            annotation.set_visible(False)


def _cells_bounds(cells: list[PrescriptionCell]) -> tuple[float, float, float, float]:
    min_x = min(cell.left for cell in cells)
    max_x = max(cell.right for cell in cells)
    min_y = min(cell.bottom for cell in cells)
    max_y = max(cell.top for cell in cells)
    return min_x, max_x, min_y, max_y


def _norm_from_cells(cells: list[PrescriptionCell]) -> mcolors.Normalize:
    rates = [cell.target_rate_kg_ha for cell in cells]
    return mcolors.Normalize(vmin=min(rates), vmax=max(rates))


def _prescription_extent(cells: list[PrescriptionCell]) -> tuple[tuple[float, float], tuple[float, float], mcolors.Normalize]:
    norm = _norm_from_cells(cells)
    min_x, max_x, min_y, max_y = _cells_bounds(cells)
    margin_x = max((max_x - min_x) * 0.04, 0.5)
    margin_y = max((max_y - min_y) * 0.06, 0.5)
    return (min_x - margin_x, max_x + margin_x), (min_y - margin_y, max_y + margin_y), norm


def _clamp_frame_index(result: SimulationResult, frame_index: int) -> int:
    if not result.frames:
        return 0
    return max(0, min(frame_index, len(result.frames) - 1))


def _points_array(points: list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return EMPTY_OFFSETS.copy()
    return np.asarray(points, dtype=float)


def _group_row_points(
    frame: SimulationFrame,
) -> tuple[list, list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    valid_rows = [decision for decision in sorted(frame.row_decisions, key=lambda item: item.row_index) if decision.status != "out_of_field"]
    kan_points: list[tuple[float, float]] = []
    mlp_points: list[tuple[float, float]] = []
    out_points: list[tuple[float, float]] = []

    for decision in frame.row_decisions:
        point = (decision.x_m, decision.y_m)
        if decision.status == "out_of_field":
            out_points.append(point)
        elif decision.selected_model == "inverse_KAN":
            kan_points.append(point)
        else:
            mlp_points.append(point)

    return valid_rows, kan_points, mlp_points, out_points


def _decision_color(decision) -> str:
    if decision.status == "out_of_field":
        return OUT_COLOR
    if decision.selected_model == "inverse_KAN":
        return KAN_COLOR
    return MLP_COLOR


def _current_frame_window(
    result: SimulationResult,
    frame: SimulationFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    row_xs = [item.x_m for item in frame.row_decisions]
    row_ys = [item.y_m for item in frame.row_decisions]
    x_margin = max(result.machine_config.row_spacing_m * 4, 1.2)
    y_margin = max(result.machine_config.row_spacing_m * 2, 1.2)
    return (min(row_xs) - x_margin, max(row_xs) + x_margin), (min(row_ys) - y_margin, max(row_ys) + y_margin)


def _cell_intersects_window(
    cell: PrescriptionCell,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> bool:
    return not (
        cell.right < xlim[0]
        or cell.left > xlim[1]
        or cell.top < ylim[0]
        or cell.bottom > ylim[1]
    )


def _draw_cells(ax, cells: list[PrescriptionCell], norm: mcolors.Normalize, show_labels: bool = True) -> None:
    for cell in cells:
        color = MAP_CMAP(norm(cell.target_rate_kg_ha))
        rect = Rectangle(
            (cell.left, cell.bottom),
            cell.width_m,
            cell.height_m,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.96,
        )
        ax.add_patch(rect)
        if show_labels:
            ax.text(
                cell.center_x_m,
                cell.center_y_m,
                f"{cell.zone_id}\n{cell.target_rate_kg_ha:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#0f172a",
            )


def _draw_cells_with_handles(ax, cells: list[PrescriptionCell], norm: mcolors.Normalize) -> tuple[list[Rectangle], list[object]]:
    patches: list[Rectangle] = []
    labels: list[object] = []
    for cell in cells:
        color = MAP_CMAP(norm(cell.target_rate_kg_ha))
        patch = Rectangle(
            (cell.left, cell.bottom),
            cell.width_m,
            cell.height_m,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.96,
        )
        ax.add_patch(patch)
        label = ax.text(
            cell.center_x_m,
            cell.center_y_m,
            f"{cell.zone_id}\n{cell.target_rate_kg_ha:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#0f172a",
        )
        patches.append(patch)
        labels.append(label)
    return patches, labels


def _create_annotation_artists(ax, count: int, *, font_size: float) -> list[object]:
    return [
        ax.text(
            0.0,
            0.0,
            "",
            fontsize=font_size,
            color="#0f172a",
            ha="left",
            va="bottom",
            visible=False,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f8fafc", "edgecolor": "none", "alpha": 0.9},
            zorder=7,
        )
        for _ in range(count)
    ]


def _draw_trajectories(ax, result: SimulationResult) -> None:
    pass_points: dict[int, list[tuple[float, float]]] = {}
    for frame in result.frames:
        pass_points.setdefault(frame.pass_id, []).append((frame.machine_center_x_m, frame.machine_center_y_m))
    for pass_id, points in pass_points.items():
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.plot(xs, ys, color=PATH_COLOR, linewidth=1.4, alpha=0.35)
        ax.scatter(xs[0], ys[0], color=PATH_COLOR, marker=">", s=22, alpha=0.7)
        ax.text(xs[0], ys[0], f"P{pass_id}", fontsize=8, color=PATH_COLOR, ha="left", va="bottom")


def _annotation_position(
    decision,
    *,
    row_spacing_m: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float, str, str]:
    horizontal_gap = max(row_spacing_m * 0.35, 0.14)
    vertical_gap = max(row_spacing_m * 0.18, 0.10)

    if decision.x_m <= xlim[0] + horizontal_gap * 1.5:
        label_x = decision.x_m + horizontal_gap
        ha = "left"
    elif decision.x_m >= xlim[1] - horizontal_gap * 1.5:
        label_x = decision.x_m - horizontal_gap
        ha = "right"
    elif decision.row_index % 2 == 0:
        label_x = decision.x_m + horizontal_gap
        ha = "left"
    else:
        label_x = decision.x_m - horizontal_gap
        ha = "right"

    if decision.y_m <= ylim[0] + vertical_gap * 1.6:
        label_y = decision.y_m + vertical_gap
        va = "bottom"
    elif decision.y_m >= ylim[1] - vertical_gap * 1.6:
        label_y = decision.y_m - vertical_gap
        va = "top"
    elif decision.row_index % 2 == 0:
        label_y = decision.y_m + vertical_gap
        va = "bottom"
    else:
        label_y = decision.y_m - vertical_gap
        va = "top"

    return label_x, label_y, ha, va


def _overview_annotation_position(
    decision,
    *,
    row_spacing_m: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float, str, str]:
    x_span = max(xlim[1] - xlim[0], row_spacing_m * 6, 1.0)
    y_span = max(ylim[1] - ylim[0], row_spacing_m * 4, 1.0)
    horizontal_gap = max(row_spacing_m * 0.32, x_span * 0.018, 0.18)
    vertical_gap = max(row_spacing_m * 0.16, y_span * 0.028, 0.12)

    if decision.x_m <= xlim[0] + horizontal_gap * 1.5:
        label_x = decision.x_m + horizontal_gap
        ha = "left"
    elif decision.x_m >= xlim[1] - horizontal_gap * 1.5:
        label_x = decision.x_m - horizontal_gap
        ha = "right"
    elif decision.row_index % 2 == 0:
        label_x = decision.x_m + horizontal_gap
        ha = "left"
    else:
        label_x = decision.x_m - horizontal_gap
        ha = "right"

    if decision.y_m <= ylim[0] + vertical_gap * 1.8:
        label_y = decision.y_m + vertical_gap
        va = "bottom"
    elif decision.y_m >= ylim[1] - vertical_gap * 1.8:
        label_y = decision.y_m - vertical_gap
        va = "top"
    elif decision.row_index % 2 == 0:
        label_y = decision.y_m + vertical_gap
        va = "bottom"
    else:
        label_y = decision.y_m - vertical_gap
        va = "top"

    return label_x, label_y, ha, va


def _set_axes_style(ax, xlim: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X 坐标 / m")
    ax.set_ylabel("Y 坐标 / m")
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def _create_preview_figure_layout(
    *,
    figsize: tuple[float, float],
    dpi: int,
    top: float,
    bottom: float,
) -> tuple[object, object, object, object]:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.22],
        left=0.08,
        right=0.96,
        top=top,
        bottom=bottom,
        wspace=0.14,
    )
    ax = fig.add_subplot(outer[0, 0])
    sidebar = outer[0, 1].subgridspec(2, 1, height_ratios=[0.34, 0.66], hspace=0.10)
    legend_ax = fig.add_subplot(sidebar[0, 0])
    colorbar_ax = fig.add_subplot(sidebar[1, 0])
    legend_ax.axis("off")
    return fig, ax, legend_ax, colorbar_ax


def _add_colorbar(fig, cax, norm: mcolors.Normalize) -> None:
    sm = cm.ScalarMappable(norm=norm, cmap=MAP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.tick_right()
    cbar.set_label("目标施肥量 (kg/ha)")


def _add_marker_legend(legend_ax) -> None:
    handles = [
        Line2D([0], [0], color=PATH_COLOR, lw=1.5, alpha=0.5, label="机器轨迹"),
        Line2D([0], [0], color=SPAN_COLOR, lw=2.2, label="当前机器跨排行线"),
        Line2D([0], [0], marker="*", color=SPAN_COLOR, linestyle="", markersize=10, label="机器中心"),
        Line2D([0], [0], marker="o", color=KAN_COLOR, linestyle="", markersize=7, label="KAN 决策点"),
        Line2D([0], [0], marker="s", color=MLP_COLOR, linestyle="", markersize=7, label="MLP 外推点"),
        Line2D([0], [0], marker="x", color=OUT_COLOR, linestyle="", markersize=7, label="地块外排点"),
    ]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8, borderaxespad=0.0)


def create_overview_preview_state(result: SimulationResult, frame_index: int = 0) -> OverviewPreviewState:
    cells = list(result.prescription_cells)
    xlim, ylim, norm = _prescription_extent(cells)

    fig, ax, legend_ax, colorbar_ax = _create_preview_figure_layout(
        figsize=(11.5, 8.2),
        dpi=160,
        top=0.92,
        bottom=0.10,
    )
    _draw_cells(ax, cells, norm, show_labels=len(cells) <= 30)
    _draw_trajectories(ax, result)
    span_artist, = ax.plot([], [], color=SPAN_COLOR, linewidth=2.2, alpha=0.85, zorder=4)
    center_artist = ax.scatter([], [], color=SPAN_COLOR, marker="*", s=120, zorder=5)
    kan_artist = ax.scatter([], [], color=KAN_COLOR, marker="o", s=36, zorder=6)
    mlp_artist = ax.scatter([], [], color=MLP_COLOR, marker="s", s=36, zorder=6)
    out_artist = ax.scatter([], [], color=OUT_COLOR, marker="x", s=28, zorder=5)
    max_row_count = max((len(frame.row_decisions) for frame in result.frames), default=0)
    annotation_artists = _create_annotation_artists(ax, max_row_count, font_size=6.5)
    _set_axes_style(ax, xlim, ylim, "变量施肥作业地图总览")
    _add_colorbar(fig, colorbar_ax, norm)
    _add_marker_legend(legend_ax)

    state = OverviewPreviewState(
        result=result,
        figure=fig,
        ax=ax,
        center_artist=center_artist,
        span_artist=span_artist,
        kan_artist=kan_artist,
        mlp_artist=mlp_artist,
        out_artist=out_artist,
        annotation_artists=annotation_artists,
    )
    state.update_frame(frame_index)
    return state


def create_current_preview_state(
    result: SimulationResult,
    frame_index: int = 0,
    *,
    detailed: bool,
) -> CurrentFramePreviewState:
    cells = list(result.prescription_cells)
    norm = _norm_from_cells(cells)

    fig, ax, legend_ax, colorbar_ax = _create_preview_figure_layout(
        figsize=(10.5, 7.2),
        dpi=170,
        top=0.90,
        bottom=0.10,
    )
    cell_patches, cell_labels = _draw_cells_with_handles(ax, cells, norm)
    _draw_trajectories(ax, result)
    span_artist, = ax.plot([], [], color=SPAN_COLOR, linewidth=2.2, alpha=0.85, zorder=4)
    center_artist = ax.scatter([], [], color=SPAN_COLOR, marker="*", s=120, zorder=5)
    kan_artist = ax.scatter([], [], color=KAN_COLOR, marker="o", s=36, zorder=6)
    mlp_artist = ax.scatter([], [], color=MLP_COLOR, marker="s", s=36, zorder=6)
    out_artist = ax.scatter([], [], color=OUT_COLOR, marker="x", s=28, zorder=5)

    max_row_count = max((len(frame.row_decisions) for frame in result.frames), default=0)
    annotation_artists = _create_annotation_artists(ax, max_row_count, font_size=7)

    _set_axes_style(ax, (0.0, 1.0), (0.0, 1.0), "当前时刻地图细节图")
    _add_colorbar(fig, colorbar_ax, norm)
    _add_marker_legend(legend_ax)

    state = CurrentFramePreviewState(
        result=result,
        figure=fig,
        ax=ax,
        cells=cells,
        cell_patches=cell_patches,
        cell_labels=cell_labels,
        center_artist=center_artist,
        span_artist=span_artist,
        kan_artist=kan_artist,
        mlp_artist=mlp_artist,
        out_artist=out_artist,
        annotation_artists=annotation_artists,
    )
    state.update_frame(frame_index, detailed=detailed)
    return state


def create_overview_figure(result: SimulationResult, frame_index: int = 0):
    return create_overview_preview_state(result, frame_index=frame_index).figure


def create_current_frame_figure(result: SimulationResult, frame_index: int = 0):
    return create_current_preview_state(result, frame_index=frame_index, detailed=True).figure


def create_legend_figure(result: SimulationResult):
    cells = list(result.prescription_cells)
    norm = _norm_from_cells(cells)

    fig = plt.figure(figsize=(8.8, 5.0), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])

    sm = cm.ScalarMappable(norm=norm, cmap=MAP_CMAP)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_bar)
    ax_bar.set_title("目标施肥量\nkg/ha", fontsize=11)

    ax_legend.axis("off")
    handles = [
        Line2D([0], [0], color=PATH_COLOR, lw=1.5, alpha=0.5, label="机器历史轨迹"),
        Line2D([0], [0], color=SPAN_COLOR, lw=2.2, label="当前机器跨排行线"),
        Line2D([0], [0], marker="*", color=SPAN_COLOR, linestyle="", markersize=10, label="机器中心"),
        Line2D([0], [0], marker="o", color=KAN_COLOR, linestyle="", markersize=7, label="inverse_KAN 域内决策"),
        Line2D([0], [0], marker="s", color=MLP_COLOR, linestyle="", markersize=7, label="inverse_MLP 外推决策"),
        Line2D([0], [0], marker="x", color=OUT_COLOR, linestyle="", markersize=7, label="地块外单排位置"),
    ]
    ax_legend.legend(handles=handles, loc="center left", frameon=False, fontsize=10)
    ax_legend.text(
        0.0,
        0.08,
        "色块表示目标施肥量分布，颜色越深表示处方量越高。",
        transform=ax_legend.transAxes,
        fontsize=10,
        color="#0f172a",
    )
    fig.suptitle("变量施肥地图图例", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.10, wspace=0.25)
    return fig


def create_prescription_overview_figure(prescription: PrescriptionMap):
    cells = list(prescription.cells)
    xlim, ylim, norm = _prescription_extent(cells)

    fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=160)
    _draw_cells(ax, cells, norm, show_labels=len(cells) <= 30)
    _set_axes_style(ax, xlim, ylim, "处方图总览")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.10)
    cax = fig.add_axes([0.90, 0.10, 0.03, 0.82])
    _add_colorbar(fig, cax, norm)
    return fig


def create_prescription_legend_figure(prescription: PrescriptionMap):
    cells = list(prescription.cells)
    norm = _norm_from_cells(cells)

    fig = plt.figure(figsize=(8.8, 5.0), dpi=170)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])

    sm = cm.ScalarMappable(norm=norm, cmap=MAP_CMAP)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_bar)
    ax_bar.set_title("目标施肥量\nkg/ha", fontsize=11)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.78,
        "色块表示处方图上的目标施肥量分布。",
        transform=ax_text.transAxes,
        fontsize=10,
        color="#0f172a",
    )
    ax_text.text(
        0.0,
        0.52,
        "加载处方图后，可以先在总览图中查看分区和目标量。",
        transform=ax_text.transAxes,
        fontsize=10,
        color="#334155",
    )
    ax_text.text(
        0.0,
        0.26,
        "运行仿真后，这里会切换为包含机器轨迹、机器中心和单排决策标记的完整图例。",
        transform=ax_text.transAxes,
        fontsize=10,
        color="#334155",
    )
    fig.suptitle("处方图图例", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.10, wspace=0.25)
    return fig


def export_visual_assets(result: SimulationResult, output_dir: str | Path, frame_index: int = 0) -> dict[str, Path]:
    target_dir = Path(output_dir)
    overview_path = target_dir / "map_overview.png"
    current_frame_path = target_dir / "map_current_frame.png"
    legend_path = target_dir / "map_legend.png"

    overview_figure = create_overview_figure(result, frame_index=frame_index)
    overview_figure.savefig(overview_path, bbox_inches="tight")
    plt.close(overview_figure)

    current_frame_figure = create_current_frame_figure(result, frame_index=frame_index)
    current_frame_figure.savefig(current_frame_path, bbox_inches="tight")
    plt.close(current_frame_figure)

    legend_figure = create_legend_figure(result)
    legend_figure.savefig(legend_path, bbox_inches="tight")
    plt.close(legend_figure)

    return {
        "map_overview_png": overview_path,
        "map_current_frame_png": current_frame_path,
        "map_legend_png": legend_path,
    }
