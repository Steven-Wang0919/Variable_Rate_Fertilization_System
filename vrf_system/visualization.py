from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .annotations import format_row_annotation
from .domain import ExportedArtifacts, PrescriptionCell, SimulationResult


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MAP_CMAP = cm.get_cmap("YlGnBu")
KAN_COLOR = "#059669"
MLP_COLOR = "#d97706"
OUT_COLOR = "#6b7280"
PATH_COLOR = "#2563eb"
SPAN_COLOR = "#111827"


def _cells_bounds(cells: list[PrescriptionCell]) -> tuple[float, float, float, float]:
    min_x = min(cell.left for cell in cells)
    max_x = max(cell.right for cell in cells)
    min_y = min(cell.bottom for cell in cells)
    max_y = max(cell.top for cell in cells)
    return min_x, max_x, min_y, max_y


def _norm_from_cells(cells: list[PrescriptionCell]) -> mcolors.Normalize:
    rates = [cell.target_rate_kg_ha for cell in cells]
    return mcolors.Normalize(vmin=min(rates), vmax=max(rates))


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


def _draw_current_frame(
    ax,
    result: SimulationResult,
    frame_index: int,
    *,
    annotate_rows: bool = False,
    annotation_xlim: tuple[float, float] | None = None,
    annotation_ylim: tuple[float, float] | None = None,
) -> None:
    frame = result.frames[frame_index]
    valid_rows = [item for item in frame.row_decisions if item.status != "out_of_field"]
    if valid_rows:
        ordered_rows = sorted(valid_rows, key=lambda item: item.row_index)
        span_x = [item.x_m for item in ordered_rows]
        span_y = [item.y_m for item in ordered_rows]
        ax.plot(span_x, span_y, color=SPAN_COLOR, linewidth=2.2, alpha=0.85, zorder=4)

    ax.scatter(
        [frame.machine_center_x_m],
        [frame.machine_center_y_m],
        color=SPAN_COLOR,
        marker="*",
        s=120,
        zorder=5,
    )

    for decision in frame.row_decisions:
        if decision.status == "out_of_field":
            ax.scatter(decision.x_m, decision.y_m, color=OUT_COLOR, marker="x", s=28, zorder=5)
            color = OUT_COLOR
        else:
            if decision.selected_model == "inverse_KAN":
                marker = "o"
                color = KAN_COLOR
            else:
                marker = "s"
                color = MLP_COLOR
            ax.scatter(decision.x_m, decision.y_m, color=color, marker=marker, s=36, zorder=6)

        if annotate_rows and annotation_xlim is not None and annotation_ylim is not None:
            label_x, label_y, ha, va = _annotation_position(
                decision,
                row_spacing_m=result.machine_config.row_spacing_m,
                xlim=annotation_xlim,
                ylim=annotation_ylim,
            )
            ax.text(
                label_x,
                label_y,
                format_row_annotation(decision),
                fontsize=7,
                color=color,
                ha=ha,
                va=va,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f8fafc", "edgecolor": "none", "alpha": 0.9},
                zorder=7,
            )
        elif decision.status != "out_of_field":
            ax.text(
                decision.x_m,
                decision.y_m + 0.08,
                f"R{decision.row_index}",
                fontsize=7,
                color=color,
                ha="center",
                va="bottom",
            )


def _set_axes_style(ax, xlim: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X 坐标 / m")
    ax.set_ylabel("Y 坐标 / m")
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def _add_colorbar(fig, ax, norm: mcolors.Normalize) -> None:
    sm = cm.ScalarMappable(norm=norm, cmap=MAP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("目标施肥量 (kg/ha)")


def _add_marker_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=PATH_COLOR, lw=1.5, alpha=0.5, label="机具轨迹"),
        Line2D([0], [0], color=SPAN_COLOR, lw=2.2, label="当前机具跨排连线"),
        Line2D([0], [0], marker="*", color=SPAN_COLOR, linestyle="", markersize=10, label="机具中心"),
        Line2D([0], [0], marker="o", color=KAN_COLOR, linestyle="", markersize=7, label="KAN 决策点"),
        Line2D([0], [0], marker="s", color=MLP_COLOR, linestyle="", markersize=7, label="MLP 外推点"),
        Line2D([0], [0], marker="x", color=OUT_COLOR, linestyle="", markersize=7, label="地块外排点"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8)


def create_overview_figure(result: SimulationResult, frame_index: int = 0):
    cells = list(result.prescription_cells)
    norm = _norm_from_cells(cells)
    min_x, max_x, min_y, max_y = _cells_bounds(cells)
    margin_x = max((max_x - min_x) * 0.04, 0.5)
    margin_y = max((max_y - min_y) * 0.06, 0.5)

    fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=160)
    _draw_cells(ax, cells, norm, show_labels=len(cells) <= 30)
    _draw_trajectories(ax, result)
    if result.frames:
        _draw_current_frame(ax, result, max(0, min(frame_index, len(result.frames) - 1)), annotate_rows=False)
    _set_axes_style(
        ax,
        (min_x - margin_x, max_x + margin_x),
        (min_y - margin_y, max_y + margin_y),
        "变量施肥作业地图总览",
    )
    _add_colorbar(fig, ax, norm)
    _add_marker_legend(ax)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.10)
    return fig


def create_current_frame_figure(result: SimulationResult, frame_index: int = 0):
    cells = list(result.prescription_cells)
    norm = _norm_from_cells(cells)
    frame = result.frames[max(0, min(frame_index, len(result.frames) - 1))]
    row_xs = [item.x_m for item in frame.row_decisions]
    row_ys = [item.y_m for item in frame.row_decisions]
    x_margin = max(result.machine_config.row_spacing_m * 4, 1.2)
    y_margin = max(result.machine_config.row_spacing_m * 2, 1.2)
    xlim = (min(row_xs) - x_margin, max(row_xs) + x_margin)
    ylim = (min(row_ys) - y_margin, max(row_ys) + y_margin)
    visible_cells = [
        cell
        for cell in cells
        if not (cell.right < xlim[0] or cell.left > xlim[1] or cell.top < ylim[0] or cell.bottom > ylim[1])
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=170)
    _draw_cells(ax, visible_cells, norm, show_labels=True)
    _draw_trajectories(ax, result)
    _draw_current_frame(
        ax,
        result,
        frame_index,
        annotate_rows=True,
        annotation_xlim=xlim,
        annotation_ylim=ylim,
    )
    _set_axes_style(
        ax,
        xlim,
        ylim,
        f"当前时刻地图细节图  |  第 {frame.pass_id} 趟  |  {frame.timestamp_ms} ms",
    )
    _add_colorbar(fig, ax, norm)
    _add_marker_legend(ax)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.10)
    return fig


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
        Line2D([0], [0], color=PATH_COLOR, lw=1.5, alpha=0.5, label="机具历史轨迹"),
        Line2D([0], [0], color=SPAN_COLOR, lw=2.2, label="当前机具跨排连线"),
        Line2D([0], [0], marker="*", color=SPAN_COLOR, linestyle="", markersize=10, label="机具中心"),
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
