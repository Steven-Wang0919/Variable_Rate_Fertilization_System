from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STANDARD_OPENINGS_MM = (20.0, 35.0, 50.0)


@dataclass(slots=True)
class Bounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return float(self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return float(self.max_y - self.min_y)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class PrescriptionCell:
    cell_id: str
    center_x_m: float
    center_y_m: float
    width_m: float
    height_m: float
    target_rate_kg_ha: float
    zone_id: str

    @property
    def left(self) -> float:
        return self.center_x_m - self.width_m / 2.0

    @property
    def right(self) -> float:
        return self.center_x_m + self.width_m / 2.0

    @property
    def bottom(self) -> float:
        return self.center_y_m - self.height_m / 2.0

    @property
    def top(self) -> float:
        return self.center_y_m + self.height_m / 2.0

    def contains(self, x_m: float, y_m: float, eps: float = 1e-9) -> bool:
        return (
            self.left - eps <= float(x_m) <= self.right + eps
            and self.bottom - eps <= float(y_m) <= self.top + eps
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MachineConfig:
    row_count: int = 6
    row_spacing_m: float = 0.6
    travel_speed_kmh: float = 6.0
    sample_period_ms: int = 200
    machine_center_to_row_origin_m: float = 0.0
    row_offsets_m: list[float] = field(default_factory=list)

    def validate(self) -> None:
        if int(self.row_count) <= 0:
            raise ValueError("行数必须大于 0。")
        if float(self.row_spacing_m) <= 0:
            raise ValueError("行距必须大于 0。")
        if float(self.travel_speed_kmh) <= 0:
            raise ValueError("作业速度必须大于 0。")
        if int(self.sample_period_ms) <= 0:
            raise ValueError("采样周期必须大于 0。")
        if self.row_offsets_m and len(self.row_offsets_m) != int(self.row_count):
            raise ValueError("自定义排位偏移数量必须与行数一致。")

    def resolved_row_offsets(self) -> list[float]:
        self.validate()
        if self.row_offsets_m:
            return [float(v) for v in self.row_offsets_m]
        center = (int(self.row_count) - 1) / 2.0
        return [round((idx - center) * float(self.row_spacing_m), 6) for idx in range(int(self.row_count))]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row_offsets_m"] = self.resolved_row_offsets()
        return payload


@dataclass(slots=True)
class ModelBundleConfig:
    name: str
    model_type: str
    model_path: Path
    meta_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "meta_path": str(self.meta_path),
        }


@dataclass(slots=True)
class RowDecision:
    timestamp_ms: int
    pass_id: int
    row_index: int
    x_m: float
    y_m: float
    zone_id: str
    target_rate_kg_ha: float
    target_mass_g_min: float
    strategy_opening_mm: float
    target_speed_r_min: float
    selected_model: str
    domain_status: str
    status: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SimulationFrame:
    timestamp_ms: int
    pass_id: int
    machine_center_x_m: float
    machine_center_y_m: float
    direction_sign: int
    row_decisions: list[RowDecision]


@dataclass(slots=True)
class ExportedArtifacts:
    output_dir: Path
    row_command_timeline: Path
    model_routing_trace: Path
    simulation_summary: Path
    map_overview_png: Path
    map_current_frame_png: Path
    map_legend_png: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "row_command_timeline": str(self.row_command_timeline),
            "model_routing_trace": str(self.model_routing_trace),
            "simulation_summary": str(self.simulation_summary),
            "map_overview_png": str(self.map_overview_png),
            "map_current_frame_png": str(self.map_current_frame_png),
            "map_legend_png": str(self.map_legend_png),
        }


@dataclass(slots=True)
class SimulationResult:
    frames: list[SimulationFrame]
    machine_config: MachineConfig
    prescription_path: Path
    prescription_cells: list[PrescriptionCell]
    summary: dict[str, Any]

    def flatten_decisions(self) -> list[RowDecision]:
        decisions: list[RowDecision] = []
        for frame in self.frames:
            decisions.extend(frame.row_decisions)
        return decisions
