from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .domain import Bounds, PrescriptionCell


REQUIRED_COLUMNS = (
    "cell_id",
    "center_x_m",
    "center_y_m",
    "width_m",
    "height_m",
    "target_rate_kg_ha",
    "zone_id",
)


class PrescriptionValidationError(ValueError):
    """处方图校验异常。"""


@dataclass(slots=True)
class PrescriptionMap:
    cells: list[PrescriptionCell]
    bounds: Bounds
    source_path: Path

    @classmethod
    def from_csv(cls, path: str | Path) -> "PrescriptionMap":
        source_path = Path(path).resolve()
        if not source_path.exists():
            raise PrescriptionValidationError(f"处方图文件不存在：{source_path}")

        df = pd.read_csv(source_path)
        cls._validate_dataframe(df, source_path)

        cells = [
            PrescriptionCell(
                cell_id=str(row["cell_id"]),
                center_x_m=float(row["center_x_m"]),
                center_y_m=float(row["center_y_m"]),
                width_m=float(row["width_m"]),
                height_m=float(row["height_m"]),
                target_rate_kg_ha=float(row["target_rate_kg_ha"]),
                zone_id="" if pd.isna(row["zone_id"]) else str(row["zone_id"]),
            )
            for _, row in df.iterrows()
        ]
        min_x = min(cell.left for cell in cells)
        max_x = max(cell.right for cell in cells)
        min_y = min(cell.bottom for cell in cells)
        max_y = max(cell.top for cell in cells)

        ordered_cells = sorted(cells, key=lambda item: (item.center_y_m, item.center_x_m))
        return cls(
            cells=ordered_cells,
            bounds=Bounds(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y),
            source_path=source_path,
        )

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, source_path: Path) -> None:
        missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
        if missing:
            raise PrescriptionValidationError(
                f"处方图缺少必要字段：{', '.join(missing)}。文件：{source_path}"
            )
        if df.empty:
            raise PrescriptionValidationError("处方图为空，无法进行变量施肥决策。")
        if df["cell_id"].astype(str).duplicated().any():
            duplicated = df.loc[df["cell_id"].astype(str).duplicated(), "cell_id"].astype(str).tolist()
            raise PrescriptionValidationError(f"处方图存在重复 cell_id：{', '.join(duplicated[:5])}")
        if (df["width_m"] <= 0).any() or (df["height_m"] <= 0).any():
            raise PrescriptionValidationError("处方图中的 width_m 和 height_m 必须大于 0。")
        if (df["target_rate_kg_ha"] < 0).any():
            raise PrescriptionValidationError("处方图中的 target_rate_kg_ha 不能为负数。")

    def find_cell(self, x_m: float, y_m: float) -> PrescriptionCell | None:
        for cell in self.cells:
            if cell.contains(x_m, y_m):
                return cell
        return None

    def rate_range(self) -> tuple[float, float]:
        values = [cell.target_rate_kg_ha for cell in self.cells]
        return min(values), max(values)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([cell.to_record() for cell in self.cells])
