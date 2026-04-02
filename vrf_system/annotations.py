from __future__ import annotations

from .domain import RowDecision

OUT_OF_FIELD_TEXT = "\u5730\u5757\u5916"


def format_row_annotation(decision: RowDecision) -> str:
    if decision.status == "out_of_field":
        return f"R{decision.row_index} | {OUT_OF_FIELD_TEXT}"
    return f"R{decision.row_index} | {decision.strategy_opening_mm:.1f}mm\n{decision.target_speed_r_min:.1f}r/min"
