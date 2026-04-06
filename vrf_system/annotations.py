from __future__ import annotations

from .domain import RowDecision
from .routing_spec import OUT_OF_FIELD

OUT_OF_FIELD_TEXT = "\u5730\u5757\u5916"


def format_row_annotation(decision: RowDecision) -> str:
    if decision.row_state == OUT_OF_FIELD:
        return f"R{decision.row_index} | {OUT_OF_FIELD_TEXT}"
    return f"R{decision.row_index} | {decision.opening_mm:.1f}mm\n{decision.speed_r_min_cmd:.1f}r/min"
