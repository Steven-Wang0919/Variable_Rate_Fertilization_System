from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .defaults import DEFAULT_OUTPUT_ROOT
from .domain import ExportedArtifacts, SimulationResult


def export_simulation_result(
    result: SimulationResult,
    output_root: str | Path | None = None,
) -> ExportedArtifacts:
    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_dir = Path(root).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions = [decision.to_record() for decision in result.flatten_decisions()]
    decision_df = pd.DataFrame(decisions)
    row_command_timeline = output_dir / "row_command_timeline.csv"
    routing_trace = output_dir / "model_routing_trace.csv"
    summary_path = output_dir / "simulation_summary.json"

    decision_df.to_csv(row_command_timeline, index=False, encoding="utf-8-sig")
    decision_df[
        [
            "timestamp_ms",
            "pass_id",
            "row_index",
            "x_m",
            "y_m",
            "selected_model",
            "domain_status",
            "strategy_opening_mm",
            "target_speed_r_min",
            "status",
        ]
    ].to_csv(routing_trace, index=False, encoding="utf-8-sig")

    summary_payload = dict(result.summary)
    summary_payload["prescription_path"] = str(result.prescription_path)
    summary_payload["exported_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return ExportedArtifacts(
        output_dir=output_dir,
        row_command_timeline=row_command_timeline,
        model_routing_trace=routing_trace,
        simulation_summary=summary_path,
    )
