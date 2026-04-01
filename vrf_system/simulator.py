from __future__ import annotations

from collections import Counter

from .domain import MachineConfig, SimulationFrame, SimulationResult
from .engine import ModelRouter, build_row_decision, target_mass_from_rate
from .prescription import PrescriptionMap


class FieldSimulator:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def run(self, prescription_map: PrescriptionMap, machine_config: MachineConfig) -> SimulationResult:
        machine_config.validate()
        row_offsets = machine_config.resolved_row_offsets()
        min_offset = min(row_offsets)
        max_offset = max(row_offsets)
        pass_spacing = float(machine_config.row_count) * float(machine_config.row_spacing_m)
        if pass_spacing <= 0:
            raise ValueError("机具作业幅宽必须大于 0。")

        center_y = prescription_map.bounds.min_y - min_offset
        pass_centers: list[float] = []
        while True:
            pass_centers.append(center_y)
            if center_y + max_offset >= prescription_map.bounds.max_y - 1e-9:
                break
            center_y += pass_spacing

        speed_m_s = float(machine_config.travel_speed_kmh) * 1000.0 / 3600.0
        sample_distance = max(speed_m_s * (float(machine_config.sample_period_ms) / 1000.0), 0.01)

        frames: list[SimulationFrame] = []
        timestamp_ms = 0
        x_min = prescription_map.bounds.min_x
        x_max = prescription_map.bounds.max_x
        longitudinal_offset = float(machine_config.machine_center_to_row_origin_m)

        for pass_idx, pass_center_y in enumerate(pass_centers, start=1):
            direction_sign = 1 if pass_idx % 2 == 1 else -1
            application_positions = self._build_axis_positions(
                x_min if direction_sign == 1 else x_max,
                x_max if direction_sign == 1 else x_min,
                sample_distance,
            )
            for application_x in application_positions:
                machine_center_x = application_x - direction_sign * longitudinal_offset
                row_decisions = []
                for row_index, row_offset in enumerate(row_offsets, start=1):
                    row_x = application_x
                    row_y = pass_center_y + row_offset
                    cell = prescription_map.find_cell(row_x, row_y)
                    if cell is None:
                        row_decisions.append(
                            build_row_decision(
                                timestamp_ms=timestamp_ms,
                                pass_id=pass_idx,
                                row_index=row_index,
                                x_m=row_x,
                                y_m=row_y,
                                zone_id="",
                                target_rate_kg_ha=0.0,
                                target_mass_g_min=0.0,
                                strategy_opening_mm=0.0,
                                target_speed_r_min=0.0,
                                selected_model="none",
                                domain_status="out_of_field",
                                status="out_of_field",
                            )
                        )
                        continue

                    target_mass = target_mass_from_rate(
                        cell.target_rate_kg_ha,
                        machine_config.row_spacing_m,
                        machine_config.travel_speed_kmh,
                    )
                    strategy_opening = self.router.select_strategy_opening(target_mass)
                    target_speed, selected_model, domain_status, status = self.router.predict(
                        target_mass,
                        strategy_opening,
                    )
                    row_decisions.append(
                        build_row_decision(
                            timestamp_ms=timestamp_ms,
                            pass_id=pass_idx,
                            row_index=row_index,
                            x_m=row_x,
                            y_m=row_y,
                            zone_id=cell.zone_id,
                            target_rate_kg_ha=cell.target_rate_kg_ha,
                            target_mass_g_min=target_mass,
                            strategy_opening_mm=strategy_opening,
                            target_speed_r_min=target_speed,
                            selected_model=selected_model,
                            domain_status=domain_status,
                            status=status,
                        )
                    )
                frames.append(
                    SimulationFrame(
                        timestamp_ms=timestamp_ms,
                        pass_id=pass_idx,
                        machine_center_x_m=machine_center_x,
                        machine_center_y_m=pass_center_y,
                        direction_sign=direction_sign,
                        row_decisions=row_decisions,
                    )
                )
                timestamp_ms += int(machine_config.sample_period_ms)

        summary = self._build_summary(frames, prescription_map, machine_config)
        return SimulationResult(
            frames=frames,
            machine_config=machine_config,
            prescription_path=prescription_map.source_path,
            summary=summary,
        )

    @staticmethod
    def _build_axis_positions(start: float, end: float, step: float) -> list[float]:
        direction = 1.0 if end >= start else -1.0
        values = [float(start)]
        current = float(start)
        while (direction > 0 and current + step < end) or (direction < 0 and current - step > end):
            current += direction * step
            values.append(float(current))
        if values[-1] != float(end):
            values.append(float(end))
        return values

    @staticmethod
    def _build_summary(
        frames: list[SimulationFrame],
        prescription_map: PrescriptionMap,
        machine_config: MachineConfig,
    ) -> dict[str, object]:
        decisions = [decision for frame in frames for decision in frame.row_decisions]
        status_counts = Counter(decision.status for decision in decisions)
        model_counts = Counter(decision.selected_model for decision in decisions if decision.selected_model != "none")
        domain_counts = Counter(decision.domain_status for decision in decisions)
        valid = [decision for decision in decisions if decision.status == "ok"]
        avg_rate = sum(item.target_rate_kg_ha for item in valid) / len(valid) if valid else 0.0
        avg_speed = sum(item.target_speed_r_min for item in valid) / len(valid) if valid else 0.0
        extrapolation_count = sum(
            1
            for item in decisions
            if item.selected_model == "inverse_MLP" and item.domain_status != "in_domain"
        )
        return {
            "frame_count": len(frames),
            "pass_count": len({frame.pass_id for frame in frames}),
            "total_row_decisions": len(decisions),
            "machine_config": machine_config.to_dict(),
            "prescription_bounds": prescription_map.bounds.to_dict(),
            "status_counts": dict(status_counts),
            "selected_model_counts": dict(model_counts),
            "domain_status_counts": dict(domain_counts),
            "average_target_rate_kg_ha": round(avg_rate, 4),
            "average_target_speed_r_min": round(avg_speed, 4),
            "extrapolation_count": extrapolation_count,
        }
