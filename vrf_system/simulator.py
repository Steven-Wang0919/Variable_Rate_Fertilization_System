from __future__ import annotations

from collections import Counter

from .domain import MachineConfig, SimulationFrame, SimulationResult
from .engine import ModelRouter, build_row_decision, target_mass_from_rate
from .prescription import PrescriptionMap
from .routing_spec import EXPERIMENTAL_EXTRAPOLATION, IN_FIELD, OUT_OF_FIELD


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
            raise ValueError("机器作业幅宽必须大于 0。")

        # Align the first active row to the first centerline inside the field
        # instead of placing the outermost row directly on the boundary.
        center_y = prescription_map.bounds.min_y + float(machine_config.row_spacing_m) / 2.0 - min_offset
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
                                row_state=OUT_OF_FIELD,
                                opening_mm=0.0,
                                raw_speed_r_min=0.0,
                                speed_r_min_cmd=0.0,
                                model_route="",
                                mass_state="",
                                route_mode="",
                                speed_clip_state="",
                                confidence_level="",
                            )
                        )
                        continue

                    target_mass = target_mass_from_rate(
                        cell.target_rate_kg_ha,
                        machine_config.row_spacing_m,
                        machine_config.travel_speed_kmh,
                    )
                    strategy_opening = self.router.select_strategy_opening(target_mass)
                    (
                        raw_speed_r_min,
                        speed_r_min_cmd,
                        model_route,
                        mass_state,
                        route_mode,
                        speed_clip_state,
                        confidence_level,
                    ) = self.router.predict(target_mass, strategy_opening)
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
                            row_state=IN_FIELD,
                            opening_mm=strategy_opening,
                            raw_speed_r_min=raw_speed_r_min,
                            speed_r_min_cmd=speed_r_min_cmd,
                            model_route=model_route,
                            mass_state=mass_state,
                            route_mode=route_mode,
                            speed_clip_state=speed_clip_state,
                            confidence_level=confidence_level,
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
            prescription_cells=list(prescription_map.cells),
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
        in_field_decisions = [decision for decision in decisions if decision.row_state == IN_FIELD]
        row_state_counts = Counter(decision.row_state for decision in decisions)
        model_route_counts = Counter(decision.model_route for decision in in_field_decisions)
        mass_state_counts = Counter(decision.mass_state for decision in in_field_decisions)
        route_mode_counts = Counter(decision.route_mode for decision in in_field_decisions)
        confidence_level_counts = Counter(decision.confidence_level for decision in in_field_decisions)
        avg_rate = (
            sum(item.target_rate_kg_ha for item in in_field_decisions) / len(in_field_decisions)
            if in_field_decisions
            else 0.0
        )
        avg_speed_cmd = (
            sum(item.speed_r_min_cmd for item in in_field_decisions) / len(in_field_decisions)
            if in_field_decisions
            else 0.0
        )
        avg_raw_speed = (
            sum(item.raw_speed_r_min for item in in_field_decisions) / len(in_field_decisions)
            if in_field_decisions
            else 0.0
        )
        extrapolation_count = sum(
            1 for item in in_field_decisions if item.route_mode == EXPERIMENTAL_EXTRAPOLATION
        )
        return {
            "frame_count": len(frames),
            "pass_count": len({frame.pass_id for frame in frames}),
            "total_row_decisions": len(decisions),
            "machine_config": machine_config.to_dict(),
            "prescription_bounds": prescription_map.bounds.to_dict(),
            "row_state_counts": dict(row_state_counts),
            "model_route_counts": dict(model_route_counts),
            "mass_state_counts": dict(mass_state_counts),
            "route_mode_counts": dict(route_mode_counts),
            "speed_clip_state_counts": {},
            "confidence_level_counts": dict(confidence_level_counts),
            "average_target_rate_kg_ha": round(avg_rate, 4),
            "average_speed_r_min_cmd": round(avg_speed_cmd, 4),
            "average_raw_speed_r_min": round(avg_raw_speed, 4),
            "experimental_extrapolation_count": extrapolation_count,
        }
