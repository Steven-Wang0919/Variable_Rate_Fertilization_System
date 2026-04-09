from __future__ import annotations

from dataclasses import dataclass

from .domain import RowDecision
from .model_runtime import ModelBundle
from .routing_spec import IN_FIELD, INVERSE_KAN, inverse_route_for, select_strategy_opening


def target_mass_from_rate(rate_kg_ha: float, row_spacing_m: float, travel_speed_kmh: float) -> float:
    return float(rate_kg_ha) * float(row_spacing_m) * float(travel_speed_kmh) * 1.6666667


@dataclass(slots=True)
class ModelRouter:
    kan_bundle: ModelBundle
    mlp_bundle: ModelBundle

    def select_strategy_opening(self, target_mass_g_min: float) -> float:
        return select_strategy_opening(target_mass_g_min)

    def route(self, target_mass_g_min: float, opening_mm: float) -> tuple[ModelBundle, str, str, str]:
        route = inverse_route_for(target_mass_g_min, opening_mm)
        bundle = self.kan_bundle if route.model_route == INVERSE_KAN else self.mlp_bundle
        return bundle, route.mass_state, route.route_mode, route.confidence_level

    def predict(self, target_mass_g_min: float, opening_mm: float) -> tuple[float, float, str, str, str, str, str]:
        bundle, mass_state, route_mode, confidence_level = self.route(target_mass_g_min, opening_mm)
        raw_speed = float(bundle.predict_speed(target_mass_g_min, opening_mm))
        speed_r_min_cmd = raw_speed
        speed_clip_state = ""
        return (
            raw_speed,
            speed_r_min_cmd,
            bundle.config.name,
            mass_state,
            route_mode,
            speed_clip_state,
            confidence_level,
        )


def build_row_decision(
    *,
    timestamp_ms: int,
    pass_id: int,
    row_index: int,
    x_m: float,
    y_m: float,
    zone_id: str,
    target_rate_kg_ha: float,
    target_mass_g_min: float,
    row_state: str = IN_FIELD,
    opening_mm: float,
    raw_speed_r_min: float,
    speed_r_min_cmd: float,
    model_route: str,
    mass_state: str,
    route_mode: str,
    speed_clip_state: str,
    confidence_level: str,
) -> RowDecision:
    return RowDecision(
        timestamp_ms=int(timestamp_ms),
        pass_id=int(pass_id),
        row_index=int(row_index),
        x_m=float(x_m),
        y_m=float(y_m),
        zone_id=str(zone_id),
        target_rate_kg_ha=float(target_rate_kg_ha),
        target_mass_g_min=float(target_mass_g_min),
        row_state=str(row_state),
        opening_mm=float(opening_mm),
        raw_speed_r_min=float(raw_speed_r_min),
        speed_r_min_cmd=float(speed_r_min_cmd),
        model_route=str(model_route),
        mass_state=str(mass_state),
        route_mode=str(route_mode),
        speed_clip_state=str(speed_clip_state),
        confidence_level=str(confidence_level),
    )
