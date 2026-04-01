from __future__ import annotations

from dataclasses import dataclass

from .domain import RowDecision, STANDARD_OPENINGS_MM
from .model_runtime import ModelBundle


def target_mass_from_rate(rate_kg_ha: float, row_spacing_m: float, travel_speed_kmh: float) -> float:
    return float(rate_kg_ha) * float(row_spacing_m) * float(travel_speed_kmh) * 1.6666667


@dataclass(slots=True)
class ModelRouter:
    kan_bundle: ModelBundle
    mlp_bundle: ModelBundle

    def select_strategy_opening(self, target_mass_g_min: float) -> float:
        policy = self.kan_bundle.policy or self.mlp_bundle.policy
        target_openings = tuple(policy.get("target_openings_mm", STANDARD_OPENINGS_MM))
        low_mid = float(policy["threshold_low_mid"])
        mid_high = float(policy["threshold_mid_high"])
        value = float(target_mass_g_min)
        if value < low_mid:
            return float(target_openings[0])
        if value < mid_high:
            return float(target_openings[1])
        return float(target_openings[2])

    def route(self, target_mass_g_min: float, opening_mm: float) -> tuple[ModelBundle, str]:
        mass_in_domain = self.kan_bundle.training_domain["target_mass_min"] <= float(target_mass_g_min) <= self.kan_bundle.training_domain["target_mass_max"]
        opening_in_domain = self.kan_bundle.training_domain["opening_min"] <= float(opening_mm) <= self.kan_bundle.training_domain["opening_max"]
        opening_standard = float(opening_mm) in tuple(self.kan_bundle.policy.get("target_openings_mm", STANDARD_OPENINGS_MM))

        if mass_in_domain and opening_in_domain and opening_standard:
            return self.kan_bundle, "in_domain"
        if not mass_in_domain and opening_standard:
            return self.mlp_bundle, "mass_extrapolation"
        if mass_in_domain and not opening_standard:
            return self.mlp_bundle, "opening_extrapolation"
        return self.mlp_bundle, "mass_and_opening_extrapolation"

    def predict(self, target_mass_g_min: float, opening_mm: float) -> tuple[float, str, str, str]:
        bundle, domain_status = self.route(target_mass_g_min, opening_mm)
        predicted_speed = float(bundle.predict_speed(target_mass_g_min, opening_mm))
        status = "ok"
        if predicted_speed < 0:
            predicted_speed = 0.0
            status = "clamped_low"
            domain_status = f"{domain_status}_speed_floor"
        return predicted_speed, bundle.config.name, domain_status, status


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
    strategy_opening_mm: float,
    target_speed_r_min: float,
    selected_model: str,
    domain_status: str,
    status: str,
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
        strategy_opening_mm=float(strategy_opening_mm),
        target_speed_r_min=float(target_speed_r_min),
        selected_model=str(selected_model),
        domain_status=str(domain_status),
        status=str(status),
    )
