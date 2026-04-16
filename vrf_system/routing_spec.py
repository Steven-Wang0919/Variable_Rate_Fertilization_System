from __future__ import annotations

from dataclasses import dataclass


OPENING_SUPPORT_MIN_MM = 20.0
OPENING_SUPPORT_MAX_MM = 50.0
SPEED_SUPPORT_MIN_R_MIN = 20.0
SPEED_SUPPORT_MAX_R_MIN = 60.0
GLOBAL_MASS_SUPPORT_MIN_G_MIN = 1144.27021563529
GLOBAL_MASS_SUPPORT_MAX_G_MIN = 8234.02272554259

STRATEGY_OPENINGS_MM = (20.0, 35.0, 50.0)
T_LOW_MID_G_MIN = 3325.246337890625
T_MID_HIGH_G_MIN = 5305.74169921875

IN_FIELD = "IN_FIELD"
OUT_OF_FIELD = "OUT_OF_FIELD"

IN_OPENING_SUPPORT = "IN_OPENING_SUPPORT"
OUT_OF_OPENING_SUPPORT = "OUT_OF_OPENING_SUPPORT"

BELOW_SPEED_SUPPORT = "BELOW_SPEED_SUPPORT"
IN_SPEED_SUPPORT = "IN_SPEED_SUPPORT"
ABOVE_SPEED_SUPPORT = "ABOVE_SPEED_SUPPORT"

BELOW_GLOBAL_SUPPORT = "BELOW_GLOBAL_SUPPORT"
IN_GLOBAL_SUPPORT = "IN_GLOBAL_SUPPORT"
ABOVE_GLOBAL_SUPPORT = "ABOVE_GLOBAL_SUPPORT"

NORMAL_INTERPOLATION = "NORMAL_INTERPOLATION"
SPEED_EDGE_EXTRAPOLATION = "SPEED_EDGE_EXTRAPOLATION"
OPENING_EXTRAPOLATION = "OPENING_EXTRAPOLATION"
DOUBLE_EXTRAPOLATION_EXPERIMENTAL = "DOUBLE_EXTRAPOLATION_EXPERIMENTAL"
FORWARD_DOUBLE_EXTRAPOLATION_UNSUPPORTED_MESSAGE = "暂不支持双越界预测"

NORMAL_IN_RANGE = "NORMAL_IN_RANGE"
EXPERIMENTAL_EXTRAPOLATION = "EXPERIMENTAL_EXTRAPOLATION"
LOW_TAIL_NEAR_EXTRAPOLATION = "LOW_TAIL_NEAR_EXTRAPOLATION"
LOW_TAIL_FAR_EXTRAPOLATION = "LOW_TAIL_FAR_EXTRAPOLATION"
HIGH_TAIL_EXTRAPOLATION = "HIGH_TAIL_EXTRAPOLATION"
LOW_TAIL_NEAR_THRESHOLD_PCT = 20.0
INVERSE_EXTRAPOLATION_ROUTE_MODES = frozenset(
    (
        EXPERIMENTAL_EXTRAPOLATION,
        LOW_TAIL_NEAR_EXTRAPOLATION,
        LOW_TAIL_FAR_EXTRAPOLATION,
        HIGH_TAIL_EXTRAPOLATION,
    )
)

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"

FORWARD_KAN = "forward_KAN"
FORWARD_MLP = "forward_MLP"
INVERSE_KAN = "inverse_KAN"
INVERSE_MLP = "inverse_MLP"


@dataclass(frozen=True, slots=True)
class ForwardRoute:
    model_route: str
    opening_state: str
    speed_state: str
    route_mode: str
    confidence_level: str


@dataclass(frozen=True, slots=True)
class InverseRoute:
    model_route: str
    mass_state: str
    route_mode: str
    confidence_level: str


def select_strategy_opening(target_mass_g_min: float) -> float:
    value = float(target_mass_g_min)
    if value < T_LOW_MID_G_MIN:
        return STRATEGY_OPENINGS_MM[0]
    if value < T_MID_HIGH_G_MIN:
        return STRATEGY_OPENINGS_MM[1]
    return STRATEGY_OPENINGS_MM[2]


def opening_state(opening_mm: float) -> str:
    value = float(opening_mm)
    if OPENING_SUPPORT_MIN_MM <= value <= OPENING_SUPPORT_MAX_MM:
        return IN_OPENING_SUPPORT
    return OUT_OF_OPENING_SUPPORT


def speed_state(speed_r_min: float) -> str:
    value = float(speed_r_min)
    if value < SPEED_SUPPORT_MIN_R_MIN:
        return BELOW_SPEED_SUPPORT
    if value > SPEED_SUPPORT_MAX_R_MIN:
        return ABOVE_SPEED_SUPPORT
    return IN_SPEED_SUPPORT


def mass_state(target_mass_g_min: float) -> str:
    value = float(target_mass_g_min)
    if value < GLOBAL_MASS_SUPPORT_MIN_G_MIN:
        return BELOW_GLOBAL_SUPPORT
    if value > GLOBAL_MASS_SUPPORT_MAX_G_MIN:
        return ABOVE_GLOBAL_SUPPORT
    return IN_GLOBAL_SUPPORT


def forward_route_for(opening_mm: float, speed_r_min: float) -> ForwardRoute:
    current_opening_state = opening_state(opening_mm)
    current_speed_state = speed_state(speed_r_min)

    if current_opening_state == OUT_OF_OPENING_SUPPORT and current_speed_state != IN_SPEED_SUPPORT:
        raise ValueError(FORWARD_DOUBLE_EXTRAPOLATION_UNSUPPORTED_MESSAGE)

    if current_opening_state == IN_OPENING_SUPPORT and current_speed_state == IN_SPEED_SUPPORT:
        model_route = FORWARD_KAN
        route_mode = NORMAL_INTERPOLATION
        confidence_level = HIGH
    elif current_opening_state == IN_OPENING_SUPPORT:
        model_route = FORWARD_MLP
        route_mode = SPEED_EDGE_EXTRAPOLATION
        confidence_level = LOW
    else:
        model_route = FORWARD_MLP
        route_mode = OPENING_EXTRAPOLATION
        confidence_level = LOW

    return ForwardRoute(
        model_route=model_route,
        opening_state=current_opening_state,
        speed_state=current_speed_state,
        route_mode=route_mode,
        confidence_level=confidence_level,
    )


def inverse_route_for(target_mass_g_min: float, strategy_opening_mm: float) -> InverseRoute:
    current_mass_state = mass_state(target_mass_g_min)
    opening_value = float(strategy_opening_mm)
    if opening_value not in STRATEGY_OPENINGS_MM:
        raise ValueError(f"Unsupported strategy opening: {opening_value}")

    if current_mass_state == IN_GLOBAL_SUPPORT:
        return InverseRoute(
            model_route=INVERSE_KAN,
            mass_state=current_mass_state,
            route_mode=NORMAL_IN_RANGE,
            confidence_level=HIGH,
        )
    if current_mass_state == BELOW_GLOBAL_SUPPORT:
        pct_low = (
            (GLOBAL_MASS_SUPPORT_MIN_G_MIN - float(target_mass_g_min))
            / GLOBAL_MASS_SUPPORT_MIN_G_MIN
            * 100.0
        )
        if pct_low < LOW_TAIL_NEAR_THRESHOLD_PCT:
            return InverseRoute(
                model_route=INVERSE_KAN,
                mass_state=current_mass_state,
                route_mode=LOW_TAIL_NEAR_EXTRAPOLATION,
                confidence_level=MEDIUM,
            )
        return InverseRoute(
            model_route=INVERSE_MLP,
            mass_state=current_mass_state,
            route_mode=LOW_TAIL_FAR_EXTRAPOLATION,
            confidence_level=LOW,
        )
    if current_mass_state == ABOVE_GLOBAL_SUPPORT:
        return InverseRoute(
            model_route=INVERSE_KAN,
            mass_state=current_mass_state,
            route_mode=HIGH_TAIL_EXTRAPOLATION,
            confidence_level=MEDIUM,
        )
    raise ValueError(f"Unsupported inverse routing state: {current_mass_state}")
