from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from vrf_system.domain import ModelBundleConfig
from vrf_system.engine import ModelRouter
from vrf_system.model_runtime import ModelBundle
from vrf_system.routing_spec import T_LOW_MID_G_MIN, T_MID_HIGH_G_MIN


class ConstantPredictor:
    def __init__(self, pred_norm: float) -> None:
        self.pred_norm = float(pred_norm)

    def predict(self, features_norm) -> np.ndarray:
        batch_size = np.asarray(features_norm).shape[0]
        return np.full((batch_size,), self.pred_norm, dtype=np.float32)


def build_inverse_bundle(*, name: str, model_type: str, pred_norm: float) -> ModelBundle:
    model_file = "model.joblib" if model_type == "inverse_MLP" else "model.pth"
    return ModelBundle(
        config=ModelBundleConfig(
            name=name,
            model_type=model_type,
            model_path=Path(name) / model_file,
            meta_path=Path(name) / "meta.json",
        ),
        meta={
            "training_domain": {
                "feature_names": ["target_mass_g_min", "opening_mm"],
                "target_name": "speed_r_min",
                "target_mass_min": 1196.3232421875,
                "target_mass_max": 8121.93408203125,
                "opening_min": 20.0,
                "opening_max": 50.0,
                "speed_min": 20.0,
                "speed_max": 60.0,
            },
            "normalization_params": {
                "X_min": [[1196.3232421875, 20.0]],
                "X_max": [[8121.93408203125, 50.0]],
                "y_min": 20.0,
                "y_max": 60.0,
            },
        },
        model=ConstantPredictor(pred_norm=pred_norm),
    )


class InverseRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = ModelRouter(
            kan_bundle=build_inverse_bundle(name="inverse_KAN", model_type="inverse_KAN", pred_norm=0.6),
            mlp_bundle=build_inverse_bundle(name="inverse_MLP", model_type="inverse_MLP", pred_norm=-0.05),
        )

    def test_select_strategy_opening_should_use_frozen_thresholds(self) -> None:
        self.assertEqual(self.router.select_strategy_opening(T_LOW_MID_G_MIN - 0.1), 20.0)
        self.assertEqual(self.router.select_strategy_opening(T_LOW_MID_G_MIN), 35.0)
        self.assertEqual(self.router.select_strategy_opening(T_MID_HIGH_G_MIN - 0.1), 35.0)
        self.assertEqual(self.router.select_strategy_opening(T_MID_HIGH_G_MIN), 50.0)

    def test_predict_should_use_inverse_mlp_for_low_end_experimental_extrapolation(self) -> None:
        raw_speed, speed_cmd, model_route, mass_state, route_mode, clip_state, confidence = self.router.predict(1000.0, 20.0)

        self.assertAlmostEqual(raw_speed, 18.0, places=5)
        self.assertEqual(speed_cmd, 20.0)
        self.assertEqual(model_route, "inverse_MLP")
        self.assertEqual(mass_state, "BELOW_GLOBAL_SUPPORT")
        self.assertEqual(route_mode, "EXPERIMENTAL_EXTRAPOLATION")
        self.assertEqual(clip_state, "CLIPPED_TO_20")
        self.assertEqual(confidence, "LOW")

    def test_predict_should_use_inverse_kan_inside_global_support(self) -> None:
        raw_speed, speed_cmd, model_route, mass_state, route_mode, clip_state, confidence = self.router.predict(4000.0, 35.0)

        self.assertAlmostEqual(raw_speed, 44.0, places=5)
        self.assertEqual(speed_cmd, 44.0)
        self.assertEqual(model_route, "inverse_KAN")
        self.assertEqual(mass_state, "IN_GLOBAL_SUPPORT")
        self.assertEqual(route_mode, "NORMAL_IN_RANGE")
        self.assertEqual(clip_state, "NOT_CLIPPED")
        self.assertEqual(confidence, "HIGH")

    def test_predict_should_use_inverse_kan_for_high_end_experimental_extrapolation(self) -> None:
        high_clip_router = ModelRouter(
            kan_bundle=build_inverse_bundle(name="inverse_KAN", model_type="inverse_KAN", pred_norm=1.3),
            mlp_bundle=build_inverse_bundle(name="inverse_MLP", model_type="inverse_MLP", pred_norm=-0.05),
        )
        raw_speed, speed_cmd, model_route, mass_state, route_mode, clip_state, confidence = high_clip_router.predict(9000.0, 50.0)

        self.assertAlmostEqual(raw_speed, 72.0, places=5)
        self.assertEqual(speed_cmd, 60.0)
        self.assertEqual(model_route, "inverse_KAN")
        self.assertEqual(mass_state, "ABOVE_GLOBAL_SUPPORT")
        self.assertEqual(route_mode, "EXPERIMENTAL_EXTRAPOLATION")
        self.assertEqual(clip_state, "CLIPPED_TO_60")
        self.assertEqual(confidence, "MEDIUM")

    def test_route_should_reject_na_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported inverse routing combination"):
            self.router.route(1000.0, 35.0)
        with self.assertRaisesRegex(ValueError, "Unsupported inverse routing combination"):
            self.router.route(9000.0, 20.0)
