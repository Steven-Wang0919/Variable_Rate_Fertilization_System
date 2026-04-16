from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from vrf_system.controller import SimulationController
from vrf_system.domain import ModelBundleConfig
from vrf_system.model_runtime import ModelArtifactError, ModelBundle


class ConstantPredictor:
    def __init__(self, pred_norm: float) -> None:
        self.pred_norm = float(pred_norm)

    def predict(self, features_norm) -> np.ndarray:
        batch_size = np.asarray(features_norm).shape[0]
        return np.full((batch_size,), self.pred_norm, dtype=np.float32)


def build_forward_bundle(*, name: str, model_type: str, pred_norm: float) -> ModelBundle:
    model_file = "model.joblib" if model_type == "forward_MLP" else "model.pth"
    return ModelBundle(
        config=ModelBundleConfig(
            name=name,
            model_type=model_type,
            model_path=Path(name) / model_file,
            meta_path=Path(name) / "meta.json",
        ),
        meta={
            "training_domain": {
                "feature_names": ["opening_mm", "speed_r_min"],
                "target_name": "mass_g_min",
                "opening_min": 20.0,
                "opening_max": 50.0,
                "speed_min": 20.0,
                "speed_max": 60.0,
                "mass_min": 0.0,
                "mass_max": 100.0,
            },
            "normalization_params": {
                "X_min": [[20.0, 20.0]],
                "X_max": [[50.0, 60.0]],
                "y_min": 0.0,
                "y_max": 100.0,
            },
        },
        model=ConstantPredictor(pred_norm=pred_norm),
    )


class ForwardPredictionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = SimulationController()
        self.controller.forward_kan_bundle = build_forward_bundle(
            name="forward_KAN",
            model_type="forward_KAN",
            pred_norm=0.5,
        )
        self.controller.forward_mlp_bundle = build_forward_bundle(
            name="forward_MLP",
            model_type="forward_MLP",
            pred_norm=0.75,
        )

    def test_predict_forward_mass_should_raise_when_forward_bundles_missing(self) -> None:
        self.controller.forward_kan_bundle = None
        self.controller.forward_mlp_bundle = None
        with self.assertRaisesRegex(ModelArtifactError, "前向 KAN 和 MLP 预测模型"):
            self.controller.predict_forward_mass(35.0, 40.0)

    def test_predict_forward_mass_should_use_forward_kan_inside_opening_support(self) -> None:
        result = self.controller.predict_forward_mass(
            35.0,
            40.0,
            row_spacing_m=0.5,
            travel_speed_kmh=6.0,
        )

        self.assertAlmostEqual(result.mass_hat_g_min, 50.0, places=5)
        self.assertAlmostEqual(result.equivalent_rate_kg_ha or 0.0, 10.0, places=5)
        self.assertEqual(result.model_route, "forward_KAN")
        self.assertEqual(result.opening_state, "IN_OPENING_SUPPORT")
        self.assertEqual(result.speed_state, "IN_SPEED_SUPPORT")
        self.assertEqual(result.route_mode, "NORMAL_INTERPOLATION")
        self.assertEqual(result.confidence_level, "HIGH")
        self.assertIs(self.controller.last_forward_prediction, result)

    def test_predict_forward_mass_should_keep_forward_kan_for_speed_only_extrapolation(self) -> None:
        result = self.controller.predict_forward_mass(35.0, 10.0)

        self.assertAlmostEqual(result.mass_hat_g_min, 75.0, places=5)
        self.assertEqual(result.model_route, "forward_MLP")
        self.assertEqual(result.opening_state, "IN_OPENING_SUPPORT")
        self.assertEqual(result.speed_state, "BELOW_SPEED_SUPPORT")
        self.assertEqual(result.route_mode, "SPEED_EDGE_EXTRAPOLATION")
        self.assertEqual(result.confidence_level, "LOW")

    def test_predict_forward_mass_should_use_forward_mlp_for_high_speed_extrapolation(self) -> None:
        result = self.controller.predict_forward_mass(35.0, 70.0)

        self.assertAlmostEqual(result.mass_hat_g_min, 75.0, places=5)
        self.assertEqual(result.model_route, "forward_MLP")
        self.assertEqual(result.opening_state, "IN_OPENING_SUPPORT")
        self.assertEqual(result.speed_state, "ABOVE_SPEED_SUPPORT")
        self.assertEqual(result.route_mode, "SPEED_EDGE_EXTRAPOLATION")
        self.assertEqual(result.confidence_level, "LOW")

    def test_predict_forward_mass_should_use_forward_mlp_for_opening_extrapolation(self) -> None:
        result = self.controller.predict_forward_mass(10.0, 40.0)

        self.assertAlmostEqual(result.mass_hat_g_min, 75.0, places=5)
        self.assertEqual(result.model_route, "forward_MLP")
        self.assertEqual(result.opening_state, "OUT_OF_OPENING_SUPPORT")
        self.assertEqual(result.speed_state, "IN_SPEED_SUPPORT")
        self.assertEqual(result.route_mode, "OPENING_EXTRAPOLATION")
        self.assertEqual(result.confidence_level, "LOW")

    def test_predict_forward_mass_should_reject_double_extrapolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "暂不支持双越界预测"):
            self.controller.predict_forward_mass(10.0, 10.0)

    def test_predict_forward_mass_should_not_clamp_negative_output(self) -> None:
        self.controller.forward_kan_bundle = build_forward_bundle(
            name="forward_KAN",
            model_type="forward_KAN",
            pred_norm=-0.1,
        )

        result = self.controller.predict_forward_mass(35.0, 40.0)

        self.assertEqual(result.mass_hat_g_min, -10.0)
        self.assertEqual(result.model_route, "forward_KAN")
        self.assertEqual(result.route_mode, "NORMAL_INTERPOLATION")

    def test_predict_forward_mass_should_skip_equivalent_rate_when_context_invalid(self) -> None:
        result = self.controller.predict_forward_mass(
            35.0,
            40.0,
            row_spacing_m=0.0,
            travel_speed_kmh=6.0,
        )

        self.assertIsNone(result.equivalent_rate_kg_ha)
