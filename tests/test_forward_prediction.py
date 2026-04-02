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


def build_forward_bundle(*, pred_norm: float) -> ModelBundle:
    return ModelBundle(
        config=ModelBundleConfig(
            name="forward_KAN",
            model_type="forward_KAN",
            model_path=Path("forward_KAN/model.pth"),
            meta_path=Path("forward_KAN/meta.json"),
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

    def test_predict_forward_mass_should_raise_when_model_missing(self) -> None:
        with self.assertRaisesRegex(ModelArtifactError, "前向 KAN 预测模型"):
            self.controller.predict_forward_mass(35.0, 40.0)

    def test_predict_forward_mass_should_return_mass_and_equivalent_rate(self) -> None:
        self.controller.forward_kan_bundle = build_forward_bundle(pred_norm=0.5)

        result = self.controller.predict_forward_mass(
            35.0,
            40.0,
            row_spacing_m=0.5,
            travel_speed_kmh=6.0,
        )

        self.assertAlmostEqual(result.predicted_mass_g_min, 50.0, places=5)
        self.assertAlmostEqual(result.equivalent_rate_kg_ha or 0.0, 10.0, places=5)
        self.assertEqual(result.selected_model, "forward_KAN")
        self.assertEqual(result.domain_status, "in_domain")
        self.assertEqual(result.status, "ok")
        self.assertIs(self.controller.last_forward_prediction, result)

    def test_predict_forward_mass_should_report_extrapolation_modes(self) -> None:
        self.controller.forward_kan_bundle = build_forward_bundle(pred_norm=0.5)
        cases = [
            ((10.0, 40.0), "opening_extrapolation"),
            ((35.0, 10.0), "speed_extrapolation"),
            ((10.0, 10.0), "opening_and_speed_extrapolation"),
        ]

        for (opening_mm, speed_r_min), expected_status in cases:
            with self.subTest(opening_mm=opening_mm, speed_r_min=speed_r_min):
                result = self.controller.predict_forward_mass(opening_mm, speed_r_min)
                self.assertEqual(result.domain_status, expected_status)

    def test_predict_forward_mass_should_clamp_negative_output(self) -> None:
        self.controller.forward_kan_bundle = build_forward_bundle(pred_norm=-0.1)

        result = self.controller.predict_forward_mass(35.0, 40.0)

        self.assertEqual(result.predicted_mass_g_min, 0.0)
        self.assertEqual(result.status, "clamped_low")
        self.assertEqual(result.domain_status, "in_domain")

    def test_predict_forward_mass_should_skip_equivalent_rate_when_context_invalid(self) -> None:
        self.controller.forward_kan_bundle = build_forward_bundle(pred_norm=0.5)

        result = self.controller.predict_forward_mass(
            35.0,
            40.0,
            row_spacing_m=0.0,
            travel_speed_kmh=6.0,
        )

        self.assertIsNone(result.equivalent_rate_kg_ha)
