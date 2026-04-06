from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vrf_system.defaults import (
    DEFAULT_CANONICAL_PREDICTIONS,
    DEFAULT_FORWARD_CANONICAL_PREDICTIONS,
    DEFAULT_FORWARD_KAN_ARTIFACT_DIR,
    DEFAULT_FORWARD_MLP_ARTIFACT_DIR,
    DEFAULT_KAN_ARTIFACT_DIR,
    DEFAULT_MLP_ARTIFACT_DIR,
)
from vrf_system.model_runtime import (
    build_default_forward_model_configs,
    build_default_model_configs,
    bundle_config_from_artifact_dir,
    export_kan_model_to_npz,
    load_model_bundle,
)


@unittest.skipUnless(
    DEFAULT_CANONICAL_PREDICTIONS.exists()
    and DEFAULT_KAN_ARTIFACT_DIR.exists()
    and DEFAULT_MLP_ARTIFACT_DIR.exists(),
    "默认反向模型工件不存在，跳过模型加载测试。",
)
class InverseModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        kan_config, mlp_config = build_default_model_configs()
        cls.kan_bundle = load_model_bundle(kan_config)
        cls.mlp_bundle = load_model_bundle(mlp_config)
        cls.kan_inputs = np.load(DEFAULT_KAN_ARTIFACT_DIR / "test_inputs.npy")
        cls.mlp_inputs = np.load(DEFAULT_MLP_ARTIFACT_DIR / "test_inputs.npy")

    def test_inverse_models_should_load_and_predict_finite_values(self) -> None:
        kan_predicted = [
            self.kan_bundle.predict_speed(target_mass_g_min=row[0], opening_mm=row[1])
            for row in self.kan_inputs[:10]
        ]
        mlp_predicted = [
            self.mlp_bundle.predict_speed(target_mass_g_min=row[0], opening_mm=row[1])
            for row in self.mlp_inputs[:10]
        ]

        self.assertTrue(all(np.isfinite(kan_predicted)))
        self.assertTrue(all(np.isfinite(mlp_predicted)))

    def test_inverse_kan_npz_export_should_still_predict_finite_values(self) -> None:
        outputs_root = Path.cwd() / "outputs"
        outputs_root.mkdir(exist_ok=True)
        tmp = outputs_root / "test_inverse_kan_npz"
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            artifact_dir = tmp / "inverse_KAN"
            artifact_dir.mkdir(parents=True)
            shutil.copy2(DEFAULT_KAN_ARTIFACT_DIR / "meta.json", artifact_dir / "meta.json")
            export_kan_model_to_npz(DEFAULT_KAN_ARTIFACT_DIR / "model.pth", artifact_dir / "model.npz")

            config = bundle_config_from_artifact_dir("inverse_KAN_npz", "inverse_KAN", artifact_dir)
            bundle = load_model_bundle(config)
            predicted = [
                bundle.predict_speed(target_mass_g_min=row[0], opening_mm=row[1])
                for row in self.kan_inputs[:10]
            ]
            self.assertTrue(all(np.isfinite(predicted)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(
    DEFAULT_FORWARD_CANONICAL_PREDICTIONS.exists()
    and DEFAULT_FORWARD_KAN_ARTIFACT_DIR.exists()
    and DEFAULT_FORWARD_MLP_ARTIFACT_DIR.exists(),
    "默认前向模型工件不存在，跳过前向一致性测试。",
)
class ForwardModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        forward_kan_config, forward_mlp_config = build_default_forward_model_configs()
        cls.forward_kan_bundle = load_model_bundle(forward_kan_config)
        cls.forward_mlp_bundle = load_model_bundle(forward_mlp_config)
        cls.reference_df = pd.read_csv(DEFAULT_FORWARD_CANONICAL_PREDICTIONS)
        cls.forward_kan_inputs = np.load(DEFAULT_FORWARD_KAN_ARTIFACT_DIR / "test_inputs.npy")
        cls.forward_mlp_inputs = np.load(DEFAULT_FORWARD_MLP_ARTIFACT_DIR / "test_inputs.npy")

    def test_forward_kan_prediction_matches_reference_output(self) -> None:
        predicted = [
            self.forward_kan_bundle.predict_mass(opening_mm=row[0], speed_r_min=row[1])
            for row in self.forward_kan_inputs
        ]
        expected = self.reference_df["KAN_pred"].to_numpy(dtype=float)
        np.testing.assert_allclose(predicted, expected, atol=1e-4, rtol=1e-5)

    def test_forward_mlp_prediction_matches_reference_output(self) -> None:
        predicted = [
            self.forward_mlp_bundle.predict_mass(opening_mm=row[0], speed_r_min=row[1])
            for row in self.forward_mlp_inputs
        ]
        expected = self.reference_df["MLP_pred"].to_numpy(dtype=float)
        np.testing.assert_allclose(predicted, expected, atol=1e-4, rtol=1e-5)
