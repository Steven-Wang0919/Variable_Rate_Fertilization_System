from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vrf_system.defaults import (
    DEFAULT_CANONICAL_PREDICTIONS,
    DEFAULT_KAN_ARTIFACT_DIR,
    DEFAULT_MLP_ARTIFACT_DIR,
)
from vrf_system.model_runtime import (
    build_default_model_configs,
    bundle_config_from_artifact_dir,
    export_kan_model_to_npz,
    load_model_bundle,
)


@unittest.skipUnless(
    DEFAULT_CANONICAL_PREDICTIONS.exists()
    and DEFAULT_KAN_ARTIFACT_DIR.exists()
    and DEFAULT_MLP_ARTIFACT_DIR.exists(),
    "默认论文模型工件不存在，跳过模型一致性测试。",
)
class ModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        kan_config, mlp_config = build_default_model_configs()
        cls.kan_bundle = load_model_bundle(kan_config)
        cls.mlp_bundle = load_model_bundle(mlp_config)
        cls.reference_df = pd.read_csv(DEFAULT_CANONICAL_PREDICTIONS)
        cls.kan_inputs = np.load(DEFAULT_KAN_ARTIFACT_DIR / "test_inputs.npy")
        cls.mlp_inputs = np.load(DEFAULT_MLP_ARTIFACT_DIR / "test_inputs.npy")

    def test_inverse_mlp_prediction_matches_reference_output(self) -> None:
        predicted = [
            self.mlp_bundle.predict_speed(target_mass_g_min=row[0], opening_mm=row[1])
            for row in self.mlp_inputs
        ]
        expected = self.reference_df["inverse_MLP_pred"].to_numpy(dtype=float)
        np.testing.assert_allclose(predicted, expected, atol=1e-4, rtol=1e-5)

    def test_inverse_kan_prediction_matches_reference_output(self) -> None:
        predicted = [
            self.kan_bundle.predict_speed(target_mass_g_min=row[0], opening_mm=row[1])
            for row in self.kan_inputs
        ]
        expected = self.reference_df["inverse_KAN_pred"].to_numpy(dtype=float)
        np.testing.assert_allclose(predicted, expected, atol=1e-4, rtol=1e-5)

    def test_inverse_kan_npz_prediction_matches_reference_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "inverse_KAN"
            artifact_dir.mkdir(parents=True)
            shutil.copy2(DEFAULT_KAN_ARTIFACT_DIR / "meta.json", artifact_dir / "meta.json")
            export_kan_model_to_npz(DEFAULT_KAN_ARTIFACT_DIR / "model.pth", artifact_dir / "model.npz")

            config = bundle_config_from_artifact_dir("inverse_KAN_npz", "inverse_KAN", artifact_dir)
            bundle = load_model_bundle(config)
            predicted = [
                bundle.predict_speed(target_mass_g_min=row[0], opening_mm=row[1])
                for row in self.kan_inputs
            ]
            expected = self.reference_df["inverse_KAN_pred"].to_numpy(dtype=float)
            np.testing.assert_allclose(predicted, expected, atol=1e-4, rtol=1e-5)
