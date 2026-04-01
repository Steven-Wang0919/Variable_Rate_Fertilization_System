from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT
COMPARE_ROOT = WORKSPACE_ROOT.parent / "ComPare"
DEFAULT_COMPARE_RUN = COMPARE_ROOT / "runs" / "20260326T200342_compare_all"
DEFAULT_KAN_ARTIFACT_DIR = DEFAULT_COMPARE_RUN / "artifacts" / "inverse" / "inverse_KAN"
DEFAULT_MLP_ARTIFACT_DIR = DEFAULT_COMPARE_RUN / "artifacts" / "inverse" / "inverse_MLP"
DEFAULT_CANONICAL_PREDICTIONS = DEFAULT_COMPARE_RUN / "inverse_model_predictions_all.csv"
DEFAULT_SAMPLE_PRESCRIPTION = WORKSPACE_ROOT / "samples" / "prescription_grid.csv"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs"
