from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMPARE_RUN_NAME = "20260326T200342_compare_all"


def _first_existing_path(*candidates: Path) -> Path:
    resolved = [candidate.resolve() for candidate in candidates]
    for candidate in resolved:
        if candidate.exists():
            return candidate
    return resolved[0]


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    project_root: Path
    executable_dir: Path
    bundle_root: Path
    compare_root: Path
    model_artifacts_root: Path
    samples_root: Path
    default_compare_run: Path
    default_kan_artifact_dir: Path
    default_mlp_artifact_dir: Path
    default_canonical_predictions: Path
    default_forward_kan_artifact_dir: Path
    default_forward_canonical_predictions: Path
    default_sample_prescription: Path
    default_output_root: Path


def resolve_runtime_paths(
    *,
    project_root: str | Path | None = None,
    executable_path: str | Path | None = None,
    bundle_root: str | Path | None = None,
    frozen: bool | None = None,
) -> RuntimePaths:
    resolved_project_root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()

    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if is_frozen:
        resolved_executable_dir = Path(executable_path or sys.executable).resolve().parent
        resolved_bundle_root = Path(bundle_root or getattr(sys, "_MEIPASS", resolved_executable_dir)).resolve()
        default_output_root = resolved_executable_dir / "outputs"
    else:
        resolved_executable_dir = resolved_project_root
        resolved_bundle_root = resolved_project_root
        default_output_root = resolved_project_root / "outputs"

    packaged_artifacts_root = _first_existing_path(
        resolved_executable_dir / "model_artifacts",
        resolved_bundle_root / "model_artifacts",
        resolved_project_root / "model_artifacts",
    )
    samples_root = _first_existing_path(
        resolved_executable_dir / "samples",
        resolved_bundle_root / "samples",
        resolved_project_root / "samples",
    )
    compare_root = _first_existing_path(
        resolved_project_root.parent / "ComPare",
        resolved_executable_dir / "ComPare",
        resolved_bundle_root / "ComPare",
    )
    default_compare_run = compare_root / "runs" / DEFAULT_COMPARE_RUN_NAME

    default_kan_artifact_dir = _first_existing_path(
        packaged_artifacts_root / "inverse_KAN",
        default_compare_run / "artifacts" / "inverse" / "inverse_KAN",
    )
    default_mlp_artifact_dir = _first_existing_path(
        packaged_artifacts_root / "inverse_MLP",
        default_compare_run / "artifacts" / "inverse" / "inverse_MLP",
    )
    default_forward_kan_artifact_dir = _first_existing_path(
        packaged_artifacts_root / "forward_KAN",
        default_compare_run / "artifacts" / "forward" / "KAN",
    )
    default_canonical_predictions = default_compare_run / "inverse_model_predictions_all.csv"
    default_forward_canonical_predictions = default_compare_run / "forward_model_predictions.csv"
    default_sample_prescription = _first_existing_path(
        samples_root / "prescription_grid.csv",
        resolved_project_root / "samples" / "prescription_grid.csv",
    )

    return RuntimePaths(
        project_root=resolved_project_root,
        executable_dir=resolved_executable_dir,
        bundle_root=resolved_bundle_root,
        compare_root=compare_root,
        model_artifacts_root=packaged_artifacts_root,
        samples_root=samples_root,
        default_compare_run=default_compare_run,
        default_kan_artifact_dir=default_kan_artifact_dir,
        default_mlp_artifact_dir=default_mlp_artifact_dir,
        default_canonical_predictions=default_canonical_predictions,
        default_forward_kan_artifact_dir=default_forward_kan_artifact_dir,
        default_forward_canonical_predictions=default_forward_canonical_predictions,
        default_sample_prescription=default_sample_prescription,
        default_output_root=default_output_root,
    )


RUNTIME_PATHS = resolve_runtime_paths()

PROJECT_ROOT = RUNTIME_PATHS.project_root
WORKSPACE_ROOT = PROJECT_ROOT
COMPARE_ROOT = RUNTIME_PATHS.compare_root
MODEL_ARTIFACTS_ROOT = RUNTIME_PATHS.model_artifacts_root
SAMPLES_ROOT = RUNTIME_PATHS.samples_root
DEFAULT_COMPARE_RUN = RUNTIME_PATHS.default_compare_run
DEFAULT_KAN_ARTIFACT_DIR = RUNTIME_PATHS.default_kan_artifact_dir
DEFAULT_MLP_ARTIFACT_DIR = RUNTIME_PATHS.default_mlp_artifact_dir
DEFAULT_FORWARD_KAN_ARTIFACT_DIR = RUNTIME_PATHS.default_forward_kan_artifact_dir
DEFAULT_INVERSE_CANONICAL_PREDICTIONS = RUNTIME_PATHS.default_canonical_predictions
DEFAULT_CANONICAL_PREDICTIONS = RUNTIME_PATHS.default_canonical_predictions
DEFAULT_FORWARD_CANONICAL_PREDICTIONS = RUNTIME_PATHS.default_forward_canonical_predictions
DEFAULT_SAMPLE_PRESCRIPTION = RUNTIME_PATHS.default_sample_prescription
DEFAULT_OUTPUT_ROOT = RUNTIME_PATHS.default_output_root
