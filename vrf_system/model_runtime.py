from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - 打包后的精简运行时允许不带 torch
    torch = None

from .defaults import DEFAULT_KAN_ARTIFACT_DIR, DEFAULT_MLP_ARTIFACT_DIR
from .domain import ModelBundleConfig


EPS = 1e-8
KAN_NPZ_KEYS = {
    "kan1_input_grid": "kan1.input_grid",
    "kan1_base_weight": "kan1.base_weight",
    "kan1_spline_weight": "kan1.spline_weight",
    "kan2_input_grid": "kan2.input_grid",
    "kan2_base_weight": "kan2.base_weight",
    "kan2_spline_weight": "kan2.spline_weight",
}


class ModelArtifactError(RuntimeError):
    """模型工件异常。"""


@dataclass(slots=True)
class NumpyKANLayer:
    input_grid: np.ndarray
    base_weight: np.ndarray
    spline_weight: np.ndarray
    grid_size: int
    spline_order: int

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        base_out = _silu(x) @ self.base_weight.T
        bases = _kan_b_splines(x, self.input_grid, self.grid_size, self.spline_order).reshape(x.shape[0], -1)
        spline_weights = self.spline_weight.reshape(self.base_weight.shape[0], -1)
        spline_out = bases @ spline_weights.T
        return np.asarray(base_out + spline_out, dtype=np.float32)


@dataclass(slots=True)
class NumpyInverseKANModel:
    kan1: NumpyKANLayer
    kan2: NumpyKANLayer

    def predict(self, features_norm: np.ndarray) -> np.ndarray:
        hidden = self.kan1.forward(features_norm)
        output = self.kan2.forward(hidden)
        return output.reshape(-1)


@dataclass(slots=True)
class ModelBundle:
    config: ModelBundleConfig
    meta: dict[str, Any]
    model: Any

    @property
    def training_domain(self) -> dict[str, float]:
        return dict(self.meta.get("training_domain", {}))

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self.meta.get("extra", {}).get("policy", {}))

    @property
    def norm(self) -> dict[str, Any]:
        return dict(self.meta.get("normalization_params", {}))

    def in_training_domain(self, target_mass_g_min: float, opening_mm: float) -> bool:
        domain = self.training_domain
        return (
            float(domain["target_mass_min"]) <= float(target_mass_g_min) <= float(domain["target_mass_max"])
            and float(domain["opening_min"]) <= float(opening_mm) <= float(domain["opening_max"])
        )

    def predict_speed(self, target_mass_g_min: float, opening_mm: float) -> float:
        features = np.asarray([[float(target_mass_g_min), float(opening_mm)]], dtype=np.float32)
        x_min = np.asarray(self.norm["X_min"], dtype=np.float32)
        x_max = np.asarray(self.norm["X_max"], dtype=np.float32)
        y_min = float(self.norm["y_min"])
        y_max = float(self.norm["y_max"])
        features_norm = (features - x_min) / (x_max - x_min + EPS)

        if self.config.model_type == "inverse_MLP":
            pred_norm = np.asarray(self.model.predict(features_norm), dtype=np.float32).reshape(-1)
        elif self.config.model_type == "inverse_KAN":
            pred_norm = np.asarray(self.model.predict(features_norm), dtype=np.float32).reshape(-1)
        else:
            raise ModelArtifactError(f"不支持的模型类型：{self.config.model_type}")

        pred_raw = pred_norm * (y_max - y_min + EPS) + y_min
        return float(pred_raw.reshape(-1)[0])


def _silu(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / (1.0 + np.exp(-x)), dtype=np.float32)


def _kan_b_splines(
    x: np.ndarray,
    input_grid: np.ndarray,
    grid_size: int,
    spline_order: int,
) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=np.float32)
    h = (grid[:, -1:] - grid[:, 0:1]) / max(grid_size, 1)

    left_span = np.arange(spline_order, 0, -1, dtype=np.float32).reshape(1, -1)
    right_span = np.arange(1, spline_order + 1, dtype=np.float32).reshape(1, -1)
    left_pad = grid[:, 0:1] - left_span * h
    right_pad = grid[:, -1:] + right_span * h
    grid = np.concatenate([left_pad, grid, right_pad], axis=1)

    x_expanded = np.asarray(x, dtype=np.float32)[:, :, None]
    grid_expanded = grid[None, :, :]
    bases = ((x_expanded >= grid_expanded[:, :, :-1]) & (x_expanded < grid_expanded[:, :, 1:])).astype(np.float32)

    for order in range(1, spline_order + 1):
        denom1 = grid_expanded[:, :, order:-1] - grid_expanded[:, :, : -(order + 1)]
        denom2 = grid_expanded[:, :, order + 1 :] - grid_expanded[:, :, 1:-order]
        term1 = (x_expanded - grid_expanded[:, :, : -(order + 1)]) / (denom1 + 1e-12) * bases[:, :, :-1]
        term2 = (grid_expanded[:, :, order + 1 :] - x_expanded) / (denom2 + 1e-12) * bases[:, :, 1:]
        bases = term1 + term2
    return np.ascontiguousarray(bases, dtype=np.float32)


def _resolve_kan_model_path(artifact_dir: Path) -> Path:
    return artifact_dir / "model.npz" if (artifact_dir / "model.npz").exists() else artifact_dir / "model.pth"


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _numpy_kan_layer(prefix: str, state: dict[str, np.ndarray]) -> NumpyKANLayer:
    input_grid = np.asarray(state[f"{prefix}.input_grid"], dtype=np.float32)
    base_weight = np.asarray(state[f"{prefix}.base_weight"], dtype=np.float32)
    spline_weight = np.asarray(state[f"{prefix}.spline_weight"], dtype=np.float32)
    grid_size = int(input_grid.shape[1] - 1)
    spline_order = int(spline_weight.shape[2] - grid_size)
    return NumpyKANLayer(
        input_grid=input_grid,
        base_weight=base_weight,
        spline_weight=spline_weight,
        grid_size=grid_size,
        spline_order=spline_order,
    )


def _load_kan_npz_state(model_path: str | Path) -> dict[str, np.ndarray]:
    model_path = Path(model_path).resolve()
    archive = np.load(model_path, allow_pickle=False)
    state = {
        state_key: np.asarray(archive[npz_key], dtype=np.float32)
        for npz_key, state_key in KAN_NPZ_KEYS.items()
    }
    archive.close()
    return state


def _load_kan_state_dict(model_path: str | Path) -> dict[str, np.ndarray]:
    if torch is None:
        raise ModelArtifactError("当前运行环境缺少 torch，无法读取 .pth 版 KAN 模型，请改用 model.npz。")
    raw_state = torch.load(Path(model_path).resolve(), map_location="cpu")
    return {key: _tensor_to_numpy(value) for key, value in raw_state.items()}


def _build_numpy_kan_model(state: dict[str, np.ndarray]) -> NumpyInverseKANModel:
    return NumpyInverseKANModel(
        kan1=_numpy_kan_layer("kan1", state),
        kan2=_numpy_kan_layer("kan2", state),
    )


def export_kan_model_to_npz(source_model_path: str | Path, output_path: str | Path) -> Path:
    state = _load_kan_state_dict(source_model_path)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_state = {
        npz_key: np.asarray(state[state_key], dtype=np.float32)
        for npz_key, state_key in KAN_NPZ_KEYS.items()
    }
    np.savez_compressed(output_path, **export_state)
    return output_path


def build_default_model_configs() -> tuple[ModelBundleConfig, ModelBundleConfig]:
    kan = ModelBundleConfig(
        name="inverse_KAN",
        model_type="inverse_KAN",
        model_path=_resolve_kan_model_path(DEFAULT_KAN_ARTIFACT_DIR),
        meta_path=DEFAULT_KAN_ARTIFACT_DIR / "meta.json",
    )
    mlp = ModelBundleConfig(
        name="inverse_MLP",
        model_type="inverse_MLP",
        model_path=DEFAULT_MLP_ARTIFACT_DIR / "model.joblib",
        meta_path=DEFAULT_MLP_ARTIFACT_DIR / "meta.json",
    )
    return kan, mlp


def bundle_config_from_artifact_dir(name: str, model_type: str, artifact_dir: str | Path) -> ModelBundleConfig:
    artifact_path = Path(artifact_dir).resolve()
    if model_type == "inverse_KAN":
        model_path = _resolve_kan_model_path(artifact_path)
    elif model_type == "inverse_MLP":
        model_path = artifact_path / "model.joblib"
    else:
        raise ModelArtifactError(f"未知模型类型：{model_type}")
    return ModelBundleConfig(
        name=name,
        model_type=model_type,
        model_path=model_path,
        meta_path=artifact_path / "meta.json",
    )


def load_model_bundle(config: ModelBundleConfig) -> ModelBundle:
    if not config.model_path.exists():
        raise ModelArtifactError(f"模型文件不存在：{config.model_path}")
    if not config.meta_path.exists():
        raise ModelArtifactError(f"模型元数据不存在：{config.meta_path}")

    meta = json.loads(config.meta_path.read_text(encoding="utf-8"))
    if config.model_type == "inverse_MLP":
        model = joblib.load(config.model_path)
    elif config.model_type == "inverse_KAN":
        if config.model_path.suffix.lower() == ".npz":
            state = _load_kan_npz_state(config.model_path)
        else:
            state = _load_kan_state_dict(config.model_path)
        model = _build_numpy_kan_model(state)
    else:
        raise ModelArtifactError(f"未知模型类型：{config.model_type}")
    return ModelBundle(config=config, meta=meta, model=model)
