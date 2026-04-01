from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .defaults import DEFAULT_KAN_ARTIFACT_DIR, DEFAULT_MLP_ARTIFACT_DIR
from .domain import ModelBundleConfig


EPS = 1e-8


class ModelArtifactError(RuntimeError):
    """模型工件异常。"""


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 10,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        self.input_grid = torch.einsum(
            "i,j->ij",
            torch.ones(in_features),
            torch.linspace(grid_range[0], grid_range[1], grid_size + 1),
        )
        self.input_grid = nn.Parameter(self.input_grid, requires_grad=False)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            coeff = self.curve2coeff(self.input_grid, noise)
            self.spline_weight.data.copy_(
                (self.scale_spline if self.scale_spline is not None else 1.0) * coeff
            )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.input_grid
        h = (grid[:, -1:] - grid[:, 0:1]) / self.grid_size
        device = grid.device

        left_span = torch.arange(
            self.spline_order, 0, -1, device=device, dtype=grid.dtype
        ).unsqueeze(0)
        right_span = torch.arange(
            1, self.spline_order + 1, device=device, dtype=grid.dtype
        ).unsqueeze(0)
        left_pad = grid[:, 0:1] - left_span * h
        right_pad = grid[:, -1:] + right_span * h

        grid = torch.cat([left_pad, grid, right_pad], dim=1)
        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(0)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)

        for order in range(1, self.spline_order + 1):
            denom1 = grid[:, :, order:-1] - grid[:, :, : -(order + 1)]
            denom2 = grid[:, :, order + 1 :] - grid[:, :, 1:-order]
            term1 = (x - grid[:, :, : -(order + 1)]) / (denom1 + 1e-12) * bases[:, :, :-1]
            term2 = (grid[:, :, order + 1 :] - x) / (denom2 + 1e-12) * bases[:, :, 1:]
            bases = term1 + term2
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        matrix_a = self.b_splines(x.transpose(0, 1)).transpose(0, 1)
        matrix_b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(matrix_a, matrix_b).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out


class InverseKANModel(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 1) -> None:
        super().__init__()
        self.kan1 = KANLayer(input_dim, hidden_dim, grid_size=10)
        self.kan2 = KANLayer(hidden_dim, output_dim, grid_size=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan2(self.kan1(x))


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
            with torch.no_grad():
                tensor_in = torch.as_tensor(features_norm, dtype=torch.float32)
                pred_norm = self.model(tensor_in).cpu().numpy().reshape(-1)
        else:
            raise ModelArtifactError(f"不支持的模型类型：{self.config.model_type}")

        pred_raw = pred_norm * (y_max - y_min + EPS) + y_min
        return float(pred_raw.reshape(-1)[0])


def build_default_model_configs() -> tuple[ModelBundleConfig, ModelBundleConfig]:
    kan = ModelBundleConfig(
        name="inverse_KAN",
        model_type="inverse_KAN",
        model_path=DEFAULT_KAN_ARTIFACT_DIR / "model.pth",
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
        model_name = "model.pth"
    elif model_type == "inverse_MLP":
        model_name = "model.joblib"
    else:
        raise ModelArtifactError(f"未知模型类型：{model_type}")
    return ModelBundleConfig(
        name=name,
        model_type=model_type,
        model_path=artifact_path / model_name,
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
        hidden_dim = int(meta.get("best_config", {}).get("hidden_dim", 32))
        model = InverseKANModel(hidden_dim=hidden_dim)
        state_dict = torch.load(config.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise ModelArtifactError(f"未知模型类型：{config.model_type}")
    return ModelBundle(config=config, meta=meta, model=model)
