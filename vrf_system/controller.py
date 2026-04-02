from __future__ import annotations

from pathlib import Path

from .defaults import DEFAULT_SAMPLE_PRESCRIPTION
from .domain import ForwardPredictionResult, MachineConfig
from .engine import ModelRouter
from .exporters import export_simulation_result
from .model_runtime import (
    ModelArtifactError,
    ModelBundle,
    build_default_forward_model_config,
    build_default_model_configs,
    bundle_config_from_artifact_dir,
    load_model_bundle,
)
from .prescription import PrescriptionMap
from .simulator import FieldSimulator


MASS_PER_RATE_FACTOR = 1.6666667


def _forward_domain_status(bundle: ModelBundle, opening_mm: float, speed_r_min: float) -> str:
    domain = bundle.training_domain
    opening_in_domain = float(domain["opening_min"]) <= float(opening_mm) <= float(domain["opening_max"])
    speed_in_domain = float(domain["speed_min"]) <= float(speed_r_min) <= float(domain["speed_max"])
    if opening_in_domain and speed_in_domain:
        return "in_domain"
    if not opening_in_domain and speed_in_domain:
        return "opening_extrapolation"
    if opening_in_domain and not speed_in_domain:
        return "speed_extrapolation"
    return "opening_and_speed_extrapolation"


def _equivalent_rate_kg_ha(
    predicted_mass_g_min: float,
    *,
    row_spacing_m: float | None,
    travel_speed_kmh: float | None,
) -> float | None:
    if row_spacing_m is None or travel_speed_kmh is None:
        return None
    row_spacing = float(row_spacing_m)
    travel_speed = float(travel_speed_kmh)
    if row_spacing <= 0 or travel_speed <= 0:
        return None
    return float(predicted_mass_g_min) / (row_spacing * travel_speed * MASS_PER_RATE_FACTOR)


class SimulationController:
    def __init__(self) -> None:
        self.kan_bundle: ModelBundle | None = None
        self.mlp_bundle: ModelBundle | None = None
        self.forward_kan_bundle: ModelBundle | None = None
        self.router: ModelRouter | None = None
        self.prescription_map: PrescriptionMap | None = None
        self.last_result = None
        self.last_forward_prediction: ForwardPredictionResult | None = None

    def load_default_models(self) -> tuple[ModelBundle, ModelBundle]:
        kan_config, mlp_config = build_default_model_configs()
        return self.load_models(kan_config, mlp_config)

    def load_models_from_dirs(self, kan_dir: str | Path, mlp_dir: str | Path) -> tuple[ModelBundle, ModelBundle]:
        kan_config = bundle_config_from_artifact_dir("inverse_KAN", "inverse_KAN", kan_dir)
        mlp_config = bundle_config_from_artifact_dir("inverse_MLP", "inverse_MLP", mlp_dir)
        return self.load_models(kan_config, mlp_config)

    def load_models(self, kan_config, mlp_config) -> tuple[ModelBundle, ModelBundle]:
        self.kan_bundle = load_model_bundle(kan_config)
        self.mlp_bundle = load_model_bundle(mlp_config)
        self.router = ModelRouter(self.kan_bundle, self.mlp_bundle)
        return self.kan_bundle, self.mlp_bundle

    def load_default_forward_model(self) -> ModelBundle:
        return self.load_forward_model(build_default_forward_model_config())

    def load_forward_model_from_dir(self, forward_kan_dir: str | Path) -> ModelBundle:
        forward_config = bundle_config_from_artifact_dir("forward_KAN", "forward_KAN", forward_kan_dir)
        return self.load_forward_model(forward_config)

    def load_forward_model(self, forward_config) -> ModelBundle:
        self.forward_kan_bundle = load_model_bundle(forward_config)
        return self.forward_kan_bundle

    def load_prescription(self, csv_path: str | Path) -> PrescriptionMap:
        self.prescription_map = PrescriptionMap.from_csv(csv_path)
        return self.prescription_map

    def load_sample_prescription(self) -> PrescriptionMap:
        return self.load_prescription(DEFAULT_SAMPLE_PRESCRIPTION)

    def run_simulation(self, machine_config: MachineConfig | None = None):
        if self.router is None:
            raise ModelArtifactError("请先加载 KAN 与 MLP 模型包。")
        if self.prescription_map is None:
            raise ValueError("请先导入处方图。")
        config = machine_config or MachineConfig()
        simulator = FieldSimulator(self.router)
        self.last_result = simulator.run(self.prescription_map, config)
        return self.last_result

    def predict_forward_mass(
        self,
        opening_mm: float,
        speed_r_min: float,
        *,
        row_spacing_m: float | None = None,
        travel_speed_kmh: float | None = None,
    ) -> ForwardPredictionResult:
        if self.forward_kan_bundle is None:
            raise ModelArtifactError("请先加载前向 KAN 预测模型。")

        domain_status = _forward_domain_status(self.forward_kan_bundle, opening_mm, speed_r_min)
        predicted_mass = float(self.forward_kan_bundle.predict_mass(opening_mm, speed_r_min))
        status = "ok"
        if predicted_mass < 0:
            predicted_mass = 0.0
            status = "clamped_low"

        equivalent_rate = _equivalent_rate_kg_ha(
            predicted_mass,
            row_spacing_m=row_spacing_m,
            travel_speed_kmh=travel_speed_kmh,
        )
        self.last_forward_prediction = ForwardPredictionResult(
            opening_mm=float(opening_mm),
            speed_r_min=float(speed_r_min),
            predicted_mass_g_min=float(predicted_mass),
            equivalent_rate_kg_ha=equivalent_rate,
            selected_model=self.forward_kan_bundle.config.name,
            domain_status=domain_status,
            status=status,
        )
        return self.last_forward_prediction

    def export_last_result(
        self,
        output_root: str | Path | None = None,
        highlighted_frame_index: int = 0,
    ):
        if self.last_result is None:
            raise ValueError("当前没有可导出的仿真结果。")
        return export_simulation_result(
            self.last_result,
            output_root=output_root,
            highlighted_frame_index=highlighted_frame_index,
        )
