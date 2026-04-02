from __future__ import annotations

from pathlib import Path

from .defaults import DEFAULT_SAMPLE_PRESCRIPTION
from .domain import MachineConfig
from .engine import ModelRouter
from .exporters import export_simulation_result
from .model_runtime import (
    ModelArtifactError,
    ModelBundle,
    build_default_model_configs,
    bundle_config_from_artifact_dir,
    load_model_bundle,
)
from .prescription import PrescriptionMap
from .simulator import FieldSimulator


class SimulationController:
    def __init__(self) -> None:
        self.kan_bundle: ModelBundle | None = None
        self.mlp_bundle: ModelBundle | None = None
        self.router: ModelRouter | None = None
        self.prescription_map: PrescriptionMap | None = None
        self.last_result = None

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
