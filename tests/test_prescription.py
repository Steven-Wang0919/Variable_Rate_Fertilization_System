from __future__ import annotations

import csv
import shutil
import unittest
from pathlib import Path

from vrf_system.prescription import PrescriptionMap, PrescriptionValidationError


REQUIRED_HEADERS = [
    "cell_id",
    "center_x_m",
    "center_y_m",
    "width_m",
    "height_m",
    "target_rate_kg_ha",
    "zone_id",
]
ROUTING_COVERAGE_SAMPLE = Path("samples/prescription_grid_routing_coverage.csv")


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


class PrescriptionMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = Path("outputs/test_prescription")
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_missing_column_should_raise_chinese_error(self) -> None:
        csv_path = self.output_dir / "bad.csv"
        write_csv(
            csv_path,
            REQUIRED_HEADERS[:-1],
            [["A1", 1, 1, 2, 2, 320]],
        )
        with self.assertRaisesRegex(PrescriptionValidationError, "处方图缺少必要字段"):
            PrescriptionMap.from_csv(csv_path)

    def test_duplicate_cell_id_should_raise(self) -> None:
        csv_path = self.output_dir / "dup.csv"
        write_csv(
            csv_path,
            REQUIRED_HEADERS,
            [
                ["A1", 1, 1, 2, 2, 320, "Z1"],
                ["A1", 3, 1, 2, 2, 340, "Z2"],
            ],
        )
        with self.assertRaisesRegex(PrescriptionValidationError, "处方图存在重复 cell_id"):
            PrescriptionMap.from_csv(csv_path)

    def test_find_cell_should_return_zone(self) -> None:
        csv_path = self.output_dir / "ok.csv"
        write_csv(
            csv_path,
            REQUIRED_HEADERS,
            [
                ["A1", 1, 1, 2, 2, 320, "Z1"],
                ["A2", 3, 1, 2, 2, 420, "Z2"],
            ],
        )
        prescription = PrescriptionMap.from_csv(csv_path)
        hit = prescription.find_cell(1.2, 0.8)
        miss = prescription.find_cell(8.0, 8.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.zone_id, "Z1")
        self.assertIsNone(miss)

    def test_routing_coverage_sample_should_load_with_expected_grid(self) -> None:
        prescription = PrescriptionMap.from_csv(ROUTING_COVERAGE_SAMPLE)

        self.assertEqual(len(prescription.cells), 30)
        self.assertEqual(prescription.bounds.width, 15.0)
        self.assertEqual(prescription.bounds.height, 18.0)
        self.assertEqual(prescription.rate_range(), (120.0, 1850.0))
        self.assertEqual(
            {cell.zone_id for cell in prescription.cells},
            {"EL", "L", "M", "H", "EH"},
        )
        self.assertEqual(prescription.cells[0].cell_id, "R01C01")
        self.assertEqual(prescription.cells[-1].cell_id, "R06C05")
