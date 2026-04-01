from __future__ import annotations

import csv
import tempfile
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


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


class PrescriptionMapTests(unittest.TestCase):
    def test_missing_column_should_raise_chinese_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "bad.csv"
            write_csv(
                csv_path,
                REQUIRED_HEADERS[:-1],
                [["A1", 1, 1, 2, 2, 320]],
            )
            with self.assertRaisesRegex(PrescriptionValidationError, "缺少必要字段"):
                PrescriptionMap.from_csv(csv_path)

    def test_duplicate_cell_id_should_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "dup.csv"
            write_csv(
                csv_path,
                REQUIRED_HEADERS,
                [
                    ["A1", 1, 1, 2, 2, 320, "Z1"],
                    ["A1", 3, 1, 2, 2, 340, "Z2"],
                ],
            )
            with self.assertRaisesRegex(PrescriptionValidationError, "重复 cell_id"):
                PrescriptionMap.from_csv(csv_path)

    def test_find_cell_should_return_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ok.csv"
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
