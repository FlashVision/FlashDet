"""Tests for AnalyticsReport — JSON, text, and HTML report generation."""

import json
import os
import tempfile

import pytest

from flashdet.analytics.report import AnalyticsReport


class TestAnalyticsReport:
    """Report generation tests."""

    @pytest.fixture
    def report_with_sections(self):
        r = AnalyticsReport(title="Test Report", model_name="FlashDet-N")
        r.add_benchmark({"fps": 120.5, "latency_ms": 8.3, "params": 1_500_000, "model_size_mb": 5.8})
        r.add_flops({
            "total_flops": 2_800_000_000, "total_macs": 1_400_000_000,
            "total_params": 1_500_000, "flops_readable": "2.80G",
            "macs_readable": "1.40G", "params_readable": "1.50M",
        })
        r.add_metrics({"mAP": 0.425, "mAP_50": 0.612, "mAP_75": 0.380,
                       "mAP_small": 0.180, "mAP_medium": 0.420, "mAP_large": 0.580})
        r.add_error_analysis({"summary": {"classification": 10, "localization": 5, "missed": 3}})
        return r

    def test_to_dict(self, report_with_sections):
        d = report_with_sections.to_dict()
        assert d["title"] == "Test Report"
        assert "metadata" in d
        assert "sections" in d
        assert "benchmark" in d["sections"]
        assert "metrics" in d["sections"]

    def test_to_json_valid(self, report_with_sections):
        j = report_with_sections.to_json()
        parsed = json.loads(j)
        assert parsed["title"] == "Test Report"
        assert parsed["sections"]["benchmark"]["fps"] == 120.5

    def test_to_json_saves_file(self, report_with_sections, tmp_path):
        path = str(tmp_path / "report.json")
        report_with_sections.to_json(path=path)
        assert os.path.isfile(path)
        with open(path) as f:
            data = json.load(f)
        assert data["title"] == "Test Report"

    def test_to_text(self, report_with_sections):
        text = report_with_sections.to_text()
        assert "Test Report" in text
        assert "FPS" in text
        assert "mAP" in text
        assert "FlashDet-N" in text

    def test_to_text_saves_file(self, report_with_sections, tmp_path):
        path = str(tmp_path / "report.txt")
        report_with_sections.to_text(path=path)
        assert os.path.isfile(path)

    def test_to_html(self, report_with_sections):
        html = report_with_sections.to_html()
        assert "<html>" in html
        assert "Test Report" in html
        assert "120.5" in html

    def test_to_html_saves_file(self, report_with_sections, tmp_path):
        path = str(tmp_path / "report.html")
        report_with_sections.to_html(path=path)
        assert os.path.isfile(path)

    def test_empty_report(self):
        r = AnalyticsReport()
        text = r.to_text()
        assert "FlashDet Analytics Report" in text

    def test_compare_models(self):
        results = [
            {"fps": 120, "latency_ms": 8.3, "mAP": 0.42, "params": 1_500_000},
            {"fps": 80, "latency_ms": 12.5, "mAP": 0.48, "params": 6_000_000},
        ]
        text = AnalyticsReport.compare_models(results, model_names=["FlashDet-N", "YOLOv8-S"])
        assert "FlashDet-N" in text
        assert "YOLOv8-S" in text
        assert "fps" in text
