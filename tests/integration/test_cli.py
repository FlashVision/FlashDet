"""Integration tests — CLI commands."""

import subprocess
import sys

import pytest


class TestCLI:
    """Test flashdet CLI commands."""

    def _run(self, args):
        result = subprocess.run(
            [sys.executable, "-m", "flashdet.cli"] + args,
            capture_output=True, text=True, timeout=60,
        )
        return result

    def test_version(self):
        result = self._run(["version"])
        assert result.returncode == 0
        assert "FlashDet" in result.stdout

    def test_settings(self):
        result = self._run(["settings"])
        assert result.returncode == 0
        assert "Python" in result.stdout
        assert "PyTorch" in result.stdout

    def test_check(self):
        result = self._run(["check"])
        assert result.returncode == 0
        assert "passed" in result.stdout.lower() or "✓" in result.stdout

    def test_help(self):
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "train" in result.stdout
        assert "predict" in result.stdout

    def test_train_help(self):
        result = self._run(["train", "--help"])
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--epochs" in result.stdout

    def test_invalid_command(self):
        result = self._run(["nonexistent_command"])
        assert result.returncode != 0

    def test_datasets_list(self):
        result = self._run(["datasets"])
        assert result.returncode == 0
        assert "dataset" in result.stdout.lower() or "Available" in result.stdout
