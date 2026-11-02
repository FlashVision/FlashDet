"""Tests for the Pico-Zoo backbone/neck and Pico training-log fixes."""

import csv
from types import SimpleNamespace

import pytest
import torch

from flashdet.models.architectures.flashdet import FlashDet, FlashDetPico
from flashdet.models.backbone.pico_zoo_backbone import (
    PICO_BACKBONE_NAMES,
    PicoZooBackbone,
    get_pico_preset,
    normalize_backbone_name,
)
from flashdet.models.detector import build_model
from flashdet.models.neck.pico_zoo_neck import (
    PICO_NECK_FAMILIES,
    PicoZooNeck,
    resolve_neck_family,
)


def test_all_table_names_resolve():
    assert len(PICO_BACKBONE_NAMES) == 40
    for name in PICO_BACKBONE_NAMES:
        preset = get_pico_preset(name)
        assert preset["name"] == name
        assert preset["family"] in ("rep", "ghost", "attention", "mamba", "lcnet")
        assert len(preset["stage_channels"]) == 3


def test_backbone_aliases():
    assert normalize_backbone_name("mobilenet_v4") == "mobilenetv4"
    assert normalize_backbone_name("shufflenet_v2") == "shufflenetv2"
    assert normalize_backbone_name("pp_lcnet_v2") == "pp_lcnetv2"
    with pytest.raises(ValueError, match="Unknown Pico backbone"):
        normalize_backbone_name("not_a_real_backbone")


@pytest.mark.parametrize("name", PICO_BACKBONE_NAMES)
def test_every_backbone_forward(name):
    model = FlashDet(num_classes=5, size="p", backbone_type=name).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    assert out["o2o_cls"].shape[0] == 1
    assert out["o2o_reg"].shape[-1] == 4


@pytest.mark.parametrize("family", PICO_NECK_FAMILIES)
def test_every_neck_family_forward(family):
    model = FlashDet(
        num_classes=5, size="p", backbone_type="repvit", neck_type=family
    ).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 320, 320))
    assert out["o2o_cls"].shape[0] == 1


def test_neck_auto_matches_backbone_family():
    assert resolve_neck_family("edgenext") == "attention"
    assert resolve_neck_family("mambaout") == "mamba"
    assert resolve_neck_family("pp_lcnet") == "lcnet"
    model = FlashDet(num_classes=5, size="p", backbone_type="edgenext")
    assert isinstance(model, FlashDetPico)
    assert model.neck_type == "attention"
    assert isinstance(model.neck, PicoZooNeck)


def test_rep_backbone_fuses_without_changing_output():
    model = PicoZooBackbone("repvit").eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        before = model(x)
        model.fuse()
        after = model(x)
    for a, b in zip(before, after):
        assert torch.allclose(a, b, atol=1e-5)


def test_build_model_passes_pico_selectors():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            num_classes=7,
            size="p",
            backbone_type="edgenext",
            neck_type="ghost",
        ),
        total_epochs=10,
    )
    model = build_model(cfg, architecture="flashdet")
    assert isinstance(model, FlashDetPico)
    assert model.backbone_type == "edgenext"
    assert model.neck_type == "ghost"
    assert isinstance(model.neck, PicoZooNeck)


def test_training_csv_aligns_validation_by_epoch(tmp_path):
    from train import _save_training_csv

    history = {
        "epoch": [1, 2, 3, 4, 5],
        "lr": [0.1, 0.1, 0.1, 0.1, 0.1],
        "train_loss": [5, 4, 3, 2, 1],
        "val_loss": [9.0, 8.0],
        "mAP@0.5": [0.10, 0.20],
        "train_box": [1, 1, 1, 1, 1],
        "train_cls": [1, 1, 1, 1, 1],
        "train_l1": [1, 1, 1, 1, 1],
        "val_box": [2, 2],
        "val_cls": [3, 3],
        "val_l1": [4, 4],
        "val_epoch": [1, 5],
    }
    path = tmp_path / "training_log.csv"
    _save_training_csv(history, str(path))

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 5
    assert rows[0]["val_loss"] == "9.000000"
    assert rows[0]["val_epoch"] == "1.000000"
    assert rows[1]["val_loss"] == ""
    assert rows[3]["val_loss"] == ""
    assert rows[4]["val_loss"] == "8.000000"
    assert rows[4]["mAP@0.5"] == "0.200000"
    assert rows[4]["val_epoch"] == "5.000000"
