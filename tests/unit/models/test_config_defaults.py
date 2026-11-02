"""Config/defaults tests — guard against hardcoded architecture values."""

from types import SimpleNamespace

import pytest

from flashdet.cfg import get_config, load_yaml_config
from flashdet.models.architectures.flashdet import FlashDet
from flashdet.models.detector import build_model


def test_default_fpn_strides_are_three_levels():
    cfg = get_config()
    assert cfg.model.strides == [8, 16, 32]


def test_head_levels_follow_strides():
    model = FlashDet(num_classes=5, size="n", strides=(8, 16, 32))
    assert len(model.head.o2o_heads) == 3
    assert len(model.head.o2m_heads) == 3


def test_four_level_strides_are_rejected():
    with pytest.raises(ValueError, match="exactly 3 FPN strides"):
        FlashDet(num_classes=5, size="n", strides=(8, 16, 32, 64))


def test_pico_architecture_values_can_be_overridden():
    model = FlashDet(
        num_classes=5,
        size="p",
        backbone_type="repvit",
        stem_channels=16,
        stage_depths=(1, 1, 1),
        neck_channels=64,
    )
    assert model.backbone.out_channels == [32, 64, 128]
    assert model.neck.out_channels == 64
    assert len(model.head.o2o_heads) == 3


def test_build_model_uses_config_selectors():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            num_classes=5,
            size="p",
            strides=[8, 16, 32],
            backbone_type="repvit",
            neck_type="ghost",
            pico_stem_channels=16,
            pico_stage_depths=(1, 1, 1),
            pico_neck_channels=64,
        ),
        total_epochs=10,
    )
    model = build_model(cfg, architecture="flashdet")
    assert model.backbone_type == "repvit"
    assert model.neck_type == "ghost"
    assert model.backbone.out_channels == [32, 64, 128]
    assert model.neck.out_channels == 64


def test_yaml_config_applies_new_fields(tmp_path):
    path = tmp_path / "pico.yaml"
    path.write_text(
        """
model:
  size: p
  input_size: [416, 416]
  backbone_type: edgenext
  neck_type: attention
  pico_stem_channels: 24
  pico_stage_depths: [1, 2, 1]
  pico_neck_channels: 96
  strides: [8, 16, 32]
train:
  nbs: 32
  lrf: 0.05
  grad_clip_norm: 5.0
  ema_decay: 0.999
  ema_warmup: 100
  multi_scale: true
  multi_scale_sizes: [256, 320, 416]
  val_score_thr: 0.02
  max_det: 100
""",
        encoding="utf-8",
    )
    cfg = load_yaml_config(str(path))
    assert cfg.model.size == "p"
    assert cfg.model.backbone_type == "edgenext"
    assert cfg.model.neck_type == "attention"
    assert cfg.model.pico_neck_channels == 96
    assert cfg.model.strides == [8, 16, 32]
    assert cfg.train.nbs == 32
    assert cfg.train.lrf == 0.05
    assert cfg.train.grad_clip_norm == 5.0
    assert cfg.train.ema_decay == 0.999
    assert cfg.train.multi_scale is True
    assert cfg.train.multi_scale_sizes == [256, 320, 416]
    assert cfg.train.val_score_thr == 0.02
    assert cfg.train.max_det == 100
