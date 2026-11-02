# Pico-Zoo: Backbone + Neck Selectors

`FlashDet-P` (`size="p"`) can now use any architecture name from the Pico model
table. Each name selects a lightweight backbone family and a matching neck
family, while keeping the same 3-level FPN interface, STAL assignment, E2EDualHead
and NMS-free inference path.

## Supported architectures

| Category | Names |
|---|---|
| A. Reparameterized CNN | `repvit`, `mobileone`, `fastvit`, `repghostnet` |
| B. Lightweight attention | `edgenext`, `mobilevitv2`, `efficientformerv2`, `shvit`, `efficientvit`, `swiftformer` |
| C. Lightweight CNN | `efficientnetv2`, `emo`, `fasternet`, `starnet`, `mobilenetv4`, `ghostnetv2` |
| D. Visual Mamba / SSM | `mobilemamba`, `efficientvmamba`, `mambavision`, `mambaout`, `efficientvim` |
| E. Detection-oriented CNN | `pp_lcnet`, `lcnet`, `esnet`, `pp_lcnetv2`, `peleenet`, `shufflenetv2` |
| F. Modern efficient CNN | `inceptionnext`, `unireplknet`, `convnextv2`, `efficientrep`, `rexnet` |
| G. Lightweight transformers | `mobileformer`, `edgevit`, `tinyvit`, `levit`, `lowformer` |
| H. Additional SSM | `localmamba`, `vmamba`, `plainmamba` |

`shufflenetv2` uses the existing pretrained `LiteBackbone`. `pico_v2` and
`repnext` remain available as the original RepNeXt backbone.

## Family mapping

| Backbone family | Block | Matching neck |
|---|---|---|
| Reparameterized | multi-branch DW + structural reparameterization | `rep` |
| Attention | depthwise local mixing + SE/gating | `attention` |
| Ghost / cheap CNN | cheap feature generation + pointwise projection | `ghost` |
| Mamba / SSM | MambaOut-style gated DW mixer (no CUDA kernel needed) | `mamba` |
| Detection CNN | depthwise separable + SE | `lcnet` |

The attention/Mamba transformer names use lightweight CNN approximations designed
for 320/416-pixel Pico inference. They are not full reproductions of the original
papers, but they preserve the family's block design and are deployable on CPU,
mobile and edge devices.

## Usage

```python
from flashdet import FlashDet

# Backbone and neck are chosen automatically from the family
model = FlashDet(num_classes=80, size="p", backbone_type="edgenext")

# Explicit neck family
model = FlashDet(
    num_classes=80,
    size="p",
    backbone_type="repvit",
    neck_type="ghost",
)

print(model.get_model_info())
```

```bash
# RepViT-style Pico
python train.py --model-size p --backbone repvit \
  --input-size 320 --epochs 300 --amp \
  --train-images data/coco2017/train --val-images data/coco2017/valid

# MambaOut-style Pico with matching Mamba neck
python train.py --model-size p --backbone mambaout --neck mamba ...

# PP-LCNet-style Pico with LCNet neck
python train.py --model-size p --backbone pp_lcnet --neck lcnet ...
```

Or use the parameterized helper:

```bash
BACKBONE=edgenext NECK=attention bash scripts/train_pico_zoo.sh
```

## Checkpoints

Training checkpoints and inference weights now store `backbone_type` and
`neck_type`. `test.py`, `flashdet predict` and `flashdet export` rebuild the exact
same backbone/neck automatically.
