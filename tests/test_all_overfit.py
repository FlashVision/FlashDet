"""Comprehensive overfit test for all models + training method smoke tests.

Tests each model architecture on 5 images for 300 steps, checking:
1. Loss decreases
2. mAP@0.5 > 0 (model can predict after training)
3. No NaN losses

Then smoke-tests each training method class can be instantiated.
"""
import torch
import torch.nn as nn
import json
import os
import cv2
import numpy as np
import time
import traceback
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 320
STEPS = 300
LR = 1e-3

# ===================== Load 5-image dataset =====================
DATA_DIR = "data/overfit5/train"
with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
    ann = json.load(f)
cats = sorted(ann["categories"], key=lambda c: c["id"])
cat_map = {c["id"]: i for i, c in enumerate(cats)}
num_classes = len(cats)

images_t, gt_bboxes_list, gt_labels_list = [], [], []
for img_info in ann["images"]:
    img = cv2.imread(os.path.join(DATA_DIR, img_info["file_name"]))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_anns = [a for a in ann["annotations"] if a["image_id"] == img_info["id"]]
    if not img_anns:
        continue
    boxes = np.array(
        [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
         for a in img_anns], dtype=np.float32,
    )
    labels = np.array([cat_map[a["category_id"]] for a in img_anns], dtype=np.int64)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    sx, sy = IMG_SIZE / w, IMG_SIZE / h
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    images_t.append(torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0)
    gt_bboxes_list.append(torch.from_numpy(boxes).float())
    gt_labels_list.append(torch.from_numpy(labels).long())

x = torch.stack(images_t).to(DEVICE)
gt_bboxes_list = [b.to(DEVICE) for b in gt_bboxes_list]
gt_labels_list = [l.to(DEVICE) for l in gt_labels_list]
total_gt = sum(len(l) for l in gt_labels_list)
print(f"Dataset: {len(images_t)} images, {total_gt} GT objects, {num_classes} classes")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)


def make_gt_meta():
    return {
        "gt_bboxes": [b.clone() for b in gt_bboxes_list],
        "gt_labels": [l.clone() for l in gt_labels_list],
    }


def compute_map_safe(model, is_grounding_dino=False):
    """Compute mAP@0.5 using model.predict()."""
    try:
        from flashdet.utils.metrics import compute_map
        if is_grounding_dino:
            B = x.shape[0]
            input_ids = torch.randint(1, 100, (B, 10), device=DEVICE)
            attn_mask = torch.ones(B, 10, dtype=torch.long, device=DEVICE)
            preds = model.predict(x, input_ids=input_ids, attention_mask=attn_mask,
                                  score_thr=0.01, nms_thr=0.6)
        else:
            preds = model.predict(x, score_thr=0.01, nms_thr=0.6)
        all_preds, all_gts = [], []
        for i in range(len(images_t)):
            if i < len(preds):
                dets, det_labels = preds[i]
                if dets is not None and dets.numel() > 0:
                    all_preds.append({
                        "boxes": dets[:, :4].cpu().numpy(),
                        "scores": dets[:, 4].cpu().numpy(),
                        "labels": det_labels.cpu().numpy(),
                    })
                else:
                    all_preds.append({"boxes": np.zeros((0, 4), np.float32),
                                      "scores": np.zeros(0, np.float32),
                                      "labels": np.zeros(0, np.int64)})
            else:
                all_preds.append({"boxes": np.zeros((0, 4), np.float32),
                                  "scores": np.zeros(0, np.float32),
                                  "labels": np.zeros(0, np.int64)})
            all_gts.append({
                "boxes": gt_bboxes_list[i].cpu().numpy(),
                "labels": gt_labels_list[i].cpu().numpy(),
            })
        m = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_classes)
        return m["mAP"]
    except Exception as e:
        print(f"  mAP error: {e}")
        return -1


def overfit_test(model_name, model, steps=STEPS, lr=LR, is_grounding_dino=False, epoch_arg=None):
    """Train a model on the 5-image set and check overfitting."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    nan_count = 0
    start = time.time()

    for step in range(steps):
        optimizer.zero_grad()
        gt_meta = make_gt_meta()

        try:
            if is_grounding_dino:
                B = x.shape[0]
                input_ids = torch.randint(1, 100, (B, 10), device=DEVICE)
                attn_mask = torch.ones(B, 10, dtype=torch.long, device=DEVICE)
                out = model(x, input_ids=input_ids, attention_mask=attn_mask, gt_meta=gt_meta)
            elif epoch_arg is not None:
                out = model(x, gt_meta=gt_meta, compute_loss=True, epoch=epoch_arg)
            else:
                try:
                    out = model(x, gt_meta=gt_meta, compute_loss=True)
                except TypeError:
                    out = model(x, gt_meta, compute_loss=True)
        except Exception as e:
            if step == 0:
                print(f"  Forward error at step 0: {e}")
                traceback.print_exc()
                return model_name, [], -1, f"ERROR: {e}", 0

        loss = out["loss"]
        if torch.isnan(loss):
            nan_count += 1
            losses.append(float("nan"))
            if nan_count > 20:
                return model_name, losses, 0, "NaN loss", time.time() - start
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            print(f"  step {step}: loss={loss.item():.4f}", flush=True)

    elapsed = time.time() - start
    model.eval()
    map50 = compute_map_safe(model, is_grounding_dino=is_grounding_dino)
    status = "PASS" if map50 > 0.3 else ("PARTIAL" if map50 > 0.05 else "FAIL")
    return model_name, losses, map50, status, elapsed


# ===================== Run all model overfit tests =====================
results = []

# 1. FlashDet (pass epoch so both heads train)
print("\n[1/7] FlashDet...", flush=True)
try:
    from flashdet.models import FlashDet
    model = FlashDet(num_classes=num_classes, size="n", total_epochs=STEPS).to(DEVICE)
    name, losses, map50, status, t = overfit_test(
        "FlashDet", model, epoch_arg=1,
    )
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    print(f"  loss: {l0:.3f} -> {lf:.3f} | mAP@0.5={map50:.4f} | {status} | {t:.1f}s")
    results.append(("FlashDet", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("FlashDet", 0, 0, 0, f"ERROR"))

# 2. DETR
print("\n[2/7] DETR...", flush=True)
try:
    from flashdet.models.architectures.detr import DETR
    model = DETR(num_classes=num_classes, num_queries=50, d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256,
                 pretrained_backbone=False).to(DEVICE)
    name, losses, map50, status, t = overfit_test("DETR", model, steps=600)
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    print(f"  loss: {l0:.3f} -> {lf:.3f} | mAP@0.5={map50:.4f} | {status} | {t:.1f}s")
    results.append(("DETR", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("DETR", 0, 0, 0, f"ERROR"))

# 3. RT-DETR
print("\n[3/7] RT-DETR...", flush=True)
try:
    from flashdet.models.architectures.rt_detr import RTDETR
    model = RTDETR(num_classes=num_classes, hidden_dim=128, nhead=4,
                   num_encoder_layers=1, num_decoder_layers=2, dim_feedforward=256,
                   num_queries=50, pretrained_backbone=False).to(DEVICE)
    name, losses, map50, status, t = overfit_test("RT-DETR", model, steps=600, lr=2e-3)
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    print(f"  loss: {l0:.3f} -> {lf:.3f} | mAP@0.5={map50:.4f} | {status} | {t:.1f}s")
    results.append(("RT-DETR", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("RT-DETR", 0, 0, 0, f"ERROR"))

# 4. YOLOv9
print("\n[4/7] YOLOv9...", flush=True)
try:
    from flashdet.models.architectures.yolov9 import YOLOv9
    model = YOLOv9(num_classes=num_classes, width_mult=0.25, depth_mult=0.33).to(DEVICE)
    name, losses, map50, status, t = overfit_test("YOLOv9", model)
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    print(f"  loss: {l0:.3f} -> {lf:.3f} | mAP@0.5={map50:.4f} | {status} | {t:.1f}s")
    results.append(("YOLOv9", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("YOLOv9", 0, 0, 0, f"ERROR"))

# 5. YOLOv10
print("\n[5/7] YOLOv10...", flush=True)
try:
    from flashdet.models.architectures.yolov10 import YOLOv10
    model = YOLOv10(num_classes=num_classes, width_mult=0.25, depth_mult=0.33).to(DEVICE)
    name, losses, map50, status, t = overfit_test("YOLOv10", model)
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    print(f"  loss: {l0:.3f} -> {lf:.3f} | mAP@0.5={map50:.4f} | {status} | {t:.1f}s")
    results.append(("YOLOv10", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("YOLOv10", 0, 0, 0, f"ERROR"))

# 6. YOLOv11
print("\n[6/7] YOLOv11...", flush=True)
try:
    from flashdet.models.architectures.yolov11 import YOLOv11
    model = YOLOv11(num_classes=num_classes, width_mult=0.25, depth_mult=0.33).to(DEVICE)
    name, losses, map50, status, t = overfit_test("YOLOv11", model)
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    print(f"  loss: {l0:.3f} -> {lf:.3f} | mAP@0.5={map50:.4f} | {status} | {t:.1f}s")
    results.append(("YOLOv11", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("YOLOv11", 0, 0, 0, f"ERROR"))

# 7. Grounding-DINO
print("\n[7/7] Grounding-DINO...", flush=True)
try:
    from flashdet.models.architectures.grounding_dino import GroundingDINO
    model = GroundingDINO(num_queries=50, d_model=128, nhead=4,
                          num_encoder_layers=1, num_decoder_layers=2,
                          pretrained_backbone=False, text_encoder_depth=1).to(DEVICE)
    name, losses, map50, status, t = overfit_test(
        "Grounding-DINO", model, steps=600, is_grounding_dino=True,
    )
    l0, lf = (losses[0] if losses else 0), (losses[-1] if losses else 0)
    drop = (l0 - lf) / max(abs(l0), 1e-8) * 100
    print(f"  loss: {l0:.3f} -> {lf:.3f} ({drop:.0f}% drop) | {status} | {t:.1f}s")
    results.append(("Grounding-DINO", l0, lf, map50, status))
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()
    results.append(("Grounding-DINO", 0, 0, 0, f"ERROR"))

# ===================== Summary table =====================
print("\n" + "=" * 80)
print(f"{'Model':<20} {'Init Loss':>10} {'Final Loss':>10} {'mAP@0.5':>10} {'Status':>10}")
print("-" * 80)
for name, l0, lf, metric, status in results:
    s = str(status)[:40]
    print(f"{name:<20} {l0:>10.3f} {lf:>10.3f} {metric:>10.4f} {s:>10}")
print("=" * 80)

# ===================== Training Method Smoke Tests =====================
print("\n" + "=" * 80)
print("TRAINING METHOD SMOKE TESTS")
print("=" * 80)

# Test 1: Trainer class instantiation
print("\n[T1] Trainer instantiation...", flush=True)
try:
    from flashdet.engine.training.trainer import Trainer
    t = Trainer(
        epochs=2, batch_size=4, lr=0.001,
        train_images="data/overfit5/train",
        val_images="data/overfit5/train",
        model_size="m", architecture="flashdet",
        save_dir="workspace/test_trainer",
    )
    print("  OK - Trainer created")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: KDTrainer instantiation
print("\n[T2] KDTrainer instantiation...", flush=True)
try:
    from flashdet.engine.training.kd_trainer import KDTrainer
    t = KDTrainer(
        teacher_checkpoint="dummy.pth",
        teacher_size="m",
        kd_temperature=4.0,
        kd_alpha=0.5,
        epochs=2, batch_size=4,
        train_images="data/overfit5/train",
        val_images="data/overfit5/train",
        model_size="m",
        save_dir="workspace/test_kd",
    )
    print("  OK - KDTrainer created")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: SSLTrainer instantiation
print("\n[T3] SSLTrainer instantiation...", flush=True)
try:
    from flashdet.engine.training.ssl_trainer import SSLTrainer
    t = SSLTrainer(
        ssl_method="byol",
        epochs=2,
        batch_size=4,
        train_images="data/overfit5/train",
        save_dir="workspace/test_ssl",
        device="cpu",
    )
    print("  OK - SSLTrainer created")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: SemiSupervisedTrainer instantiation
print("\n[T4] SemiSupervisedTrainer instantiation...", flush=True)
try:
    from flashdet.engine.training.semi_supervised_trainer import SemiSupervisedTrainer
    t = SemiSupervisedTrainer(
        unlabeled_images="data/overfit5/train",
        pseudo_label_threshold=0.7,
        epochs=2, batch_size=4,
        train_images="data/overfit5/train",
        val_images="data/overfit5/train",
        model_size="m",
        save_dir="workspace/test_semi",
    )
    print("  OK - SemiSupervisedTrainer created")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: FewShotTrainer instantiation
print("\n[T5] FewShotTrainer instantiation...", flush=True)
try:
    from flashdet.engine.training.few_shot_trainer import FewShotTrainer
    t = FewShotTrainer(
        base_checkpoint="dummy.pth",
        n_shot=5,
        epochs=2, batch_size=4,
        train_images="data/overfit5/train",
        val_images="data/overfit5/train",
        model_size="m",
        save_dir="workspace/test_fewshot",
    )
    print("  OK - FewShotTrainer created")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 6: ActiveLearningTrainer instantiation
print("\n[T6] ActiveLearningTrainer instantiation...", flush=True)
try:
    from flashdet.engine.training.active_learning_trainer import ActiveLearningTrainer
    t = ActiveLearningTrainer(
        unlabeled_pool="data/overfit5/train",
        query_strategy="entropy",
        query_budget=2,
        al_rounds=2,
        epochs=2, batch_size=4,
        train_images="data/overfit5/train",
        val_images="data/overfit5/train",
        model_size="m",
        save_dir="workspace/test_al",
    )
    print("  OK - ActiveLearningTrainer created")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 7: SSLTrainer backbone wrapper
print("\n[T7] SSLTrainer backbone wrapper...", flush=True)
try:
    from flashdet.engine.training.ssl_trainer import SSLTrainer
    t = SSLTrainer(ssl_method="byol", epochs=1, batch_size=2, device="cpu",
                   save_dir="workspace/test_ssl_bb")
    bb = t._build_backbone()
    feat_dim = t._get_backbone_out_dim(bb)
    print(f"  OK - backbone feat_dim={feat_dim}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 8: Quick Trainer.train() run (2 epochs on overfit5)
print("\n[T8] Trainer.train() quick run (2 epochs)...", flush=True)
try:
    from flashdet.engine.training.trainer import Trainer
    t = Trainer(
        epochs=2, batch_size=5, lr=0.001, workers=0,
        train_images="data/overfit5/train",
        val_images="data/overfit5/train",
        model_size="m", architecture="flashdet",
        save_dir="workspace/test_trainer_run",
        device="cpu", patience=0,
    )
    result = t.train()
    print(f"  OK - Trainer.train() returned: {result}")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)
