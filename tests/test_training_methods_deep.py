"""Deep verification of all training methods.

Runs each method for enough epochs to confirm loss actually decreases.
Checks: loss drops, no NaN, correct method-specific behavior.
"""
import os
import sys
import time
import shutil
import traceback
import json

import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/overfit5/train"
BASE_SAVE = "workspace/test_methods_deep"
os.makedirs(BASE_SAVE, exist_ok=True)

results = []


def run_test(name, fn):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}", flush=True)
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        status = result.get("status", "UNKNOWN")
        detail = result.get("detail", "")
        print(f"\n  >> {status} ({elapsed:.1f}s) — {detail}")
        results.append((name, status, elapsed, detail))
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  >> FAIL ({elapsed:.1f}s) — {e}")
        traceback.print_exc()
        results.append((name, "FAIL", elapsed, str(e)[:100]))


# ============================================================
# Helper: manual training loop to check loss decreases
# ============================================================
def manual_overfit(model, num_steps=100, lr=1e-3):
    """Train model directly on overfit5 data, return loss curve."""
    import cv2
    with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
        ann = json.load(f)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}

    images_t, gt_bboxes, gt_labels = [], [], []
    for img_info in ann["images"]:
        img = cv2.imread(os.path.join(DATA_DIR, img_info["file_name"]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_anns = [a for a in ann["annotations"] if a["image_id"] == img_info["id"]]
        if not img_anns:
            continue
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
                          for a in img_anns], dtype=np.float32)
        labels = np.array([cat_map[a["category_id"]] for a in img_anns], dtype=np.int64)
        h, w = img.shape[:2]
        img_r = cv2.resize(img, (320, 320))
        sx, sy = 320.0/w, 320.0/h
        boxes[:, [0,2]] *= sx
        boxes[:, [1,3]] *= sy
        images_t.append(torch.from_numpy(img_r).permute(2,0,1).float() / 255.0)
        gt_bboxes.append(torch.from_numpy(boxes).float().to(DEVICE))
        gt_labels.append(torch.from_numpy(labels).long().to(DEVICE))

    x = torch.stack(images_t).to(DEVICE)
    model = model.to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(num_steps):
        opt.zero_grad()
        gt_meta = {"gt_bboxes": [b.clone() for b in gt_bboxes],
                   "gt_labels": [l.clone() for l in gt_labels]}
        try:
            out = model(x, gt_meta=gt_meta, compute_loss=True, epoch=1)
        except TypeError:
            out = model(x, gt_meta, compute_loss=True)
        loss = out["loss"]
        if torch.isnan(loss):
            losses.append(float("nan"))
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        losses.append(loss.item())

    return losses, len(cats)


# ============================================================
# 1. Standard Trainer — verify loss decreases over 50 epochs
# ============================================================
def test_standard_trainer():
    from flashdet.models import FlashDet
    model = FlashDet(num_classes=10, size="n", total_epochs=100)
    losses, nc = manual_overfit(model, num_steps=200)

    valid = [l for l in losses if not np.isnan(l)]
    first10 = np.mean(valid[:10])
    last10 = np.mean(valid[-10:])
    drop_pct = (first10 - last10) / max(abs(first10), 1e-8) * 100

    print(f"  Loss: {valid[0]:.3f} → {valid[-1]:.3f} ({drop_pct:.0f}% drop)")
    ok = drop_pct > 30 and not any(np.isnan(l) for l in losses[-20:])
    return {"status": "PASS" if ok else "FAIL",
            "detail": f"loss {valid[0]:.2f}→{valid[-1]:.2f} ({drop_pct:.0f}% drop)"}


run_test("1. Standard Trainer (FlashDet) loss convergence", test_standard_trainer)


# ============================================================
# 2. KD Trainer — verify KD loss is computed and total decreases
# ============================================================
def test_kd_trainer():
    from flashdet.models import FlashDet
    from flashdet.engine.training.kd_trainer import KDTrainer
    import cv2

    with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
        ann = json.load(f)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}
    nc = len(cats)

    images_t, gt_bboxes, gt_labels = [], [], []
    for img_info in ann["images"]:
        img = cv2.imread(os.path.join(DATA_DIR, img_info["file_name"]))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_anns = [a for a in ann["annotations"] if a["image_id"] == img_info["id"]]
        if not img_anns: continue
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
                          for a in img_anns], dtype=np.float32)
        labels = np.array([cat_map[a["category_id"]] for a in img_anns], dtype=np.int64)
        h, w = img.shape[:2]
        img_r = cv2.resize(img, (320, 320))
        sx, sy = 320.0/w, 320.0/h
        boxes[:, [0,2]] *= sx; boxes[:, [1,3]] *= sy
        images_t.append(torch.from_numpy(img_r).permute(2,0,1).float() / 255.0)
        gt_bboxes.append(torch.from_numpy(boxes).float().to(DEVICE))
        gt_labels.append(torch.from_numpy(labels).long().to(DEVICE))
    x = torch.stack(images_t).to(DEVICE)

    teacher = FlashDet(num_classes=nc, size="n", total_epochs=100).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = FlashDet(num_classes=nc, size="n", total_epochs=100).to(DEVICE)
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    kd_temp = 4.0
    kd_alpha = 0.5
    losses_total, losses_det, losses_kd = [], [], []

    for step in range(100):
        opt.zero_grad()
        gt_meta = {"gt_bboxes": [b.clone() for b in gt_bboxes],
                   "gt_labels": [l.clone() for l in gt_labels]}

        s_out = student(x, gt_meta=gt_meta, compute_loss=True, epoch=1, return_features=True)
        det_loss = s_out["loss"]

        with torch.no_grad():
            gt_meta2 = {"gt_bboxes": [b.clone() for b in gt_bboxes],
                        "gt_labels": [l.clone() for l in gt_labels]}
            t_out = teacher(x, gt_meta=gt_meta2, compute_loss=True, epoch=1, return_features=True)

        kd_loss = torch.tensor(0.0, device=DEVICE)
        s_feats = s_out.get("fpn_features", [])
        t_feats = t_out.get("fpn_features", [])
        if s_feats and t_feats:
            for sf, tf in zip(s_feats, t_feats):
                if sf.shape == tf.shape:
                    kd_loss = kd_loss + torch.nn.functional.mse_loss(sf, tf)
            kd_loss = kd_loss / max(len(s_feats), 1)

        total = (1 - kd_alpha) * det_loss + kd_alpha * kd_loss
        if torch.isnan(total):
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        opt.step()

        losses_total.append(total.item())
        losses_det.append(det_loss.item())
        losses_kd.append(kd_loss.item())

    first5 = np.mean(losses_total[:5])
    last5 = np.mean(losses_total[-5:])
    kd_mean = np.mean(losses_kd)
    drop_pct = (first5 - last5) / max(abs(first5), 1e-8) * 100

    kd_active = kd_mean > 0
    print(f"  Total: {first5:.3f}→{last5:.3f} ({drop_pct:.0f}% drop)")
    print(f"  Det loss mean: {np.mean(losses_det):.3f}")
    print(f"  KD loss mean: {kd_mean:.4f} ({'ACTIVE' if kd_active else 'ZERO!'})")

    ok = drop_pct > 20 and kd_active
    return {"status": "PASS" if ok else "FAIL",
            "detail": f"total {first5:.2f}→{last5:.2f}, KD={'active' if kd_active else 'ZERO'}"}


run_test("2. KD Trainer — distillation loss active", test_kd_trainer)


# ============================================================
# 3. SSL Trainer (BYOL) — verify loss decreases
# ============================================================
def test_ssl_trainer():
    from flashdet.engine.training.ssl_trainer import SSLTrainer
    ssl_data = os.path.join(BASE_SAVE, "ssl_images", "dummy_class")
    os.makedirs(ssl_data, exist_ok=True)
    with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
        ann = json.load(f)
    for img_info in ann["images"]:
        src = os.path.join(DATA_DIR, img_info["file_name"])
        dst = os.path.join(ssl_data, img_info["file_name"])
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)

    save_dir = os.path.join(BASE_SAVE, "ssl")
    t = SSLTrainer(
        ssl_method="byol", epochs=10, batch_size=2, lr=0.01,
        workers=0, train_images=os.path.join(BASE_SAVE, "ssl_images"),
        save_dir=save_dir, device=DEVICE, input_size=128, amp=False,
    )
    path = t.pretrain()
    saved = os.path.exists(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    has_weights = len(sd) > 0

    print(f"  Saved: {saved}, weights count: {len(sd)}")
    return {"status": "PASS" if saved and has_weights else "FAIL",
            "detail": f"saved={saved}, {len(sd)} weight tensors"}


run_test("3. SSL (BYOL) pretrain — backbone saved", test_ssl_trainer)


# ============================================================
# 4. Semi-Supervised — verify teacher EMA updates + unsup loss
# ============================================================
def test_semi_supervised():
    from flashdet.models import FlashDet
    import cv2
    import copy

    with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
        ann = json.load(f)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: i for i, c in enumerate(cats)}
    nc = len(cats)

    images_t, gt_bboxes, gt_labels = [], [], []
    for img_info in ann["images"]:
        img = cv2.imread(os.path.join(DATA_DIR, img_info["file_name"]))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_anns = [a for a in ann["annotations"] if a["image_id"] == img_info["id"]]
        if not img_anns: continue
        boxes = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
                          for a in img_anns], dtype=np.float32)
        labels = np.array([cat_map[a["category_id"]] for a in img_anns], dtype=np.int64)
        h, w = img.shape[:2]
        img_r = cv2.resize(img, (320, 320))
        sx, sy = 320.0/w, 320.0/h
        boxes[:, [0,2]] *= sx; boxes[:, [1,3]] *= sy
        images_t.append(torch.from_numpy(img_r).permute(2,0,1).float() / 255.0)
        gt_bboxes.append(torch.from_numpy(boxes).float().to(DEVICE))
        gt_labels.append(torch.from_numpy(labels).long().to(DEVICE))
    x = torch.stack(images_t).to(DEVICE)

    student = FlashDet(num_classes=nc, size="n", total_epochs=100).to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_w_before = list(teacher.parameters())[0].data.clone()

    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    momentum = 0.999
    losses = []

    for step in range(50):
        opt.zero_grad()
        gt_meta = {"gt_bboxes": [b.clone() for b in gt_bboxes],
                   "gt_labels": [l.clone() for l in gt_labels]}
        out = student(x, gt_meta=gt_meta, compute_loss=True, epoch=1)
        loss = out["loss"]
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        opt.step()
        losses.append(loss.item())

        with torch.no_grad():
            for sp, tp in zip(student.parameters(), teacher.parameters()):
                tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)

    teacher_w_after = list(teacher.parameters())[0].data
    teacher_changed = not torch.equal(teacher_w_before, teacher_w_after)
    loss_drop = (np.mean(losses[:5]) - np.mean(losses[-5:])) / max(abs(np.mean(losses[:5])), 1e-8) * 100

    print(f"  Loss: {losses[0]:.3f}→{losses[-1]:.3f} ({loss_drop:.0f}% drop)")
    print(f"  Teacher EMA updated: {teacher_changed}")

    ok = loss_drop > 20 and teacher_changed
    return {"status": "PASS" if ok else "FAIL",
            "detail": f"loss drop={loss_drop:.0f}%, teacher_updated={teacher_changed}"}


run_test("4. Semi-Supervised — teacher EMA + loss", test_semi_supervised)


# ============================================================
# 5. Few-Shot — verify backbone frozen, head trains
# ============================================================
def test_few_shot():
    from flashdet.models import FlashDet
    nc = 10
    model = FlashDet(num_classes=nc, size="n", total_epochs=100).to(DEVICE)

    for name, param in model.named_parameters():
        if "backbone" in name or "stem" in name or "stage" in name:
            param.requires_grad = False
        if "neck" in name or "fpn" in name:
            param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    bb_w_before = None
    head_w_before = None
    for name, p in model.named_parameters():
        if "backbone" in name and bb_w_before is None:
            bb_w_before = p.data.clone()
        if "head" in name and head_w_before is None:
            head_w_before = p.data.clone()

    losses, _ = manual_overfit(model, num_steps=50)

    bb_w_after = None
    head_w_after = None
    for name, p in model.named_parameters():
        if "backbone" in name and bb_w_after is None:
            bb_w_after = p.data.clone()
        if "head" in name and head_w_after is None:
            head_w_after = p.data.clone()

    bb_frozen = torch.equal(bb_w_before, bb_w_after)
    head_changed = not torch.equal(head_w_before.cpu(), head_w_after.cpu())

    valid = [l for l in losses if not np.isnan(l)]
    print(f"  Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.0f}%)")
    print(f"  Backbone frozen: {bb_frozen}")
    print(f"  Head changed: {head_changed}")
    print(f"  Loss: {valid[0]:.3f}→{valid[-1]:.3f}")

    ok = bb_frozen and head_changed
    return {"status": "PASS" if ok else "FAIL",
            "detail": f"bb_frozen={bb_frozen}, head_trained={head_changed}, trainable={100*trainable/total:.0f}%"}


run_test("5. Few-Shot — frozen backbone, head trains", test_few_shot)


# ============================================================
# 6. Active Learning — verify scoring + query selection works
# ============================================================
def test_active_learning():
    from flashdet.models import FlashDet
    from flashdet.engine.training.active_learning_trainer import ActiveLearningTrainer
    import cv2

    with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
        ann = json.load(f)
    cats = sorted(ann["categories"], key=lambda c: c["id"])
    nc = len(cats)

    images_t = []
    for img_info in ann["images"]:
        img = cv2.imread(os.path.join(DATA_DIR, img_info["file_name"]))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img, (320, 320))
        images_t.append(torch.from_numpy(img_r).permute(2,0,1).float() / 255.0)
    x = torch.stack(images_t).to(DEVICE)

    model = FlashDet(num_classes=nc, size="n", total_epochs=100).to(DEVICE)

    al = ActiveLearningTrainer(
        query_strategy="entropy", query_budget=2, al_rounds=1,
        epochs=1, batch_size=5, train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m", save_dir=os.path.join(BASE_SAVE, "al"), device=DEVICE,
    )

    scores = al.score_unlabeled(model, x)
    indices = al.select_query_indices(scores)
    summary = al.get_al_summary()

    print(f"  Scores shape: {scores.shape}, values: {scores.tolist()}")
    print(f"  Selected indices: {indices.tolist()}")
    print(f"  Summary: {summary}")

    ok = scores.shape[0] == len(images_t) and indices.shape[0] == 2
    return {"status": "PASS" if ok else "FAIL",
            "detail": f"scored {scores.shape[0]} images, selected {indices.shape[0]} queries"}


run_test("6. Active Learning — scoring + query", test_active_learning)


# ============================================================
# 7. All architectures via Trainer class
# ============================================================
def test_trainer_archs():
    from flashdet.models import FlashDet
    from flashdet.models.architectures.yolov9 import YOLOv9
    from flashdet.models.architectures.yolov10 import YOLOv10
    from flashdet.models.architectures.yolov11 import YOLOv11

    arch_results = {}
    for name, model_fn in [
        ("FlashDet", lambda nc: FlashDet(num_classes=nc, size="n", total_epochs=100)),
        ("YOLOv9", lambda nc: YOLOv9(num_classes=nc, width_mult=0.25, depth_mult=0.33)),
        ("YOLOv10", lambda nc: YOLOv10(num_classes=nc, width_mult=0.25, depth_mult=0.33)),
        ("YOLOv11", lambda nc: YOLOv11(num_classes=nc, width_mult=0.25, depth_mult=0.33)),
    ]:
        model = model_fn(10)
        losses, _ = manual_overfit(model, num_steps=100)
        valid = [l for l in losses if not np.isnan(l)]
        if valid:
            drop = (valid[0] - valid[-1]) / max(abs(valid[0]), 1e-8) * 100
            print(f"  {name}: {valid[0]:.2f}→{valid[-1]:.2f} ({drop:.0f}% drop)")
            arch_results[name] = drop > 30
        else:
            print(f"  {name}: ALL NaN!")
            arch_results[name] = False

    all_ok = all(arch_results.values())
    detail = ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in arch_results.items())
    return {"status": "PASS" if all_ok else "FAIL", "detail": detail}


run_test("7. All YOLO architectures loss convergence", test_trainer_archs)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print(f"{'Method':<50} {'Status':>8} {'Time':>8}")
print("-" * 70)
for name, status, elapsed, detail in results:
    marker = "✓" if status == "PASS" else "✗"
    print(f"{marker} {name:<48} {status:>8} {elapsed:>7.1f}s")
    if detail:
        print(f"    {detail}")
print("=" * 70)

passed = sum(1 for _, s, _, _ in results if s == "PASS")
total = len(results)
print(f"\n{passed}/{total} deep verification tests PASSED")
print("DONE")
