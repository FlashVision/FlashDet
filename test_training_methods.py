"""Test all training methods end-to-end on overfit5 dataset.

Runs each trainer for a few epochs and verifies it completes without errors.
"""
import os
import sys
import time
import shutil
import traceback

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/overfit5/train"
BASE_SAVE = "workspace/test_methods"

os.makedirs(BASE_SAVE, exist_ok=True)

results = []


def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"[{name}]")
    print(f"{'='*60}", flush=True)
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"  PASS ({elapsed:.1f}s) — {result}")
        results.append((name, "PASS", elapsed, str(result)[:80]))
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAIL ({elapsed:.1f}s) — {e}")
        traceback.print_exc()
        results.append((name, "FAIL", elapsed, str(e)[:80]))


# ============================================================
# 1. Standard Trainer (5 epochs)
# ============================================================
def test_standard_trainer():
    from flashdet.engine.training.trainer import Trainer
    save_dir = os.path.join(BASE_SAVE, "standard")
    t = Trainer(
        epochs=5, batch_size=5, lr=0.001, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m", architecture="flashdet",
        save_dir=save_dir, device=DEVICE, patience=0,
    )
    result = t.train()
    assert os.path.exists(os.path.join(save_dir, "model_final_inference.pth")), "No final model saved"
    return result


run_test("1. Standard Trainer (5 epochs)", test_standard_trainer)


# ============================================================
# 2. Standard Trainer with different architectures
# ============================================================
def test_trainer_yolov9():
    from flashdet.engine.training.trainer import Trainer
    save_dir = os.path.join(BASE_SAVE, "yolov9")
    t = Trainer(
        epochs=3, batch_size=5, lr=0.001, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m", architecture="yolov9",
        save_dir=save_dir, device=DEVICE, patience=0,
    )
    return t.train()


run_test("2. Trainer + YOLOv9 arch", test_trainer_yolov9)


# ============================================================
# 3. Knowledge Distillation Trainer
# ============================================================
def test_kd_trainer():
    from flashdet.engine.training.kd_trainer import KDTrainer
    teacher_ckpt = os.path.join(BASE_SAVE, "standard", "model_final_inference.pth")
    if not os.path.exists(teacher_ckpt):
        return "SKIP — no teacher checkpoint from test 1"
    save_dir = os.path.join(BASE_SAVE, "kd")
    t = KDTrainer(
        teacher_checkpoint=teacher_ckpt,
        teacher_size="m",
        kd_temperature=4.0,
        kd_alpha=0.5,
        epochs=3, batch_size=5, lr=0.001, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m",
        save_dir=save_dir, device=DEVICE, patience=0,
    )
    return t.train()


run_test("3. KD Trainer (teacher→student)", test_kd_trainer)


# ============================================================
# 4. SSL Trainer (BYOL pretrain)
# ============================================================
def test_ssl_trainer():
    from flashdet.engine.training.ssl_trainer import SSLTrainer
    ssl_data = os.path.join(BASE_SAVE, "ssl_images", "dummy_class")
    os.makedirs(ssl_data, exist_ok=True)

    import cv2
    import json
    with open(os.path.join(DATA_DIR, "_annotations.coco.json")) as f:
        ann = json.load(f)
    for img_info in ann["images"]:
        src = os.path.join(DATA_DIR, img_info["file_name"])
        dst = os.path.join(ssl_data, img_info["file_name"])
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)

    save_dir = os.path.join(BASE_SAVE, "ssl")
    t = SSLTrainer(
        ssl_method="byol",
        epochs=3,
        batch_size=2,
        lr=0.01,
        workers=0,
        train_images=os.path.join(BASE_SAVE, "ssl_images"),
        save_dir=save_dir,
        device=DEVICE,
        input_size=128,
        amp=False,
    )
    path = t.pretrain()
    assert os.path.exists(path), f"Pretrained backbone not saved at {path}"
    return f"saved={path}"


run_test("4. SSL Trainer (BYOL, 3 epochs)", test_ssl_trainer)


# ============================================================
# 5. Semi-Supervised Trainer
# ============================================================
def test_semi_supervised_trainer():
    from flashdet.engine.training.semi_supervised_trainer import SemiSupervisedTrainer
    save_dir = os.path.join(BASE_SAVE, "semi_sup")
    t = SemiSupervisedTrainer(
        unlabeled_images=DATA_DIR,
        pseudo_label_threshold=0.7,
        unsup_loss_weight=1.0,
        warmup_teacher_epochs=1,
        epochs=3, batch_size=5, lr=0.001, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m",
        save_dir=save_dir, device=DEVICE, patience=0,
    )
    return t.train()


run_test("5. Semi-Supervised Trainer (3 epochs)", test_semi_supervised_trainer)


# ============================================================
# 6. Few-Shot Trainer
# ============================================================
def test_few_shot_trainer():
    from flashdet.engine.training.few_shot_trainer import FewShotTrainer
    base_ckpt = os.path.join(BASE_SAVE, "standard", "model_final_inference.pth")
    if not os.path.exists(base_ckpt):
        return "SKIP — no base checkpoint from test 1"
    save_dir = os.path.join(BASE_SAVE, "few_shot")
    t = FewShotTrainer(
        base_checkpoint=base_ckpt,
        n_shot=5,
        freeze_backbone=True,
        freeze_neck=True,
        head_lr_factor=10.0,
        epochs=3, batch_size=5, lr=0.0005, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m",
        save_dir=save_dir, device=DEVICE,
    )
    return t.train()


run_test("6. Few-Shot Trainer (5-shot, 3 epochs)", test_few_shot_trainer)


# ============================================================
# 7. Active Learning Trainer
# ============================================================
def test_active_learning_trainer():
    from flashdet.engine.training.active_learning_trainer import ActiveLearningTrainer
    save_dir = os.path.join(BASE_SAVE, "active_learning")
    t = ActiveLearningTrainer(
        unlabeled_pool=DATA_DIR,
        query_strategy="entropy",
        query_budget=2,
        al_rounds=2,
        epochs=2, batch_size=5, lr=0.001, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m",
        save_dir=save_dir, device=DEVICE, patience=0,
    )
    return t.train()


run_test("7. Active Learning Trainer (2 rounds x 2 epochs)", test_active_learning_trainer)


# ============================================================
# 8. Trainer with MuSGD optimizer
# ============================================================
def test_musgd():
    from flashdet.engine.training.trainer import Trainer
    save_dir = os.path.join(BASE_SAVE, "musgd")
    t = Trainer(
        epochs=3, batch_size=5, lr=0.001, workers=0,
        train_images=DATA_DIR, val_images=DATA_DIR,
        model_size="m", architecture="flashdet",
        save_dir=save_dir, device=DEVICE, patience=0,
    )
    result = t.train()
    return result


run_test("8. Trainer + MuSGD", test_musgd)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print(f"{'Method':<45} {'Status':>8} {'Time':>8}")
print("-" * 70)
for name, status, elapsed, detail in results:
    print(f"{name:<45} {status:>8} {elapsed:>7.1f}s")
    if status == "FAIL":
        print(f"  → {detail}")
print("=" * 70)

passed = sum(1 for _, s, _, _ in results if s == "PASS")
total = len(results)
print(f"\n{passed}/{total} training methods PASSED")
print("DONE")
