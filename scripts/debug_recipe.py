"""Test training recipe on 200 images with SGD (matching real config)."""
import os, sys, torch, numpy as np, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("debug_recipe")

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.data.dataset import FlashDetDataset, collate_fn
from flashdet.utils.metrics import compute_map
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", device)

ann_file = "data/coco2017/valid/_annotations.coco.json"
img_dir = "data/coco2017/valid"
input_size = (320, 320)
num_classes = 80
total_epochs = 50

train_ds = FlashDetDataset(img_dir=img_dir, ann_file=ann_file, input_size=input_size)
val_ds = FlashDetDataset(img_dir=img_dir, ann_file=ann_file, input_size=input_size)

train_indices = list(range(min(200, len(train_ds))))
val_indices = list(range(min(50, len(val_ds))))
train_sub = Subset(train_ds, train_indices)
val_sub = Subset(val_ds, val_indices)

train_loader = DataLoader(train_sub, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_sub, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2)

log.info("Train: %d images, Val: %d images", len(train_sub), len(val_sub))

model = FlashDet(num_classes=num_classes, size="p", total_epochs=total_epochs).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

for epoch in range(total_epochs):
    model.train()
    epoch_loss = 0
    n_batches = 0
    for images, gt_meta in train_loader:
        images = images.to(device)
        out = model(images, gt_meta, epoch=epoch, compute_loss=True)
        loss = out["loss"]
        if torch.isnan(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, gt_meta in val_loader:
                images = images.to(device)
                results = model.predict(images, None, score_thr=0.05)
                for i, (dets, lbls) in enumerate(results):
                    gt_boxes = gt_meta["gt_bboxes"][i]
                    gt_labels = gt_meta["gt_labels"][i]
                    if dets.numel() > 0:
                        all_preds.append({"boxes": dets[:, :4].cpu().numpy(), "scores": dets[:, 4].cpu().numpy(), "labels": lbls.cpu().numpy()})
                    else:
                        all_preds.append({"boxes": np.zeros((0, 4), dtype=np.float32), "scores": np.zeros(0, dtype=np.float32), "labels": np.zeros(0, dtype=np.int64)})
                    all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

        map_res = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=num_classes)
        n_dets = sum(p["boxes"].shape[0] for p in all_preds)
        n_gts = sum(len(g["labels"]) for g in all_gts)
        all_scores = np.concatenate([p["scores"] for p in all_preds if len(p["scores"]) > 0]) if n_dets > 0 else np.array([])
        max_score = all_scores.max() if len(all_scores) > 0 else 0

        all_pred_labels = np.concatenate([p["labels"] for p in all_preds if len(p["labels"]) > 0]) if n_dets > 0 else np.array([])
        n_unique = len(np.unique(all_pred_labels)) if len(all_pred_labels) > 0 else 0

        log.info("Epoch %2d: loss=%.4f, dets=%d/%d_gt, max_score=%.3f, cls=%d, mAP=%.4f",
                 epoch + 1, avg_loss, n_dets, n_gts, max_score, n_unique, map_res["mAP"])

log.info("DONE — recipe test complete")
