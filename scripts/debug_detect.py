"""Quick debug: overfit on 10 images, verify o2o 1:1 fix and mAP."""
import os, sys, torch, numpy as np, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("debug")

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.data.dataset import FlashDetDataset, collate_fn
from flashdet.utils.metrics import compute_map, compute_iou
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", device)

ann_file = "data/coco2017/valid/_annotations.coco.json"
img_dir = "data/coco2017/valid"

ds = FlashDetDataset(img_dir=img_dir, ann_file=ann_file, input_size=(320, 320))
tiny = Subset(ds, list(range(min(10, len(ds)))))
loader = DataLoader(tiny, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

log.info("\n=== OVERFIT TEST — Pico with o2o 1:1 fix (100 epochs) ===")
model = FlashDet(num_classes=80, size="p", total_epochs=100).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(100):
    model.train()
    epoch_loss = 0
    n_batches = 0
    for images, gt_meta in loader:
        images = images.to(device)
        out = model(images, gt_meta, epoch=epoch, compute_loss=True)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, gt_meta in loader:
                images = images.to(device)
                results = model.predict(images, None, score_thr=0.05)
                for i, (dets, lbls) in enumerate(results):
                    gt_boxes = gt_meta["gt_bboxes"][i]
                    gt_labels = gt_meta["gt_labels"][i]

                    if dets.numel() > 0:
                        boxes_np = dets[:, :4].cpu().numpy()
                        scores_np = dets[:, 4].cpu().numpy()
                        lbs_np = lbls.cpu().numpy()
                    else:
                        boxes_np = np.zeros((0, 4), dtype=np.float32)
                        scores_np = np.zeros(0, dtype=np.float32)
                        lbs_np = np.zeros(0, dtype=np.int64)

                    all_preds.append({"boxes": boxes_np, "scores": scores_np, "labels": lbs_np})
                    all_gts.append({"boxes": gt_boxes, "labels": gt_labels})

        map_res = compute_map(all_preds, all_gts, iou_threshold=0.5, num_classes=80)
        n_dets = sum(p["boxes"].shape[0] for p in all_preds)
        n_gts = sum(len(g["labels"]) for g in all_gts)

        all_scores = np.concatenate([p["scores"] for p in all_preds if len(p["scores"])>0]) if n_dets>0 else np.array([])
        max_score = all_scores.max() if len(all_scores) > 0 else 0

        all_pred_labels = np.concatenate([p["labels"] for p in all_preds if len(p["labels"])>0]) if n_dets > 0 else np.array([])
        unique_cls = len(np.unique(all_pred_labels)) if len(all_pred_labels) > 0 else 0

        log.info("  Epoch %3d: loss=%.4f, dets=%d/%d_gt, max_score=%.4f, unique_cls=%d, mAP=%.4f",
                 epoch+1, avg_loss, n_dets, n_gts, max_score, unique_cls, map_res["mAP"])

log.info("\n=== FINAL DETAILED CHECK ===")
model.eval()
with torch.no_grad():
    images, gt_meta = next(iter(loader))
    images = images.to(device)

    results = model.predict(images, None, score_thr=0.01)
    for i in range(min(3, len(results))):
        dets, lbls = results[i]
        gt_boxes = gt_meta["gt_bboxes"][i]
        gt_labels = gt_meta["gt_labels"][i]
        log.info("Image %d:", i)
        log.info("  GT: %d boxes, labels=%s", len(gt_labels), gt_labels[:8] if len(gt_labels) > 8 else gt_labels)
        log.info("  Pred: %d dets", dets.shape[0])
        if dets.shape[0] > 0:
            log.info("  Top5 scores: %s", [round(x, 4) for x in dets[:5, 4].cpu().tolist()])
            log.info("  Top5 labels: %s", lbls[:5].cpu().tolist())
            if len(gt_boxes) > 0:
                for pi in range(min(5, dets.shape[0])):
                    pbox = dets[pi, :4].cpu().numpy()
                    best_iou, best_gt_cls = 0, -1
                    for gi in range(len(gt_boxes)):
                        gbox = np.array(gt_boxes[gi]) if not hasattr(gt_boxes[gi], 'numpy') else gt_boxes[gi]
                        iou = compute_iou(pbox, np.array(gbox))
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_cls = gt_labels[gi]
                    log.info("    Pred[%d] score=%.3f cls=%d iou=%.3f gt_cls=%s",
                             pi, dets[pi, 4], lbls[pi], best_iou, best_gt_cls)

log.info("\nDONE")
