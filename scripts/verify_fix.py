"""Verify fix: hard labels + proper normalization → scores INCREASE."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.data.dataset import FlashDetDataset, collate_fn
from torch.utils.data import DataLoader, Subset

P = lambda *a, **kw: print(*a, **kw, flush=True)

ds = FlashDetDataset(
    img_dir="data/coco2017/valid",
    ann_file="data/coco2017/valid/_annotations.coco.json",
    input_size=(320, 320)
)
loader = DataLoader(Subset(ds, list(range(4))), batch_size=4, collate_fn=collate_fn, num_workers=0)
images, gt_meta = next(iter(loader))

model = FlashDet(num_classes=80, size="p", total_epochs=100).to(device)

# Check initial targets
model.train()
out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
loss_states = model._last_loss_states
P(f"Initial loss: {out['loss'].item():.4f}")
P(f"o2o_cls: {loss_states['o2o_cls'].item():.4f}, o2o_pos: {loss_states['o2o_pos']}")
P(f"o2m_cls: {loss_states['o2m_cls'].item():.4f}, o2m_pos: {loss_states['o2m_pos']}")

# Check scores before training
model.eval()
with torch.no_grad():
    results = model.predict(images.to(device), None, score_thr=0.001)
    max_before = [r[0][:, 4].max().item() if r[0].shape[0] > 0 else 0 for r in results]
P(f"\nBefore training - max scores: {[round(s, 4) for s in max_before]}")

# Train 30 steps
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for step in range(30):
    optimizer.zero_grad()
    out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
    out['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    if (step + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            results = model.predict(images.to(device), None, score_thr=0.001)
            max_scores = [r[0][:, 4].max().item() if r[0].shape[0] > 0 else 0 for r in results]
        P(f"Step {step+1:3d} - loss: {out['loss'].item():.4f}, max scores: {[round(s, 4) for s in max_scores]}")
        model.train()

# Check assignment targets after some training
from flashdet.losses.e2e_loss import _make_anchor_grid, _decode_ltrb
from flashdet.models.assignment.stal import STALAssigner
model.eval()
with torch.no_grad():
    head_out = model.head(model.neck(model.backbone(images.to(device))), training=True)
    feat_sizes = head_out['feat_sizes']
    centers, strides = _make_anchor_grid(feat_sizes, list(model.strides), device)
    cls_scores = head_out['o2o_cls'].sigmoid()
    decoded = _decode_ltrb(centers, strides, head_out['o2o_reg'][0])

    o2o_assigner = STALAssigner(topk=7, strides=tuple(model.strides), one_to_one=True)
    gt_bboxes = torch.tensor(gt_meta['gt_bboxes'][0], dtype=torch.float32, device=device)
    gt_labels = torch.tensor(gt_meta['gt_labels'][0], dtype=torch.long, device=device)
    _, _, assigned_scores, fg_mask = o2o_assigner.assign(centers, cls_scores[0], decoded, gt_bboxes, gt_labels)
    n_pos = fg_mask.sum().item()
    if n_pos > 0:
        pos_scores = assigned_scores[fg_mask]
        P(f"\no2o target scores (hard labels): min={pos_scores.min():.4f}, max={pos_scores.max():.4f}, mean={pos_scores.mean():.4f}")
        P(f"  (Should be 1.0 for GT class, 0.0 for others)")

# Full 200-step overfit test
P("\n--- 200-step overfit test ---")
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for step in range(200):
    optimizer.zero_grad()
    out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
    out['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    if (step + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            results = model.predict(images.to(device), None, score_thr=0.001)
            max_scores = [r[0][:, 4].max().item() if r[0].shape[0] > 0 else 0 for r in results]
            n_dets = [r[0].shape[0] for r in results]
        P(f"Step {step+1:3d} - loss: {out['loss'].item():.4f}, max scores: {[round(s, 4) for s in max_scores]}, n_dets: {n_dets}")
        model.train()

P("\nFIX VERIFIED" if any(s > 0.3 for s in max_scores) else "\nSTILL BROKEN")
