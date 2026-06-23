"""DEEP DEBUG: Check every step of the FlashDet-Pico pipeline."""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_printoptions(precision=4, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from flashdet.models.architectures.flashdet import FlashDet, FlashDetPico, _decode_batch_nms_free
from flashdet.data.dataset import FlashDetDataset, collate_fn
from flashdet.losses.e2e_loss import _make_anchor_grid, _decode_ltrb, _compute_branch_loss
from flashdet.models.assignment.stal import STALAssigner
from torch.utils.data import DataLoader, Subset

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ===== 1. DATA FORMAT CHECK =====
P("\n" + "="*60)
P("1. DATA FORMAT CHECK")
P("="*60)

ds = FlashDetDataset(
    img_dir="data/coco2017/valid", 
    ann_file="data/coco2017/valid/_annotations.coco.json",
    input_size=(320, 320)
)
loader = DataLoader(Subset(ds, list(range(4))), batch_size=4, collate_fn=collate_fn, num_workers=0)
images, gt_meta = next(iter(loader))

P(f"  Image shape: {images.shape}, dtype={images.dtype}")
P(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
P(f"  Image mean per channel: {images.mean(dim=(0,2,3))}")

for i in range(min(2, len(gt_meta['gt_bboxes']))):
    boxes = gt_meta['gt_bboxes'][i]
    labels = gt_meta['gt_labels'][i]
    P(f"  img {i}: boxes type={type(boxes).__name__}, shape={boxes.shape if hasattr(boxes,'shape') else len(boxes)}")
    P(f"  img {i}: labels type={type(labels).__name__}, dtype={labels.dtype if hasattr(labels,'dtype') else '?'}")
    if len(boxes) > 0:
        b = np.array(boxes)
        P(f"  img {i}: box[0] = {b[0]} (xyxy format)")
        P(f"  img {i}: box range x=[{b[:,0].min():.1f},{b[:,2].max():.1f}], y=[{b[:,1].min():.1f},{b[:,3].max():.1f}]")
        P(f"  img {i}: SHOULD BE in [0, 320] range? {'YES' if b[:,2].max() <= 320 else 'NO - PROBLEM!'}")
        P(f"  img {i}: labels = {labels[:5]}, range=[{labels.min()},{labels.max()}], unique={len(np.unique(labels))}")

# ===== 2. MODEL ARCHITECTURE CHECK =====
P("\n" + "="*60)
P("2. MODEL ARCHITECTURE CHECK")
P("="*60)

model = FlashDet(num_classes=80, size="p", total_epochs=100).to(device)
P(f"  Backbone type: {type(model.backbone).__name__}")
P(f"  Neck type: {type(model.neck).__name__}")
P(f"  Head type: {type(model.head).__name__}")
P(f"  Strides: {model.strides}")

# Check feature map sizes
model.eval()
with torch.no_grad():
    x = images.to(device)
    feats = model.backbone(x)
    P(f"  Backbone outputs: {[f.shape for f in feats]}")
    neck_feats = model.neck(feats)
    P(f"  Neck outputs: {[f.shape for f in neck_feats]}")
    head_out = model.head(neck_feats, training=False)
    P(f"  o2o_cls shape: {head_out['o2o_cls'].shape}")
    P(f"  o2o_reg shape: {head_out['o2o_reg'].shape}")
    P(f"  feat_sizes: {head_out['feat_sizes']}")

# ===== 3. ANCHOR GRID CHECK =====
P("\n" + "="*60)
P("3. ANCHOR GRID CHECK")
P("="*60)

feat_sizes = head_out['feat_sizes']
centers, strides = _make_anchor_grid(feat_sizes, list(model.strides), device)
P(f"  Total anchors: {centers.shape[0]}")
P(f"  Centers range: x=[{centers[:,0].min():.1f},{centers[:,0].max():.1f}], y=[{centers[:,1].min():.1f},{centers[:,1].max():.1f}]")
P(f"  Strides: {strides.unique().tolist()}")

for (h, w), s in zip(feat_sizes, model.strides):
    n = h * w
    P(f"    Level stride={s}: feat={h}x{w}, anchors={n}, center_range=[{s*0.5:.1f}, {(max(h,w)-0.5)*s:.1f}]")

# Check: do GT boxes overlap with anchor centers?
b0 = torch.tensor(gt_meta['gt_bboxes'][0], dtype=torch.float32, device=device)
if len(b0) > 0:
    gt_cx = (b0[:, 0] + b0[:, 2]) / 2
    gt_cy = (b0[:, 1] + b0[:, 3]) / 2
    P(f"  GT box centers: x=[{gt_cx.min():.1f},{gt_cx.max():.1f}], y=[{gt_cy.min():.1f},{gt_cy.max():.1f}]")
    # Check if any anchor center falls inside GT boxes
    for gi in range(min(3, len(b0))):
        inside = ((centers[:, 0] > b0[gi, 0]) & (centers[:, 0] < b0[gi, 2]) &
                  (centers[:, 1] > b0[gi, 1]) & (centers[:, 1] < b0[gi, 3]))
        P(f"  GT[{gi}] box={b0[gi].cpu().numpy()}, anchors_inside={inside.sum().item()}")

# ===== 4. LOSS / ASSIGNMENT CHECK =====
P("\n" + "="*60)
P("4. LOSS / ASSIGNMENT CHECK (o2o with 1:1)")
P("="*60)

model.train()
out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
P(f"  Total loss: {out['loss'].item():.4f}")

loss_states = model._last_loss_states
for k, v in loss_states.items():
    val = v.item() if hasattr(v, 'item') else v
    P(f"  {k}: {val}")

# Manual assignment check
o2o_assigner = STALAssigner(topk=7, strides=tuple(model.strides), one_to_one=True)
o2m_assigner = STALAssigner(topk=10, strides=tuple(model.strides))

model.eval()
with torch.no_grad():
    head_out = model.head(model.neck(model.backbone(images.to(device))), training=True)
    o2o_cls = head_out['o2o_cls']
    o2o_reg = head_out['o2o_reg']

    for bi in range(min(2, images.shape[0])):
        gt_bboxes = torch.tensor(gt_meta['gt_bboxes'][bi], dtype=torch.float32, device=device).reshape(-1, 4)
        gt_labels = torch.tensor(gt_meta['gt_labels'][bi], dtype=torch.long, device=device).reshape(-1)
        
        decoded = _decode_ltrb(centers, strides, o2o_reg[bi])
        cls_scores = o2o_cls[bi].sigmoid()
        
        P(f"\n  --- Image {bi}: {len(gt_labels)} GT objects ---")
        P(f"  Decoded box range: [{decoded.min():.1f}, {decoded.max():.1f}]")
        P(f"  Cls scores range: [{cls_scores.min():.4f}, {cls_scores.max():.4f}]")
        P(f"  Cls scores mean: {cls_scores.mean():.6f}")
        
        # o2o assignment
        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = o2o_assigner.assign(
            centers, cls_scores, decoded, gt_bboxes, gt_labels
        )
        n_pos = fg_mask.sum().item()
        P(f"  o2o: fg_mask={n_pos}/{centers.shape[0]}, assigned_labels unique={assigned_labels[fg_mask].unique().tolist() if n_pos > 0 else '[]'}")
        
        if n_pos > 0:
            pos_scores = assigned_scores[fg_mask]
            P(f"  o2o: target score range=[{pos_scores.min():.4f}, {pos_scores.max():.4f}], mean={pos_scores.mean():.4f}")
            P(f"  o2o: target score sum={pos_scores.sum():.4f}")
            
            # Check IoU between assigned boxes and GT
            from flashdet.models.assignment.stal import _pairwise_iou
            pos_decoded = decoded[fg_mask]
            pos_target = assigned_bboxes[fg_mask]
            iou_diag = []
            for pi in range(min(5, n_pos)):
                iou = _pairwise_iou(pos_decoded[pi:pi+1], pos_target[pi:pi+1])[0, 0].item()
                iou_diag.append(round(iou, 3))
            P(f"  o2o: pred-GT IoU (first 5 pos): {iou_diag}")
        
        # o2m assignment for comparison
        _, _, o2m_scores, o2m_fg = o2m_assigner.assign(
            centers, cls_scores, decoded, gt_bboxes, gt_labels
        )
        P(f"  o2m: fg_mask={o2m_fg.sum().item()}/{centers.shape[0]}")

# ===== 5. GRADIENT FLOW CHECK =====
P("\n" + "="*60)
P("5. GRADIENT FLOW CHECK")
P("="*60)

model.train()
model.zero_grad()
out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
out['loss'].backward()

P("  Gradient norms per module:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if 'cls_pred' in name or 'reg_pred' in name or 'weight' in name:
            if grad_norm < 1e-8:
                P(f"    *** DEAD GRADIENT: {name}: grad_norm={grad_norm:.2e}, param_norm={param.norm().item():.4f}")
            elif 'cls_pred' in name or 'reg_pred' in name:
                P(f"    {name}: grad_norm={grad_norm:.4e}, param_norm={param.norm().item():.4f}")

# Check specific cls_pred biases
for level_idx, head in enumerate(model.head.o2o_heads):
    bias = head.cls_pred.bias
    P(f"  o2o head[{level_idx}] cls_pred.bias: mean={bias.data.mean():.4f}, grad_mean={bias.grad.mean():.6f}")
    weight = head.cls_pred.weight
    P(f"  o2o head[{level_idx}] cls_pred.weight: norm={weight.data.norm():.4f}, grad_norm={weight.grad.norm():.4e}")

# ===== 6. FORWARD-BACKWARD STEP + SCORE CHECK =====
P("\n" + "="*60)
P("6. ONE TRAINING STEP - CHECK IF SCORES INCREASE")
P("="*60)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model.train()

# Before step
model.eval()
with torch.no_grad():
    results_before = model.predict(images.to(device), None, score_thr=0.001)
    max_scores_before = [r[0][:, 4].max().item() if r[0].shape[0] > 0 else 0 for r in results_before]
P(f"  Before step - max scores: {[round(s, 4) for s in max_scores_before]}")

# Do 10 gradient steps
model.train()
for step in range(10):
    optimizer.zero_grad()
    out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
    out['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

# After steps
model.eval()
with torch.no_grad():
    results_after = model.predict(images.to(device), None, score_thr=0.001)
    max_scores_after = [r[0][:, 4].max().item() if r[0].shape[0] > 0 else 0 for r in results_after]
P(f"  After 10 steps - max scores: {[round(s, 4) for s in max_scores_after]}")
P(f"  Score change: {[round(a-b, 4) for a, b in zip(max_scores_after, max_scores_before)]}")

# Check raw logits after training
with torch.no_grad():
    head_out = model.head(model.neck(model.backbone(images.to(device))), training=False)
    logits = head_out['o2o_cls']
    P(f"  Raw logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    P(f"  Raw logit mean: {logits.mean():.3f}")
    scores = logits.sigmoid()
    P(f"  Sigmoid score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    for bi in range(min(2, images.shape[0])):
        max_per_anchor = scores[bi].max(dim=1).values
        gt_labels_bi = gt_meta['gt_labels'][bi]
        gt_bboxes_bi = gt_meta['gt_bboxes'][bi]
        P(f"  img {bi}: {len(gt_labels_bi)} GTs, top5 anchor scores: {max_per_anchor.topk(5).values.cpu().tolist()}")
        
        # Check scores at anchor positions closest to GT centers
        if len(gt_bboxes_bi) > 0:
            gb = torch.tensor(gt_bboxes_bi, dtype=torch.float32, device=device)
            gt_cx = (gb[:, 0] + gb[:, 2]) / 2
            gt_cy = (gb[:, 1] + gb[:, 3]) / 2
            for gi in range(min(3, len(gb))):
                dist = ((centers[:, 0] - gt_cx[gi])**2 + (centers[:, 1] - gt_cy[gi])**2).sqrt()
                nearest = dist.argmin()
                gt_cls = int(gt_labels_bi[gi])
                score_at_gt = scores[bi, nearest, gt_cls].item()
                max_score_at_anchor = scores[bi, nearest].max().item()
                P(f"    GT[{gi}] cls={gt_cls}: nearest_anchor score_at_gt_cls={score_at_gt:.4f}, max_score={max_score_at_anchor:.4f}")

# ===== 7. LTRB DECODE CHECK =====
P("\n" + "="*60)
P("7. LTRB DECODE CHECK")
P("="*60)
with torch.no_grad():
    reg = head_out['o2o_reg']
    decoded = _decode_ltrb(centers, strides, reg[0])
    P(f"  Raw reg range: [{reg.min():.3f}, {reg.max():.3f}]")
    P(f"  After softplus: [{torch.nn.functional.softplus(reg).min():.3f}, {torch.nn.functional.softplus(reg).max():.3f}]")
    P(f"  Decoded boxes range: [{decoded.min():.1f}, {decoded.max():.1f}]")
    P(f"  Box widths range: [{(decoded[:,2]-decoded[:,0]).min():.1f}, {(decoded[:,2]-decoded[:,0]).max():.1f}]")
    P(f"  Box heights range: [{(decoded[:,3]-decoded[:,1]).min():.1f}, {(decoded[:,3]-decoded[:,1]).max():.1f}]")

P("\n" + "="*60)
P("DEEP DEBUG COMPLETE")
P("="*60)
