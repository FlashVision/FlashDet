"""Check what scores the training model actually produces."""
import os, sys, torch, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashdet.models.architectures.flashdet import FlashDet
from flashdet.data.dataset import FlashDetDataset, collate_fn
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load latest checkpoint
save_dir = "workspace/flashdet_pico_coco_v6"
ckpts = sorted(glob.glob(f"{save_dir}/*.pt") + glob.glob(f"{save_dir}/*.pth"))
if not ckpts:
    print("No checkpoint found in", save_dir)
    sys.exit(1)
print(f"Loading: {ckpts[-1]}")

model = FlashDet(num_classes=80, size="p", total_epochs=300).to(device)
ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
if "model" in ckpt:
    model.load_state_dict(ckpt["model"], strict=False)
elif "ema" in ckpt:
    model.load_state_dict(ckpt["ema"], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)
print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

# Load val data
ds = FlashDetDataset(img_dir="data/coco2017/valid", ann_file="data/coco2017/valid/_annotations.coco.json", input_size=(320, 320))
loader = DataLoader(Subset(ds, list(range(20))), batch_size=4, collate_fn=collate_fn, num_workers=0)

model.eval()
with torch.no_grad():
    for batch_idx, (images, gt_meta) in enumerate(loader):
        images = images.to(device)

        # Raw forward to check logit magnitudes
        out = model(images)
        o2o_cls = out["o2o_cls"]  # [B, N, 80]
        scores = o2o_cls.sigmoid()
        max_per_anchor = scores.max(dim=-1).values  # [B, N]

        for i in range(images.shape[0]):
            img_scores = max_per_anchor[i]
            n_above_05 = (img_scores > 0.05).sum().item()
            n_above_01 = (img_scores > 0.01).sum().item()
            n_above_001 = (img_scores > 0.001).sum().item()
            print(f"  img {batch_idx*4+i}: max_score={img_scores.max():.4f}, "
                  f">0.05: {n_above_05}, >0.01: {n_above_01}, >0.001: {n_above_001}, "
                  f"logit_range=[{o2o_cls[i].min():.2f}, {o2o_cls[i].max():.2f}]")

        # Also check predict() output
        results = model.predict(images, None, score_thr=0.001)
        for i, (dets, lbls) in enumerate(results):
            n_dets = dets.shape[0]
            ms = dets[:, 4].max().item() if n_dets > 0 else 0
            print(f"    predict@0.001: {n_dets} dets, max_score={ms:.4f}")

        if batch_idx >= 1:
            break

print("\nDONE")
