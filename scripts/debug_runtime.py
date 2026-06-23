"""Runtime diagnostic: verify alpha_init fix gives o2o gradients."""
import os, sys, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        ".cursor", "debug-387c01.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log(msg, data, hyp=""):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({"sessionId":"387c01","location":"debug_runtime.py",
                "message":msg,"data":data,"hypothesisId":hyp,
                "timestamp":int(time.time()*1000)}) + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F
from flashdet.models.architectures.flashdet import FlashDet
from flashdet.data.dataset import FlashDetDataset, collate_fn
from torch.utils.data import DataLoader, Subset

if os.path.exists("data/coco2017/valid/_annotations.coco.json"):
    img_dir, nc = "data/coco2017/valid", 80
else:
    img_dir, nc = "data/demo/valid", 10
ann_file = os.path.join(img_dir, "_annotations.coco.json")
ds = FlashDetDataset(img_dir=img_dir, ann_file=ann_file, input_size=(320, 320))
loader = DataLoader(Subset(ds, list(range(4))), batch_size=4, collate_fn=collate_fn, num_workers=0)
images, gt_meta = next(iter(loader))

# === Verify alpha_init fix ===
model = FlashDet(num_classes=nc, size="p", total_epochs=300).to(device)
log("FIX_VERIFY: alpha config", {
    "alpha_init": model.loss_fn.alpha_init,
    "alpha_final": model.loss_fn.alpha_final,
    "alpha_at_epoch0": model.loss_fn.prog_alpha(0, 300),
    "alpha_at_epoch5": model.loss_fn.prog_alpha(5, 300),
    "o2o_weight_epoch0": round(1 - model.loss_fn.prog_alpha(0, 300), 4),
    "o2o_weight_epoch5": round(1 - model.loss_fn.prog_alpha(5, 300), 4),
}, "C")

# === Gradient check with fix ===
model.train()
model.zero_grad()
out = model(images.to(device), gt_meta, epoch=0, compute_loss=True)
out['loss'].backward()

grad_data = {}
for name, param in model.named_parameters():
    if param.grad is not None and ('cls_pred' in name or 'cls_gate' in name):
        grad_data[name] = {
            "grad_norm": round(param.grad.norm().item(), 6),
            "grad_mean": round(param.grad.mean().item(), 6),
        }
log("FIX_VERIFY: o2o gradients after fix", grad_data, "C")

# Check if o2o grads are nonzero
o2o_has_grad = any(
    v["grad_norm"] > 0 for k, v in grad_data.items() if "o2o" in k
)
o2m_has_grad = any(
    v["grad_norm"] > 0 for k, v in grad_data.items() if "o2m" in k
)
log("FIX_VERIFY: gradient presence", {
    "o2o_has_nonzero_gradient": o2o_has_grad,
    "o2m_has_nonzero_gradient": o2m_has_grad,
}, "C")

# === 50-step overfit test with fix ===
log("OVERFIT_TEST: starting 50-step test", {}, "OVERFIT")

model_test = FlashDet(num_classes=nc, size="p", total_epochs=300).to(device)

model_test.eval()
with torch.no_grad():
    r = model_test.predict(images.to(device), None, score_thr=0.001)
    before = [ri[0][:, 4].max().item() if ri[0].shape[0] > 0 else 0 for ri in r]
log("OVERFIT_TEST: scores before", {"max_scores": [round(s, 4) for s in before]}, "OVERFIT")

optimizer = torch.optim.SGD(model_test.parameters(), lr=0.01, momentum=0.9)
for step in range(50):
    model_test.train()
    optimizer.zero_grad()
    out = model_test(images.to(device), gt_meta, epoch=0, compute_loss=True)
    out['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model_test.parameters(), 10.0)
    optimizer.step()
    
    if step in (9, 29, 49):
        model_test.eval()
        with torch.no_grad():
            r = model_test.predict(images.to(device), None, score_thr=0.001)
            scores = [ri[0][:, 4].max().item() if ri[0].shape[0] > 0 else 0 for ri in r]
            n_dets = [ri[0].shape[0] for ri in r]
        log(f"OVERFIT_TEST: scores at step {step+1}", {
            "max_scores": [round(s, 4) for s in scores],
            "n_detections": n_dets,
        }, "OVERFIT")

print(f"Done. Logs at {LOG_PATH}")
