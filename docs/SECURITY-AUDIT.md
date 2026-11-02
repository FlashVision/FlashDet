# Security Audit — Pen-Tester Notes

**Scope:** FlashDet CLI, Python API, checkpoint load/save, dataset download,
archive extraction and inference input handling.  
**Method:** static review of the repository + targeted exploit tests
(`tests/unit/utils/test_checkpoint_security.py`,
`tests/unit/data/test_download_security.py`,
`tests/unit/engine/test_predictor_validation.py`).  
**Principle:** treat model files, datasets and inference payloads as untrusted
unless the operator explicitly marks them trusted.

---

## 1. Findings

| ID | Severity | Issue | Impact | Status |
|---|---|---|---|---|
| S1 | **Critical** | `torch.load(..., weights_only=False)` in checkpoint/inference/training paths | Arbitrary code execution when loading a malicious `.pth` (pickle RCE, CWE-502) | **Fixed** |
| S2 | **High** | `zipfile.extractall` / `tarfile.extractall` on downloaded archives | Zip Slip / Tar Slip writes or symlinks outside the dataset directory (CWE-22) | **Fixed** |
| S3 | **Medium** | Dataset URLs used plain HTTP | Network attacker could swap dataset archives (CWE-319) | **Fixed** (HTTPS) |
| S4 | **Medium** | Inference accepted arbitrarily large images | Memory/CPU exhaustion from decompression-bomb sized arrays (CWE-400) | **Fixed** |
| S5 | **Low** | Inference input had no type/shape validation | Unclear crashes, grayscale/RGBA images failed unpredictably | **Fixed** |
| S6 | **Low** | Checkpoint config did not always record Pico `backbone_type` / `neck_type` | Wrong backbone could be rebuilt and partially loaded with `strict=False` | **Fixed** |
| S7 | **Info** | Safe loader has an opt-in unsafe mode | `FLASHDET_ALLOW_UNSAFE_LOAD=1` restores pickle RCE for legacy full-module checkpoints | Accepted risk |
| S8 | **Info** | No dependency hashing / `pip-audit` gate in CI | Supply-chain compromise of a dependency is not detected | Backlog |
| S9 | **Info** | No HTTP server in this repo | Authentication, rate limiting and request-size limits must be added by the serving layer | Deployment note |
| S10 | **Info** | Checkpoint size is not bounded | A huge checkpoint can still exhaust memory even with `weights_only=True` | Backlog |

---

## 2. What was changed

### S1 — Safe checkpoint loading

Added `flashdet.utils.checkpoint.safe_torch_load()`:

```python
# safe by default
ckpt = safe_torch_load("model.pth")

# explicit opt-in only for a trusted legacy pickle
FLASHDET_ALLOW_UNSAFE_LOAD=1 python test.py --model legacy_full_module.pth
```

All runtime loaders now use it: inference, validation, export, training resume,
fine-tune, few-shot, analytics benchmark/FLOPs/profiler, `train.py` and `test.py`.
The analytics tools also rebuild a normal model from `config` + state dict via
`load_inference_model()` instead of unpickling a whole module.

### S2 — Archive extraction

`flashdet/data/download.py::_extract` now:

- resolves every member path and rejects anything outside the destination
  (`commonpath` check),
- rejects tar symlinks/hardlinks whose link target escapes the destination,
- uses tarfile's hardened `filter="data"` on Python 3.12+.

Regression tests cover `../` zip members, absolute zip paths, tar path traversal
and tar symlink escapes.

### S3 — HTTPS

COCO and Pascal VOC registry URLs were changed from `http://` to `https://`.
**Residual risk:** there is no per-file SHA-256 pinning yet. Add checksums to the
registry if you need reproducible downloads in a hostile network.

### S4/S5 — Inference input hardening

`Predictor.preprocess` now:

- rejects non-`numpy` inputs,
- converts grayscale and BGRA to BGR,
- rejects non-3-channel arrays,
- rejects zero-size images,
- rejects images above `MAX_INPUT_PIXELS` (64M pixels by default),
- clips/converts non-uint8 arrays.

These are unit-tested.

### S6 — Checkpoint identity

`backbone_type` and `neck_type` are now written into training and inference
checkpoint configs. `test.py`, `flashdet predict`, `flashdet val` and
`flashdet export` rebuild the same Pico backbone/neck. Old Pico-v2 checkpoints
without the field still fall back to state-dict sniffing.

---

## 3. Discussion and improvements

1. **Safe-by-default beats “remember to use weights_only”.** The old code had 12
   call sites; a single forgotten `torch.load` reopens the RCE. One helper with a
   loud opt-in is easier to audit and test.
2. **Security tests are regression tests.** The malicious-pickle test asserts
   that the payload marker file is *not created*. That catches both “loader
   stopped rejecting” and “loader executed the payload” regressions.
3. **Do not trust dataset archives just because they come from a known host.**
   Path traversal is fixed; checksum pinning is the next step. Until then, run
   downloads in a container or CI sandbox.
4. **`strict=False` is a correctness footgun.** It is useful for architecture
   evolution, but inference should report when too many keys are missing. The
   Pico selector fix removes the most common cause of silently loading the wrong
   backbone.
5. **The unsafe escape hatch is deliberate.** Users with legacy whole-module
   checkpoints can still load them, but must set an environment variable. Keep
   that path out of server deployments.

### Recommended backlog

- [ ] Add SHA-256 checksums to every dataset registry entry.
- [ ] Add `pip-audit` / dependency review to CI.
- [ ] Add a checkpoint file-size and tensor-count limit.
- [ ] Add `strict=True` (or missing-key hard failure) for inference loading.
- [ ] If a server is added: auth, rate limits, max upload size, request timeouts,
      and per-request isolation.
- [ ] Fuzz the inference HTTP wrapper with malformed images, oversized JSON and
      concurrent requests.

---

## 4. Reproduction commands

```bash
# Safe loader must refuse a pickle-RCE checkpoint
pytest tests/unit/utils/test_checkpoint_security.py -v

# Zip Slip / Tar Slip must be rejected
pytest tests/unit/data/test_download_security.py -v

# Inference validation
pytest tests/unit/engine/test_predictor_validation.py -v

# Static scan for dangerous patterns
python scripts/security_scan.py
```
