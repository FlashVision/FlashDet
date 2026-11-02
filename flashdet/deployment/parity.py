"""PyTorch ↔ ONNXRuntime parity check for exported models.

Compares the Python-side wrapper output with the ONNX graph output on the
same batches so the export is verified numerically instead of assumed.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from flashdet.deployment.onnx_export import _ExportWrapper

logger = logging.getLogger(__name__)


def onnx_input_shape(onnx_path: str) -> tuple[int, int, int]:
    """Read spatial input size from the ONNX graph (falls back to 320)."""
    import onnx

    model = onnx.load(onnx_path)
    inputs = model.graph.input[0]
    dims = [d.dim_value for d in inputs.type.tensor_type.shape.dim]
    if len(dims) >= 4 and dims[2] > 0:
        return int(dims[1]), int(dims[2]), int(dims[3])
    raise ValueError(
        "ONNX input has dynamic spatial size; pass a batch explicitly"
    )


def detection_parity(model_path: str, onnx_path: str,
                     num_images: int = 2, conf: float = 0.05,
                     max_det: int = 50) -> dict:
    """Decode boxes from a PyTorch checkpoint and an ONNX file on the same
    random batches and report the matched IoU.

    This is the deployment-relevant metric: tensor-level diffs can look large
    while boxes stay almost identical (and vice versa).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required for parity checks: pip install onnxruntime"
        ) from exc

    from flashdet.utils.bbox import decode_batch_nms_free
    from flashdet.deployment.onnx_export import build_model_from_checkpoint

    model = build_model_from_checkpoint(model_path, fuse=True).eval()
    num_classes = int(model.num_classes)
    in_shape = onnx_input_shape(onnx_path)

    generator = torch.Generator().manual_seed(0)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ious: list[float] = []
    n_pytorch, n_onnx = 0, 0
    with torch.no_grad():
        for _ in range(num_images):
            x = torch.randn(1, in_shape[0], in_shape[1], in_shape[2],
                            generator=generator)
            feats = model.forward(x)
            anchors = model._get_anchors(feats["feat_sizes"], x.device)

            ref = model.predict(
                x, None, score_thr=conf, max_det=max_det,
            )[0][0]
            ref_boxes = np.asarray(ref)[:, :4]

            out = session.run(
                None, {"images": x.numpy().astype(np.float32)}
            )[0]
            dets = decode_batch_nms_free(
                torch.from_numpy(out[..., :num_classes]).float(),
                torch.from_numpy(out[..., num_classes:]).float(),
                anchors[0], anchors[1],
                img_hw=(in_shape[1], in_shape[2]),
                score_thr=conf, max_det=max_det,
            )[0]
            boxes = np.asarray(dets[0])[:, :4]
            n_pytorch += len(ref_boxes)
            n_onnx += len(boxes)
            for rb in ref_boxes:
                x1 = np.maximum(boxes[:, 0], rb[0])
                y1 = np.maximum(boxes[:, 1], rb[1])
                x2 = np.minimum(boxes[:, 2], rb[2])
                y2 = np.minimum(boxes[:, 3], rb[3])
                inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                a1 = (rb[2] - rb[0]) * (rb[3] - rb[1])
                a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                union = a1 + a2 - inter
                iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
                if iou.size:
                    ious.append(float(iou.max()))

    mean_iou = float(np.mean(ious)) if ious else 0.0
    min_iou = float(np.min(ious)) if ious else 0.0
    summary = {
        "mean_box_iou": mean_iou,
        "min_box_iou": min_iou,
        "num_pytorch_boxes": n_pytorch,
        "num_onnx_boxes": n_onnx,
    }
    logger.info(
        "Detection parity: mean IoU %.3f (min %.3f) — %d/%d boxes",
        mean_iou, min_iou, n_pytorch, n_onnx,
    )
    return summary


def check_parity(model_path: str, onnx_path: str,
                 batch: torch.Tensor | None = None,
                 rtol: float = 1e-2, atol: float = 5e-3) -> dict:
    """Compare a PyTorch checkpoint and an ONNX export on the same batches.

    Returns a dict with max/mean absolute difference and whether the two
    agree within ``rtol``/``atol``.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required for parity checks: pip install onnxruntime"
        ) from exc

    from flashdet.deployment.onnx_export import build_model_from_checkpoint

    torch_module = build_model_from_checkpoint(model_path, fuse=True)
    wrapper = _ExportWrapper(torch_module).eval()

    if batch is None:
        in_shape = onnx_input_shape(onnx_path)
        batch = torch.randn(2, in_shape[0], in_shape[1], in_shape[2])

    with torch.no_grad():
        reference = wrapper(batch).numpy().astype(np.float32)

    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {"images": batch.numpy().astype(np.float32)})[0]

    diff = np.abs(result - reference)
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    ok = np.allclose(result, reference, rtol=rtol, atol=atol)
    summary = {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "ok": bool(ok),
        "output_shape": list(reference.shape),
        "onnx_shape": list(result.shape),
    }
    state = "matches" if ok else "OUT OF TOLERANCE"
    logger.info("Parity %s — max|Δ|=%.5f mean|Δ|=%.6f", state, max_diff, mean_diff)
    if not ok:
        diff_mass = float(np.abs(reference).mean()) or 1.0
        logger.warning(
            "Relative diff %.4f%% exceeds tolerance (atol=%s, rtol=%s)",
            mean_diff / diff_mass * 100, atol, rtol,
        )
    return summary
