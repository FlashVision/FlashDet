"""Weight registry and downloader for FlashDet checkpoints.

Each entry points at a checksum-verified GitHub release asset so users can
pull a demo model in one command instead of training first.
"""

from __future__ import annotations

import hashlib
import logging
import os
from urllib.parse import urlparse
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

WEIGHTS: dict[str, dict] = {
    "p-demo-synth10": {
        "url": "https://github.com/FlashVision/FlashDet/releases/download/demo-weights-1.0/flashdet_p_demo_synth10_fp16.pth",
        "sha256": "e94525e38af48abdfef926346ce9036ff4b589514cbce7cc6a0b93c150ed4733",
        "description": (
            "FlashDet-P demo (369K params, FP16) — 12 epochs on a 10-class "
            "synthetic dataset, mAP@0.5 ~= 0.60. Pipeline demo, not an "
            "accuracy reference."
        ),
        "num_classes": 10,
    },
}

DEFAULT_SAVE_DIR = os.path.join("weights")


def list_weights() -> list[str]:
    """Available registry entries."""
    return sorted(WEIGHTS)


def _expected_sha256(item: dict) -> str | None:
    return item.get("sha256") or None


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_weights(name: str, save_dir: str | None = None,
                     force: bool = False) -> str:
    """Download a registry checkpoint and verify its checksum when known."""
    if name not in WEIGHTS:  # no mirror of arbitrary URLs
        raise KeyError(
            f"Unknown weights name {name!r}; available: {', '.join(list_weights())}"
        )
    item = WEIGHTS[name]
    target_dir = save_dir or DEFAULT_SAVE_DIR
    os.makedirs(target_dir, exist_ok=True)

    filename = "flashdet_" + name + ".pth"
    if os.path.basename(filename) != filename:
        raise ValueError("Invalid weights file name")
    path = os.path.join(target_dir, filename)

    url = urlparse(item["url"])
    expected = _expected_sha256(item)

    if os.path.isfile(path) and not force:
        digest = sha256_file(path)
        if expected is None or digest == expected:
            logger.info("Checkpoint already present: %s", path)
            return path
        logger.warning("Checksum mismatch on existing file — re-downloading")

    logger.info("Downloading %s -> %s", url.geturl(), path)
    urlretrieve(url.geturl(), path)

    actual = sha256_file(path)
    if expected is not None and actual != expected:
        os.remove(path)
        raise RuntimeError(
            f"Checksum mismatch for {name}: expected {expected}, got {actual}"
        )
    return path
