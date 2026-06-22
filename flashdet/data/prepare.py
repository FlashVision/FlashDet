"""
Dataset preparation utilities.

Supports conversion from:
  1. YOLO format   → COCO JSON  (convert_yolo_to_coco)
  2. Pascal VOC    → COCO JSON  (convert_voc_to_coco)
  3. Supervisely format → COCO JSON  (convert_supervisely_to_coco)

Class names are always read from the source dataset (YOLO data.yaml /
Supervisely meta.json) so no hardcoded PPE names pollute other datasets.
"""

import os
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

_FALLBACK_CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3", "class_4",
    "class_5", "class_6", "class_7", "class_8", "class_9",
]


def _read_yolo_class_names(yolo_dir: str) -> Optional[List[str]]:
    """
    Try to read class names from a YOLO dataset directory.

    Looks for (in order):
      1. data.yaml   — standard Roboflow / YOLOv5 / YOLOv8 export
      2. classes.txt — older YOLOv4 / darknet convention
    """
    # 1. data.yaml
    for yaml_name in ["data.yaml", "dataset.yaml", "ppe_data.yaml"]:
        yaml_path = os.path.join(yolo_dir, yaml_name)
        if os.path.isfile(yaml_path):
            try:
                import yaml
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                names = data.get("names")
                if isinstance(names, list) and names:
                    return names
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names.keys())]
            except Exception:
                pass

    # 2. classes.txt
    txt_path = os.path.join(yolo_dir, "classes.txt")
    if not os.path.isfile(txt_path):
        txt_path = os.path.join(yolo_dir, "train", "labels", "classes.txt")
    if os.path.isfile(txt_path):
        with open(txt_path) as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names

    return None


def _copy_or_link(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _flat_yolo_output_filename(img_path: str, yolo_root: str) -> str:
    rel = os.path.relpath(img_path, yolo_root)
    if rel.startswith(".."):
        return os.path.basename(img_path)
    safe = rel.replace(os.sep, "__")
    return safe if safe.lower().endswith(tuple(_IMG_EXT)) else os.path.basename(img_path)


def _find_any_coco_json(folder: str, max_depth: int = 6) -> Optional[str]:
    if not os.path.isdir(folder):
        return None
    base = os.path.abspath(folder)
    for root, dirs, files in os.walk(folder):
        depth = os.path.abspath(root)[len(base) :].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue
        if "_annotations.coco.json" in files:
            return os.path.join(root, "_annotations.coco.json")
    return None


def _has_yolo_split_layout(folder: str) -> bool:
    for split in ("train", "valid", "val", "test"):
        img_d = os.path.join(folder, split, "images")
        lbl_d = os.path.join(folder, split, "labels")
        if os.path.isdir(img_d) and os.path.isdir(lbl_d):
            return True
    return False


def _yolo_txt_beside_image(img_path: str, yolo_root: str) -> Optional[str]:
    p = Path(img_path)
    stem = p.stem
    same = p.with_suffix(".txt")
    if same.is_file():
        return str(same)
    labels_peer = p.parent / "labels" / f"{stem}.txt"
    if labels_peer.is_file():
        return str(labels_peer)
    root_lbl = Path(yolo_root) / "labels" / f"{stem}.txt"
    if root_lbl.is_file():
        return str(root_lbl)
    return None


def _line_looks_yolo(parts: List[str]) -> bool:
    if len(parts) < 5:
        return False
    try:
        int(parts[0])
        for x in parts[1:5]:
            float(x)
    except (ValueError, TypeError):
        return False
    return True


def _has_yolo_flat_txt(folder: str) -> bool:
    """True when label .txt files sit next to images (or images/ + labels/)."""
    if os.path.isdir(os.path.join(folder, "images")) and os.path.isdir(
        os.path.join(folder, "labels")
    ):
        lbl_dir = Path(folder) / "labels"
        for lf in list(lbl_dir.glob("*.txt"))[:5]:
            try:
                with open(lf) as f:
                    line = f.readline().strip()
                if line and _line_looks_yolo(line.split()):
                    return True
            except OSError:
                continue

    n_img = 0
    n_txt_side = 0
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in _IMG_EXT:
                continue
            n_img += 1
            ip = os.path.join(root, f)
            if _yolo_txt_beside_image(ip, folder):
                n_txt_side += 1
            if n_img >= 300:
                break
        if n_img >= 300:
            break

    if n_txt_side == 0:
        return False
    if n_txt_side >= 3:
        return True
    return n_img > 0 and (n_txt_side / max(n_img, 1)) >= 0.15


def _voc_ann_dir(folder: str) -> Optional[str]:
    for name in ("Annotations", "annotations"):
        d = os.path.join(folder, name)
        if os.path.isdir(d):
            try:
                if any(x.endswith(".xml") for x in os.listdir(d)):
                    return d
            except OSError:
                continue
    return None


def detect_dataset_format(folder: str) -> str:
    """
    Heuristic format label for a dataset root.

    Returns one of: ``coco``, ``yolo``, ``voc``, ``unknown``.
    Priority: COCO JSON → YOLO → Pascal VOC.
    """
    if not os.path.isdir(folder):
        return "unknown"
    if _find_any_coco_json(folder) is not None:
        return "coco"
    if _has_yolo_split_layout(folder) or _has_yolo_flat_txt(folder):
        return "yolo"
    if _voc_ann_dir(folder) is not None:
        return "voc"
    return "unknown"


def summarize_coco_root(coco_root: str) -> Dict:
    """
    Aggregate image / annotation counts and per-class distribution from COCO JSON files.

    Looks under ``train/``, ``valid/``, ``val/``, ``test/``, or a single JSON at ``coco_root``.
    """
    out: Dict = {
        "n_images": 0,
        "n_annotations": 0,
        "class_names": [],
        "distribution": {},
        "splits": {},
    }
    if not os.path.isdir(coco_root):
        return out

    paths = []
    root_json = os.path.join(coco_root, "_annotations.coco.json")
    if os.path.isfile(root_json):
        paths.append((".", root_json))
    for split in ("train", "valid", "val", "test"):
        jp = os.path.join(coco_root, split, "_annotations.coco.json")
        if os.path.isfile(jp):
            paths.append((split, jp))

    id_to_name: Dict[int, str] = {}
    dist: Counter = Counter()

    for split, jp in paths:
        try:
            with open(jp) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        cats = data.get("categories") or []
        for c in cats:
            try:
                cid = int(c["id"])
                id_to_name[cid] = str(c.get("name", f"class_{cid}"))
            except (KeyError, ValueError, TypeError):
                continue
        ni = len(data.get("images") or [])
        na = len(data.get("annotations") or [])
        out["splits"][split] = {"images": ni, "annotations": na}
        out["n_images"] += ni
        out["n_annotations"] += na
        for ann in data.get("annotations") or []:
            try:
                dist[int(ann["category_id"])] += 1
            except (KeyError, ValueError, TypeError):
                continue

    if id_to_name:
        max_id = max(id_to_name.keys())
        ordered = [
            id_to_name[i]
            for i in range(max_id + 1)
            if i in id_to_name
        ]
        if not ordered:
            ordered = [id_to_name[k] for k in sorted(id_to_name.keys())]
        out["class_names"] = ordered
    out["distribution"] = {
        id_to_name.get(k, str(k)): v for k, v in sorted(dist.items())
    }
    return out


def convert_voc_to_coco(
    voc_dir: str,
    coco_dir: str,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Convert Pascal VOC XML annotations to COCO JSON under ``train/`` and ``valid/``.

    Expects ``JPEGImages`` / ``images`` / ``img`` and ``Annotations`` / ``annotations``.
    """
    random.seed(seed)

    img_dir = None
    for name in ("JPEGImages", "images", "img"):
        p = os.path.join(voc_dir, name)
        if os.path.isdir(p):
            img_dir = p
            break

    ann_dir = _voc_ann_dir(voc_dir)
    if not img_dir:
        raise FileNotFoundError(
            f"No JPEGImages/, images/, or img/ folder in:\n{voc_dir}"
        )
    if not ann_dir:
        raise FileNotFoundError(
            f"No Annotations/ folder with .xml files in:\n{voc_dir}"
        )

    xml_files = sorted(Path(ann_dir).glob("*.xml"))
    pairs: List[Tuple[str, str, List[dict]]] = []
    all_cls = set()

    for xf in xml_files:
        tree = ET.parse(xf)
        root_el = tree.getroot()
        fn_el = root_el.find("filename")
        fname = fn_el.text.strip() if fn_el is not None and fn_el.text else xf.stem + ".jpg"
        img_path = os.path.join(img_dir, fname)
        if not os.path.isfile(img_path):
            found = False
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                cand = os.path.join(img_dir, xf.stem + ext)
                if os.path.isfile(cand):
                    fname = xf.stem + ext
                    img_path = cand
                    found = True
                    break
            if not found:
                continue

        anns = []
        for obj in root_el.findall("object"):
            name_el = obj.find("name")
            bb = obj.find("bndbox")
            if name_el is None or bb is None or not name_el.text:
                continue
            cls = name_el.text.strip()
            try:
                x1 = int(float(bb.find("xmin").text))
                y1 = int(float(bb.find("ymin").text))
                x2 = int(float(bb.find("xmax").text))
                y2 = int(float(bb.find("ymax").text))
            except (AttributeError, TypeError, ValueError):
                continue
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            anns.append({"bbox": [x1, y1, w, h], "area": w * h, "class": cls})
            all_cls.add(cls)

        if anns:
            pairs.append((fname, img_path, anns))

    if not pairs:
        raise ValueError("No valid VOC XML + image pairs found.")

    cat_names = sorted(all_cls)
    c2id = {n: i for i, n in enumerate(cat_names)}
    categories = [{"id": i, "name": n, "supercategory": "object"} for i, n in enumerate(cat_names)]

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    if len(pairs) >= 2 and n_val >= len(pairs):
        n_val = len(pairs) - 1
    if len(pairs) == 1:
        split_map = {"train": pairs, "valid": pairs}
    else:
        split_map = {"train": pairs[:-n_val], "valid": pairs[-n_val:]}

    os.makedirs(coco_dir, exist_ok=True)
    stats: Dict = {}

    for split_name, sp in split_map.items():
        if not sp:
            continue
        out_dir = os.path.join(coco_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1
        for idx, (fname, src_img, anns) in enumerate(sp, start=1):
            dst_img = os.path.join(out_dir, fname)
            _copy_or_link(src_img, dst_img)
            with Image.open(dst_img) as im:
                iw, ih = im.size
            coco["images"].append(
                {"id": idx, "file_name": fname, "width": iw, "height": ih}
            )
            for a in anns:
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": c2id.get(a["class"], 0),
                        "bbox": a["bbox"],
                        "area": a["area"],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        ann_path = os.path.join(out_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        stats[split_name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }
        print(f"[VOC→COCO] Saved {ann_path}")

    return stats


def _convert_yolo_flat_to_coco(
    yolo_dir: str,
    coco_dir: str,
    class_names: List[str],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    """YOLO without train/*/images layout: images anywhere + sibling ``labels/`` or co-located .txt."""
    random.seed(seed)
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    seen_rel = set()
    pairs: List[Tuple[str, Optional[str]]] = []
    abs_coco = os.path.abspath(coco_dir)

    for root, dirs, files in os.walk(yolo_dir):
        dirs[:] = [
            d
            for d in dirs
            if os.path.abspath(os.path.join(root, d)) != abs_coco
            and not os.path.abspath(os.path.join(root, d)).startswith(abs_coco + os.sep)
        ]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in _IMG_EXT:
                continue
            ip = os.path.normpath(os.path.join(root, f))
            rel = os.path.relpath(ip, yolo_dir)
            if rel in seen_rel:
                continue
            seen_rel.add(rel)
            lbl = _yolo_txt_beside_image(ip, yolo_dir)
            pairs.append((ip, lbl))

    pairs.sort(key=lambda x: x[0])
    if not pairs:
        raise ValueError(
            "No images found for flat YOLO conversion. "
            "Expected images with .txt labels beside them or under labels/."
        )

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    if len(pairs) >= 2 and n_val >= len(pairs):
        n_val = len(pairs) - 1
    if len(pairs) == 1:
        split_map = {"train": pairs, "valid": pairs}
    else:
        split_map = {"train": pairs[:-n_val], "valid": pairs[-n_val:]}

    os.makedirs(coco_dir, exist_ok=True)
    stats: Dict = {}

    for split_name, sp in split_map.items():
        out_dir = os.path.join(coco_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1
        img_id = 0

        for img_path, lbl_path in tqdm(sp, desc=f"yolo-flat-{split_name}"):
            fname = _flat_yolo_output_filename(img_path, yolo_dir)
            dst = os.path.join(out_dir, fname)
            _copy_or_link(img_path, dst)

            try:
                with Image.open(dst) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Skip unreadable image {dst}: {e}")
                continue

            img_id += 1
            coco["images"].append(
                {"id": img_id, "file_name": fname, "width": width, "height": height}
            )

            if lbl_path and os.path.isfile(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if not _line_looks_yolo(parts):
                            continue
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x = (cx - w / 2) * width
                        y = (cy - h / 2) * height
                        box_w = w * width
                        box_h = h * height
                        x = max(0, x)
                        y = max(0, y)
                        box_w = min(box_w, width - x)
                        box_h = min(box_h, height - y)
                        coco["annotations"].append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": class_id,
                                "bbox": [
                                    round(x, 2),
                                    round(y, 2),
                                    round(box_w, 2),
                                    round(box_h, 2),
                                ],
                                "area": round(box_w * box_h, 2),
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

        ann_path = os.path.join(out_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        stats[split_name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }
        print(f"[YOLO flat→COCO] Saved {ann_path}")

    return stats


def convert_yolo_to_coco(
    yolo_dir: str,
    coco_dir: str,
    class_names: List[str] = None,
) -> Dict:
    """
    Convert a YOLO-format dataset to COCO JSON format.

    Class names are resolved in this priority order:
      1. Explicit ``class_names`` argument (caller-supplied)
      2. ``data.yaml`` / ``classes.txt`` found inside ``yolo_dir``
      3. Hard-coded fallback (PPE names — legacy behaviour)

    Args:
        yolo_dir: Root directory of the YOLO dataset.
        coco_dir: Output directory for the COCO-format dataset.
        class_names: Optional explicit list of class names.

    Returns:
        Statistics dictionary {split: {"images": N, "annotations": M}}.
    """
    if class_names is None:
        class_names = _read_yolo_class_names(yolo_dir)
        if class_names is None:
            print(
                "[WARN] Could not read class names from data.yaml or classes.txt. "
                "Falling back to hardcoded PPE names — VERIFY this is correct!"
            )
            class_names = _FALLBACK_CLASS_NAMES

    print(f"Class names ({len(class_names)}): {class_names}")
    os.makedirs(coco_dir, exist_ok=True)
    
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    stats = {}

    for split in ["train", "valid", "test"]:
        images_dir = os.path.join(yolo_dir, split, "images")
        labels_dir = os.path.join(yolo_dir, split, "labels")
        
        if not os.path.exists(images_dir):
            print(f"Skipping {split}: {images_dir} not found")
            continue
        
        output_dir = os.path.join(coco_dir, split)
        os.makedirs(output_dir, exist_ok=True)
        
        coco = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        image_files = sorted(
            list(Path(images_dir).glob("*.jpg"))
            + list(Path(images_dir).glob("*.jpeg"))
            + list(Path(images_dir).glob("*.png"))
        )
        
        print(f"Converting {split}: {len(image_files)} images")
        
        ann_id = 1
        for img_id, img_path in enumerate(tqdm(image_files, desc=split), 1):
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue
            
            # Add image entry
            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height
            })
            
            # Create symlink
            link_path = os.path.join(output_dir, img_path.name)
            if not os.path.exists(link_path):
                try:
                    os.symlink(img_path.resolve(), link_path)
                except OSError:
                    import shutil
                    shutil.copy2(img_path, link_path)
            
            # Parse YOLO labels
            label_path = os.path.join(labels_dir, img_path.stem + ".txt")
            if os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to COCO format
                        x = (cx - w / 2) * width
                        y = (cy - h / 2) * height
                        box_w = w * width
                        box_h = h * height
                        
                        # Clamp
                        x = max(0, x)
                        y = max(0, y)
                        box_w = min(box_w, width - x)
                        box_h = min(box_h, height - y)
                        
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                            "area": round(box_w * box_h, 2),
                            "iscrowd": 0
                        })
                        ann_id += 1
        
        # Save annotations
        ann_path = os.path.join(output_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)
        
        stats[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"])
        }
        print(f"  Saved: {ann_path}")

    if stats:
        return stats

    print("No train/*/images layout — attempting flat YOLO conversion.")
    return _convert_yolo_flat_to_coco(yolo_dir, coco_dir, class_names)


def convert_supervisely_to_coco(
    supervisely_dir: str,
    coco_dir: str,
) -> Dict:
    """
    Convert a Supervisely-format project to COCO JSON format.

    Expects the standard Supervisely layout::

        supervisely_dir/
            meta.json          ← class definitions
            train/             ← OR ds0, ds1, ds2 …
                img/
                ann/
            valid/
                img/
                ann/
            test/
                img/
                ann/

    Class names are read from ``meta.json`` and sorted alphabetically so the
    0-indexed COCO ``category_id`` is always deterministic.

    Returns:
        Statistics dictionary {split: {"images": N, "annotations": M}}.
    """
    import shutil

    meta_path = os.path.join(supervisely_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"meta.json not found in {supervisely_dir}. "
            "Make sure this is a Supervisely project root."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    class_names = sorted([c["title"] for c in meta.get("classes", [])])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Class names ({len(class_names)}): {class_names}")
    os.makedirs(coco_dir, exist_ok=True)

    # Detect splits (named dirs or ds*)
    splits = {}
    for name in ["train", "valid", "val", "test"]:
        d = os.path.join(supervisely_dir, name)
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "ann")):
            canonical = "valid" if name == "val" else name
            splits[canonical] = d
    if not splits:
        ds_dirs = sorted(
            d for d in os.listdir(supervisely_dir)
            if d.startswith("ds") and os.path.isdir(os.path.join(supervisely_dir, d))
        )
        for i, ds in enumerate(ds_dirs):
            full = os.path.join(supervisely_dir, ds)
            if os.path.isdir(os.path.join(full, "ann")):
                name = ["train", "valid", "test"][i] if i < 3 else f"split_{i}"
                splits[name] = full

    stats = {}
    for split_name, split_dir in splits.items():
        print(f"\nConverting {split_name} …")
        img_dir = os.path.join(split_dir, "img")
        ann_dir = os.path.join(split_dir, "ann")
        output_dir = os.path.join(coco_dir, split_name)
        os.makedirs(output_dir, exist_ok=True)

        categories = [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(class_names)
        ]
        coco = {"images": [], "annotations": [], "categories": categories}

        img_paths = sorted(
            list(Path(img_dir).glob("*.jpg"))
            + list(Path(img_dir).glob("*.jpeg"))
            + list(Path(img_dir).glob("*.png"))
        )

        ann_id = 1
        for img_id, img_path in enumerate(tqdm(img_paths, desc=split_name), 1):
            ann_file = os.path.join(ann_dir, img_path.name + ".json")
            if not os.path.isfile(ann_file):
                ann_file = os.path.join(ann_dir, img_path.stem + ".json")

            img_w, img_h, objects = 0, 0, []
            if os.path.isfile(ann_file):
                with open(ann_file) as f:
                    ann_data = json.load(f)
                size = ann_data.get("size", {})
                img_h = size.get("height", 0)
                img_w = size.get("width", 0)
                objects = ann_data.get("objects", [])

            if img_w == 0 or img_h == 0:
                try:
                    with Image.open(img_path) as pil:
                        img_w, img_h = pil.size
                except Exception:
                    continue

            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            })

            link = os.path.join(output_dir, img_path.name)
            if not os.path.exists(link):
                try:
                    os.symlink(img_path.resolve(), link)
                except OSError:
                    shutil.copy2(str(img_path), link)

            for obj in objects:
                title = obj.get("classTitle", "")
                if title not in class_to_idx:
                    continue
                cat_idx = class_to_idx[title]
                exterior = obj.get("points", {}).get("exterior", [])
                if len(exterior) < 2:
                    continue
                xs = [p[0] for p in exterior]
                ys = [p[1] for p in exterior]
                x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
                x2, y2 = min(float(img_w), max(xs)), min(float(img_h), max(ys))
                w, h = x2 - x1, y2 - y1
                if w < 1 or h < 1:
                    continue
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_idx,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        ann_path = os.path.join(output_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            json.dump(coco, f, indent=2)

        stats[split_name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }
        print(f"  Saved: {ann_path}")

    return stats


def verify_dataset(coco_dir: str) -> bool:
    """
    Verify dataset structure.
    
    Args:
        coco_dir: COCO dataset directory
        
    Returns:
        True if dataset is valid
    """
    print("\n" + "=" * 50)
    print("Dataset Verification")
    print("=" * 50)
    
    required = ["train/_annotations.coco.json", "valid/_annotations.coco.json"]
    all_ok = True
    
    for path in required:
        full_path = os.path.join(coco_dir, path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                data = json.load(f)
            print(f"✓ {path}")
            print(f"  Images: {len(data['images'])}")
            print(f"  Annotations: {len(data['annotations'])}")
            
            # Count images with actual files
            split_dir = os.path.dirname(full_path)
            existing = sum(1 for img in data["images"] 
                         if os.path.exists(os.path.join(split_dir, img["file_name"])))
            print(f"  Files found: {existing}/{len(data['images'])}")
        else:
            print(f"✗ {path} - NOT FOUND")
            all_ok = False
    
    return all_ok
