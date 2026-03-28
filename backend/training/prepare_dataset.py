"""
Dataset preparation pipeline for weapon detection.

Downloads, converts, and merges multiple datasets into a unified
YOLO-format dataset with 4 classes: handgun, long_gun, knife, explosive.
"""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# ── Class taxonomy ──────────────────────────────────────────────────────────
CLASSES = ["handgun", "long_gun", "knife", "explosive"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Map source dataset labels → our 4-class taxonomy
LABEL_REMAP = {
    # handgun
    "pistol": "handgun",
    "handgun": "handgun",
    "revolver": "handgun",
    "derringer": "handgun",
    "hand_gun": "handgun",
    "Handgun": "handgun",
    "Pistol": "handgun",
    "gun": "handgun",  # generic "gun" → handgun (most common in datasets)
    "Gun": "handgun",
    # long_gun
    "rifle": "long_gun",
    "shotgun": "long_gun",
    "shot-gun": "long_gun",
    "assault_rifle": "long_gun",
    "smg": "long_gun",
    "submachine-gun": "long_gun",
    "submachine_gun": "long_gun",
    "carbine": "long_gun",
    "long_gun": "long_gun",
    "Rifle": "long_gun",
    "Shotgun": "long_gun",
    "machine_gun": "long_gun",
    "sniper": "long_gun",
    "heavyweapon": "long_gun",
    "RPG": "long_gun",
    # knife
    "knife": "knife",
    "Knife": "knife",
    "pocket_knife": "knife",
    "kitchen_knife": "knife",
    "machete": "knife",
    "katana": "knife",
    "dagger": "knife",
    "sword": "knife",
    "blade": "knife",
    # explosive
    "grenade": "explosive",
    "explosive": "explosive",
    "ied": "explosive",
    "bomb": "explosive",
    "Grenade": "explosive",
    "missile": "explosive",
    # Ignored classes (return None via remap_label):
    # person, gunmen, knife_attacker, blunt_object, tank, military-vehicle
}


def remap_label(label: str) -> str | None:
    """Map a source label to our taxonomy. Returns None if unmapped."""
    return LABEL_REMAP.get(label) or LABEL_REMAP.get(label.lower().replace(" ", "_"))


# ── Format converters ───────────────────────────────────────────────────────

def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path, output_dir: Path) -> dict:
    """Convert COCO JSON annotations to YOLO TXT format.

    Returns stats dict with counts per class.
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Build category id → name map
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    # Build image id → info map
    img_map = {img["id"]: img for img in coco["images"]}
    # Group annotations by image
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = Counter()

    for img_id, img_info in img_map.items():
        w, h = img_info["width"], img_info["height"]
        filename = img_info["file_name"]
        src_img = images_dir / filename
        if not src_img.exists():
            continue

        lines = []
        for ann in ann_by_img.get(img_id, []):
            cat_name = cat_map[ann["category_id"]]
            mapped = remap_label(cat_name)
            if mapped is None:
                continue

            # COCO bbox = [x, y, w, h] (top-left)
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clamp to [0, 1]
            cx, cy, nw, nh = (max(0, min(1, v)) for v in (cx, cy, nw, nh))
            if nw < 0.001 or nh < 0.001:
                continue

            class_id = CLASS_TO_ID[mapped]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[mapped] += 1

        if lines:
            stem = Path(filename).stem
            shutil.copy2(src_img, out_images / Path(filename).name)
            with open(out_labels / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

    return dict(stats)


def convert_voc_to_yolo(voc_dir: Path, output_dir: Path) -> dict:
    """Convert Pascal VOC XML annotations to YOLO TXT format."""
    annotations_dir = voc_dir / "Annotations"
    images_dir = voc_dir / "JPEGImages"
    if not annotations_dir.exists():
        annotations_dir = voc_dir  # flat structure

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = Counter()

    for xml_path in annotations_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w == 0 or h == 0:
            continue

        filename = root.find("filename").text
        src_img = images_dir / filename
        if not src_img.exists():
            # Try common image dirs
            for alt in [voc_dir / "images", voc_dir]:
                if (alt / filename).exists():
                    src_img = alt / filename
                    break

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            mapped = remap_label(name)
            if mapped is None:
                continue

            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h

            cx, cy, nw, nh = (max(0, min(1, v)) for v in (cx, cy, nw, nh))
            if nw < 0.001 or nh < 0.001:
                continue

            class_id = CLASS_TO_ID[mapped]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[mapped] += 1

        if lines and src_img.exists():
            stem = xml_path.stem
            shutil.copy2(src_img, out_images / Path(filename).name)
            with open(out_labels / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

    return dict(stats)


def convert_yolo_to_yolo(src_dir: Path, output_dir: Path, class_names: list[str]) -> dict:
    """Re-map existing YOLO-format labels to our class taxonomy.

    Args:
        src_dir: Directory with images/ and labels/ subdirs (or flat).
        output_dir: Where to write remapped files.
        class_names: Ordered list of class names from the source dataset.
    """
    # Find images and labels dirs
    src_images = src_dir / "images" if (src_dir / "images").exists() else src_dir
    src_labels = src_dir / "labels" if (src_dir / "labels").exists() else src_dir

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for label_path in src_labels.glob("*.txt"):
        if label_path.name == "classes.txt":
            continue

        stem = label_path.stem
        # Find matching image
        img_path = None
        for ext in img_exts:
            candidate = src_images / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        lines = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_class_id = int(parts[0])
                if src_class_id >= len(class_names):
                    continue
                src_name = class_names[src_class_id]
                mapped = remap_label(src_name)
                if mapped is None:
                    continue

                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if w < 0.001 or h < 0.001:
                    continue

                new_id = CLASS_TO_ID[mapped]
                lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[mapped] += 1

        if lines:
            shutil.copy2(img_path, out_images / img_path.name)
            with open(out_labels / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines) + "\n")

    return dict(stats)


# ── Validation ──────────────────────────────────────────────────────────────

def validate_annotations(dataset_dir: Path) -> dict:
    """Validate a YOLO-format dataset, removing corrupt entries.

    Returns stats about valid/invalid counts.
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    valid = 0
    removed_no_image = 0
    removed_bad_label = 0
    removed_empty = 0

    for label_path in list(labels_dir.glob("*.txt")):
        stem = label_path.stem

        # Check matching image exists
        has_image = any((images_dir / f"{stem}{ext}").exists() for ext in img_exts)
        if not has_image:
            label_path.unlink()
            removed_no_image += 1
            continue

        # Check label content
        with open(label_path, "r") as f:
            lines = f.readlines()

        clean_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:])
            except ValueError:
                continue

            if cls_id < 0 or cls_id >= len(CLASSES):
                continue
            if not all(0 <= v <= 1 for v in (cx, cy, w, h)):
                continue
            if w < 0.001 or h < 0.001:
                continue

            clean_lines.append(line.strip())

        if not clean_lines:
            label_path.unlink()
            # Also remove the orphan image
            for ext in img_exts:
                img = images_dir / f"{stem}{ext}"
                if img.exists():
                    img.unlink()
            removed_empty += 1
        elif len(clean_lines) < len(lines):
            with open(label_path, "w") as f:
                f.write("\n".join(clean_lines) + "\n")
            removed_bad_label += len(lines) - len(clean_lines)
            valid += 1
        else:
            valid += 1

    return {
        "valid_images": valid,
        "removed_no_image": removed_no_image,
        "removed_bad_labels": removed_bad_label,
        "removed_empty": removed_empty,
    }


# ── Train/Val/Test split ───────────────────────────────────────────────────

def stratified_split(
    merged_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split a merged YOLO dataset into train/val/test with stratified sampling.

    Stratifies by the *primary* class in each image (class with most annotations).
    """
    images_dir = merged_dir / "images"
    labels_dir = merged_dir / "labels"
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Build image → primary class mapping
    image_classes = {}
    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        # Find image
        img_path = None
        for ext in img_exts:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        with open(label_path, "r") as f:
            class_counts = Counter()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_counts[int(parts[0])] += 1

        if class_counts:
            primary_class = class_counts.most_common(1)[0][0]
            image_classes[stem] = (img_path, label_path, primary_class)

    # Group by primary class
    class_groups = defaultdict(list)
    for stem, (img_path, label_path, cls) in image_classes.items():
        class_groups[cls].append((stem, img_path, label_path))

    # Split each class group
    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}

    for cls, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    # Copy files to output
    for split_name, items in splits.items():
        split_images = output_dir / "images" / split_name
        split_labels = output_dir / "labels" / split_name
        split_images.mkdir(parents=True, exist_ok=True)
        split_labels.mkdir(parents=True, exist_ok=True)

        for stem, img_path, label_path in items:
            shutil.copy2(img_path, split_images / img_path.name)
            shutil.copy2(label_path, split_labels / label_path.name)

    print(f"Split complete: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

    return {k: len(v) for k, v in splits.items()}


# ── Roboflow download helper ───────────────────────────────────────────────

def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    output_dir: Path,
    format: str = "yolov8",
):
    """Download a dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip install roboflow")
        return None

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(format, location=str(output_dir))
    return dataset


# ── Main pipeline ───────────────────────────────────────────────────────────

def merge_datasets(source_dirs: list[dict], output_dir: Path) -> dict:
    """Merge multiple converted datasets into a single directory.

    Args:
        source_dirs: List of dicts with keys:
            - path: Path to dataset dir (with images/ and labels/ subdirs)
            - prefix: Unique prefix to avoid filename collisions
        output_dir: Where to write merged dataset.
    """
    merged_images = output_dir / "images"
    merged_labels = output_dir / "labels"
    merged_images.mkdir(parents=True, exist_ok=True)
    merged_labels.mkdir(parents=True, exist_ok=True)

    total = 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for source in source_dirs:
        src_path = Path(source["path"])
        prefix = source["prefix"]
        src_images = src_path / "images"
        src_labels = src_path / "labels"

        if not src_images.exists() or not src_labels.exists():
            print(f"Skipping {src_path}: missing images/ or labels/")
            continue

        count = 0
        for label_path in src_labels.glob("*.txt"):
            stem = label_path.stem
            # Find matching image
            img_path = None
            for ext in img_exts:
                candidate = src_images / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            new_stem = f"{prefix}_{stem}"
            shutil.copy2(img_path, merged_images / f"{new_stem}{img_path.suffix}")
            shutil.copy2(label_path, merged_labels / f"{new_stem}.txt")
            count += 1

        print(f"Merged {count} images from {prefix}")
        total += count

    print(f"Total merged: {total} images")
    return {"total": total}


def run_full_pipeline(
    source_configs: list[dict],
    final_output_dir: Path,
    seed: int = 42,
):
    """Run the complete dataset preparation pipeline.

    Args:
        source_configs: List of source dataset configs. Each dict has:
            - path: Path to raw dataset
            - format: "yolo", "coco", or "voc"
            - class_names: (for yolo format) list of source class names
            - coco_json: (for coco format) path to annotations JSON
            - prefix: unique prefix string
        final_output_dir: Path for the final split dataset.
    """
    base_dir = Path(__file__).parent.parent
    staging_dir = base_dir / "datasets" / "_staging"
    merged_dir = base_dir / "datasets" / "_merged"

    # Step 1: Convert each dataset
    print("=" * 60)
    print("STEP 1: Converting datasets")
    print("=" * 60)
    converted = []
    all_stats = Counter()

    for cfg in source_configs:
        src_path = Path(cfg["path"])
        prefix = cfg["prefix"]
        fmt = cfg["format"]
        out = staging_dir / prefix

        print(f"\nConverting {prefix} ({fmt})...")

        if fmt == "yolo":
            stats = convert_yolo_to_yolo(src_path, out, cfg["class_names"])
        elif fmt == "coco":
            stats = convert_coco_to_yolo(
                Path(cfg["coco_json"]), src_path / "images", out
            )
        elif fmt == "voc":
            stats = convert_voc_to_yolo(src_path, out)
        else:
            print(f"  Unknown format: {fmt}, skipping")
            continue

        print(f"  Stats: {stats}")
        all_stats.update(stats)
        converted.append({"path": str(out), "prefix": prefix})

    print(f"\nTotal annotations by class: {dict(all_stats)}")

    # Step 2: Merge
    print("\n" + "=" * 60)
    print("STEP 2: Merging datasets")
    print("=" * 60)
    merge_datasets(converted, merged_dir)

    # Step 3: Validate
    print("\n" + "=" * 60)
    print("STEP 3: Validating annotations")
    print("=" * 60)
    val_stats = validate_annotations(merged_dir)
    print(f"Validation: {val_stats}")

    # Step 4: Split
    print("\n" + "=" * 60)
    print("STEP 4: Stratified train/val/test split")
    print("=" * 60)
    split_stats = stratified_split(merged_dir, final_output_dir, seed=seed)

    # Step 5: Cleanup staging
    print("\nCleaning up staging directories...")
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(merged_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Output: {final_output_dir}")
    print(f"Splits: {split_stats}")
    print(f"Class stats: {dict(all_stats)}")
    print("=" * 60)

    return {"splits": split_stats, "class_stats": dict(all_stats)}


# ── Example usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: configure your downloaded datasets here.
    # After downloading datasets manually or via Roboflow, set up configs:
    #
    # source_configs = [
    #     {
    #         "path": "path/to/roboflow-cctv-v3",
    #         "format": "yolo",
    #         "class_names": ["pistol", "rifle", "shotgun", "knife", "grenade",
    #                         "handgun", "revolver", "smg", "machete", "sword", "bomb"],
    #         "prefix": "cctv",
    #     },
    #     {
    #         "path": "path/to/youtube-gdd",
    #         "format": "voc",
    #         "prefix": "ytgdd",
    #     },
    #     {
    #         "path": "path/to/dasci-knife",
    #         "format": "coco",
    #         "coco_json": "path/to/dasci-knife/annotations.json",
    #         "prefix": "dasci",
    #     },
    # ]
    #
    # output_dir = Path(__file__).parent.parent / "datasets" / "weapon_detection_v2"
    # run_full_pipeline(source_configs, output_dir)

    print("Dataset preparation pipeline ready.")
    print("Configure source_configs in this file or import and call run_full_pipeline().")
    print(f"Classes: {CLASSES}")
    print(f"Label remap has {len(LABEL_REMAP)} entries")
