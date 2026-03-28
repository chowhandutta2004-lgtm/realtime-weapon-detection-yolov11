"""
Merge newly downloaded knife datasets into the existing weapon_detection_v2 dataset.

This script:
1. Scans all knife datasets in _raw_knives/
2. Remaps their class labels to our 4-class taxonomy (knife = class 2)
3. Validates annotations
4. Merges into the existing weapon_detection_v2 train/val/test splits (80/10/10)
5. Generates a report of the final dataset composition

Usage:
    python merge_knife_data.py
"""

import os
import random
import shutil
import yaml
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_KNIVES_DIR = BASE_DIR / "datasets" / "_raw_knives"
DATASET_DIR = BASE_DIR / "datasets" / "weapon_detection_v2"

# Our class taxonomy
CLASSES = ["handgun", "long_gun", "knife", "explosive"]
KNIFE_CLASS_ID = 2  # knife is index 2 in our taxonomy


def parse_roboflow_classes(dataset_dir: Path) -> dict:
    """Parse class names from a Roboflow dataset's data.yaml."""
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        # Try looking one level deeper
        for f in dataset_dir.rglob("data.yaml"):
            yaml_path = f
            break

    if not yaml_path.exists():
        return {}

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    except Exception as e:
        print(f"    Warning: Could not parse {yaml_path}: {e}")
    return {}


def find_image_label_pairs(dataset_dir: Path) -> list:
    """Find all image-label pairs in a dataset directory."""
    pairs = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Find all images
    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue
        # Skip if in a non-image directory
        if any(skip in str(img_path) for skip in ["__pycache__", ".git"]):
            continue

        # Find corresponding label
        # Try same directory with .txt extension
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            # Try labels/ directory parallel to images/
            parts = list(img_path.parts)
            for i, part in enumerate(parts):
                if part.lower() == "images":
                    parts[i] = "labels"
                    label_path = Path(*parts).with_suffix(".txt")
                    break

        if label_path.exists():
            pairs.append((img_path, label_path))

    return pairs


def remap_knife_labels(label_path: Path, source_classes: dict) -> list:
    """Read a YOLO label file and remap all knife-related classes to our knife class ID.

    Returns list of remapped annotation lines, or empty list if no knife annotations.
    """
    remapped = []
    knife_keywords = [
        "knife", "blade", "dagger", "machete", "sword", "katana",
        "pocket_knife", "kitchen_knife", "cutter", "cleaver",
    ]

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        src_class_id = int(parts[0])
        src_class_name = source_classes.get(src_class_id, "").lower()

        # Check if this is a knife-related class
        is_knife = False
        if any(kw in src_class_name for kw in knife_keywords):
            is_knife = True
        elif not source_classes:
            # If we can't parse classes, assume single-class dataset = knife
            is_knife = True
        elif len(source_classes) == 1:
            # Single class dataset — it's knife
            is_knife = True

        if is_knife:
            # Validate bbox values
            try:
                coords = [float(x) for x in parts[1:5]]
                if all(0 <= c <= 1 for c in coords) and coords[2] > 0 and coords[3] > 0:
                    remapped.append(f"{KNIFE_CLASS_ID} {' '.join(parts[1:])}")
            except ValueError:
                continue

    return remapped


def merge_into_dataset(pairs_with_labels: list, prefix: str):
    """Merge image-label pairs into the existing dataset with 80/10/10 split."""
    random.seed(42)
    random.shuffle(pairs_with_labels)

    n = len(pairs_with_labels)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": pairs_with_labels[:n_train],
        "val": pairs_with_labels[n_train:n_train + n_val],
        "test": pairs_with_labels[n_train + n_val:],
    }

    total_added = 0
    for split_name, items in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i, (img_path, label_lines) in enumerate(items):
            # Generate unique filename
            new_name = f"{prefix}_{i:05d}{img_path.suffix.lower()}"

            # Copy image
            dst_img = img_dir / new_name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Write remapped label
            dst_lbl = lbl_dir / f"{prefix}_{i:05d}.txt"
            with open(dst_lbl, "w") as f:
                f.write("\n".join(label_lines) + "\n")

            total_added += 1

    return total_added, len(splits["train"]), len(splits["val"]), len(splits["test"])


def count_existing_dataset():
    """Count current annotations per class in the existing dataset."""
    counts = Counter()
    for split in ["train", "val", "test"]:
        lbl_dir = DATASET_DIR / "labels" / split
        if not lbl_dir.exists():
            continue
        for txt_file in lbl_dir.glob("*.txt"):
            try:
                with open(txt_file) as f:
                    for line in f:
                        cls_id = int(line.strip().split()[0])
                        if 0 <= cls_id < len(CLASSES):
                            counts[CLASSES[cls_id]] += 1
            except (ValueError, IndexError):
                continue
    return counts


def main():
    print("=" * 60)
    print("MERGING KNIFE DATASETS INTO TRAINING DATA")
    print("=" * 60)

    if not RAW_KNIVES_DIR.exists():
        print(f"\n[ERROR] No knife datasets found at {RAW_KNIVES_DIR}")
        print("Run download_knife_datasets.py first!")
        return

    # Show current dataset composition
    print("\n--- Current Dataset ---")
    before_counts = count_existing_dataset()
    for cls, count in sorted(before_counts.items()):
        print(f"  {cls:15s}: {count:6d} annotations")
    print(f"  {'TOTAL':15s}: {sum(before_counts.values()):6d}")

    # Process each knife dataset
    total_new_images = 0
    dataset_dirs = sorted(d for d in RAW_KNIVES_DIR.iterdir() if d.is_dir())

    if not dataset_dirs:
        print(f"\n[ERROR] No dataset directories found in {RAW_KNIVES_DIR}")
        return

    print(f"\nFound {len(dataset_dirs)} knife datasets to merge\n")

    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        print(f"Processing: {name}")

        # Parse source classes
        source_classes = parse_roboflow_classes(dataset_dir)
        if source_classes:
            print(f"  Source classes: {source_classes}")
        else:
            print("  Source classes: unknown (assuming single-class knife)")

        # Find all image-label pairs
        pairs = find_image_label_pairs(dataset_dir)
        print(f"  Found {len(pairs)} image-label pairs")

        if not pairs:
            print("  [SKIP] No valid pairs found")
            continue

        # Remap labels to knife class
        valid_pairs = []
        for img_path, lbl_path in pairs:
            remapped = remap_knife_labels(lbl_path, source_classes)
            if remapped:
                valid_pairs.append((img_path, remapped))

        print(f"  {len(valid_pairs)} images with knife annotations")

        if not valid_pairs:
            print("  [SKIP] No knife annotations found after remapping")
            continue

        # Merge into existing dataset
        prefix = f"knife_{name.replace('roboflow_', '')}"
        added, train, val, test = merge_into_dataset(valid_pairs, prefix)
        print(f"  [OK] Added {added} images (train={train}, val={val}, test={test})")
        total_new_images += added

    # Final report
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nAdded {total_new_images} new knife images")

    print("\n--- Updated Dataset ---")
    after_counts = count_existing_dataset()
    for cls in CLASSES:
        before = before_counts.get(cls, 0)
        after = after_counts.get(cls, 0)
        diff = after - before
        marker = f" (+{diff})" if diff > 0 else ""
        print(f"  {cls:15s}: {after:6d} annotations{marker}")
    print(f"  {'TOTAL':15s}: {sum(after_counts.values()):6d}")

    # Count total images
    total_images = 0
    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        if img_dir.exists():
            count = sum(1 for _ in img_dir.iterdir() if _.is_file())
            total_images += count
            print(f"  {split:15s}: {count:6d} images")
    print(f"  {'TOTAL IMAGES':15s}: {total_images:6d}")

    print(f"\nDataset ready at: {DATASET_DIR}")
    print("Next: Upload to Kaggle and retrain with train_kaggle.ipynb")


if __name__ == "__main__":
    main()
