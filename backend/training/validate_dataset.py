"""
Dataset validation and statistics for weapon detection.

Checks class balance, annotation quality, and generates summary stats.
Run after prepare_dataset.py to verify the dataset is ready for training.
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["handgun", "long_gun", "knife", "explosive"]
MIN_SAMPLES_PER_CLASS = 500


def count_annotations(dataset_dir: Path) -> dict:
    """Count images and annotations per class across all splits."""
    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            continue

        class_counts = Counter()
        image_count = 0
        images_per_class = Counter()

        for label_path in labels_dir.glob("*.txt"):
            image_count += 1
            seen_classes = set()

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < len(CLASSES):
                            class_counts[CLASSES[cls_id]] += 1
                            seen_classes.add(cls_id)

            for cls_id in seen_classes:
                images_per_class[CLASSES[cls_id]] += 1

        results[split] = {
            "images": image_count,
            "annotations": dict(class_counts),
            "images_per_class": dict(images_per_class),
        }

    return results


def check_class_balance(stats: dict) -> list[str]:
    """Flag classes with fewer than MIN_SAMPLES_PER_CLASS annotations."""
    warnings = []
    # Aggregate across splits
    total = Counter()
    for split_stats in stats.values():
        total.update(split_stats["annotations"])

    for cls in CLASSES:
        count = total.get(cls, 0)
        if count < MIN_SAMPLES_PER_CLASS:
            warnings.append(
                f"WARNING: '{cls}' has only {count} annotations "
                f"(minimum recommended: {MIN_SAMPLES_PER_CLASS})"
            )
        if count == 0:
            warnings.append(f"CRITICAL: '{cls}' has ZERO annotations!")

    # Check for severe imbalance
    counts = [total.get(c, 0) for c in CLASSES]
    if counts and max(counts) > 0:
        ratio = min(c for c in counts if c > 0) / max(counts)
        if ratio < 0.1:
            warnings.append(
                f"WARNING: Severe class imbalance detected (min/max ratio: {ratio:.2f}). "
                "Consider oversampling minority classes or adjusting class weights."
            )

    return warnings


def check_annotation_quality(dataset_dir: Path) -> dict:
    """Check for common annotation issues."""
    issues = {
        "tiny_boxes": 0,  # < 0.5% of image
        "huge_boxes": 0,  # > 90% of image
        "edge_boxes": 0,  # center within 1% of edge
        "duplicate_labels": 0,
        "total_checked": 0,
    }

    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            continue

        for label_path in labels_dir.glob("*.txt"):
            seen = set()
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    issues["total_checked"] += 1

                    if line in seen:
                        issues["duplicate_labels"] += 1
                    seen.add(line)

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    cx, cy, w, h = (float(p) for p in parts[1:5])
                    area = w * h

                    if area < 0.005:
                        issues["tiny_boxes"] += 1
                    if area > 0.9:
                        issues["huge_boxes"] += 1
                    if cx < 0.01 or cx > 0.99 or cy < 0.01 or cy > 0.99:
                        issues["edge_boxes"] += 1

    return issues


def visualize_samples(dataset_dir: Path, output_dir: Path, n_samples: int = 16, seed: int = 42):
    """Draw annotations on random sample images for visual spot-checking.

    Saves a grid image to output_dir.
    """
    colors = [
        (0, 0, 255),    # handgun: red
        (0, 0, 180),    # long_gun: dark red
        (0, 140, 255),  # knife: orange
        (200, 0, 200),  # explosive: magenta
    ]

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    if not images_dir.exists():
        print("No train split found for visualization")
        return

    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print("No label files found")
        return

    random.seed(seed)
    samples = random.sample(label_files, min(n_samples, len(label_files)))

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    annotated_images = []

    for label_path in samples:
        stem = label_path.stem
        img_path = None
        for ext in img_exts:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:5])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                color = colors[cls_id] if cls_id < len(colors) else (0, 255, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls_{cls_id}"
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Resize for grid
        img = cv2.resize(img, (416, 416))
        annotated_images.append(img)

    if not annotated_images:
        print("No images to visualize")
        return

    # Create grid
    cols = 4
    rows = (len(annotated_images) + cols - 1) // cols
    # Pad with black images
    while len(annotated_images) < rows * cols:
        annotated_images.append(np.zeros((416, 416, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        row_imgs = annotated_images[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_annotations.png"
    cv2.imwrite(str(output_path), grid)
    print(f"Sample annotations saved to {output_path}")


def generate_report(dataset_dir: Path) -> str:
    """Generate a full validation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET VALIDATION REPORT")
    lines.append(f"Dataset: {dataset_dir}")
    lines.append("=" * 60)

    # Count annotations
    stats = count_annotations(dataset_dir)

    total_images = 0
    total_annotations = Counter()

    for split, split_stats in stats.items():
        lines.append(f"\n--- {split.upper()} ---")
        lines.append(f"  Images: {split_stats['images']}")
        total_images += split_stats["images"]
        for cls in CLASSES:
            ann_count = split_stats["annotations"].get(cls, 0)
            img_count = split_stats["images_per_class"].get(cls, 0)
            lines.append(f"  {cls:12s}: {ann_count:6d} annotations in {img_count:5d} images")
            total_annotations[cls] += ann_count

    lines.append(f"\n--- TOTALS ---")
    lines.append(f"  Total images: {total_images}")
    for cls in CLASSES:
        lines.append(f"  {cls:12s}: {total_annotations[cls]:6d} total annotations")

    # Class balance warnings
    lines.append(f"\n--- CLASS BALANCE ---")
    warnings = check_class_balance(stats)
    if warnings:
        for w in warnings:
            lines.append(f"  {w}")
    else:
        lines.append("  All classes have sufficient samples.")

    # Annotation quality
    lines.append(f"\n--- ANNOTATION QUALITY ---")
    quality = check_annotation_quality(dataset_dir)
    lines.append(f"  Total annotations checked: {quality['total_checked']}")
    lines.append(f"  Tiny boxes (<0.5% area):   {quality['tiny_boxes']}")
    lines.append(f"  Huge boxes (>90% area):    {quality['huge_boxes']}")
    lines.append(f"  Edge-centered boxes:       {quality['edge_boxes']}")
    lines.append(f"  Duplicate labels:          {quality['duplicate_labels']}")

    lines.append("\n" + "=" * 60)
    report = "\n".join(lines)
    return report


if __name__ == "__main__":
    dataset_dir = Path(__file__).parent.parent / "datasets" / "weapon_detection_v2"

    if not dataset_dir.exists():
        print(f"Dataset not found at {dataset_dir}")
        print("Run prepare_dataset.py first to create the dataset.")
    else:
        report = generate_report(dataset_dir)
        print(report)

        # Save report
        report_path = dataset_dir / "validation_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {report_path}")

        # Visualize samples
        visualize_samples(dataset_dir, dataset_dir / "visualizations")
