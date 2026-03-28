"""
Build the final merged dataset from downloaded raw datasets.

Run this AFTER download_datasets.py has downloaded all raw datasets.
This script:
  1. Detects which datasets were downloaded
  2. Identifies their format and class names
  3. Converts, remaps, merges, validates, and splits them
  4. Outputs the final dataset to backend/datasets/weapon_detection_v2/

Usage:
    python build_final_dataset.py
"""

from pathlib import Path

from prepare_dataset import (
    CLASSES,
    convert_coco_to_yolo,
    convert_voc_to_yolo,
    convert_yolo_to_yolo,
    merge_datasets,
    run_full_pipeline,
    stratified_split,
    validate_annotations,
)

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "datasets" / "_raw"
FINAL_DIR = BASE_DIR / "datasets" / "weapon_detection_v2"


def detect_datasets() -> list[dict]:
    """Auto-detect which datasets have been downloaded and build configs."""
    configs = []

    # ── Roboflow Weapon yolo8 ───────────────────────────────────────────
    rf_weapon = RAW_DIR / "roboflow_weapon_yolo8"
    if rf_weapon.exists():
        # Roboflow downloads have train/valid/test splits already
        # Find the data.yaml to get class names
        yaml_path = None
        for f in rf_weapon.rglob("data.yaml"):
            yaml_path = f
            break

        class_names = _parse_roboflow_classes(yaml_path) if yaml_path else [
            "grenade", "gun", "knife", "missile", "pistol", "handgun",
            "heavyweapon", "rifle", "bomb", "tank", "military-vehicle", "RPG",
        ]

        # Add each split as a separate source
        for split in ["train", "valid", "test"]:
            split_dir = rf_weapon / split
            if not split_dir.exists():
                # Try nested structure
                for d in rf_weapon.rglob(split):
                    if d.is_dir():
                        split_dir = d.parent  # go up to find images/labels
                        break
            if split_dir.exists() and (
                (split_dir / "images").exists() or any(split_dir.glob("*.jpg"))
            ):
                configs.append({
                    "path": str(split_dir),
                    "format": "yolo",
                    "class_names": class_names,
                    "prefix": f"rfwy_{split}",
                })
        print(f"  [OK] Roboflow Weapon yolo8: {len([c for c in configs if c['prefix'].startswith('rfwy')])} splits found")
        print(f"    Classes: {class_names}")

    # ── Roboflow YOLO Weapon Detection ──────────────────────────────────
    rf_yolo = RAW_DIR / "roboflow_yolo_weapon"
    if rf_yolo.exists():
        yaml_path = None
        for f in rf_yolo.rglob("data.yaml"):
            yaml_path = f
            break

        class_names = _parse_roboflow_classes(yaml_path) if yaml_path else [
            "gunmen", "rifle", "blunt_object", "knife", "knife_attacker",
            "person", "pistol", "shot-gun", "submachine-gun",
        ]

        for split in ["train", "valid", "test"]:
            split_dir = rf_yolo / split
            if split_dir.exists():
                configs.append({
                    "path": str(split_dir),
                    "format": "yolo",
                    "class_names": class_names,
                    "prefix": f"rfyw_{split}",
                })
        print(f"  [OK] Roboflow YOLO Weapon: {len([c for c in configs if c['prefix'].startswith('rfyw')])} splits found")
        print(f"    Classes: {class_names}")

    # ── YouTube-GDD ─────────────────────────────────────────────────────
    yt_dir = RAW_DIR / "youtube_gdd"
    if yt_dir.exists():
        # YouTube-GDD structure:
        #   YouTube-GDD/images/{train,val,test}/*.jpg
        #   labels_only_gun/{train,val}/*.txt
        # We need to assemble per-split dirs with images/ and labels/ subdirs

        yt_images_base = yt_dir / "YouTube-GDD" / "images"
        yt_labels_base = yt_dir / "labels_only_gun"

        yt_count = 0
        for split in ["train", "val"]:
            split_images = yt_images_base / split
            split_labels = yt_labels_base / split
            if split_images.exists() and split_labels.exists():
                # Create a staging dir with images/ and labels/ symlinks
                staging = RAW_DIR / "youtube_gdd_prepared" / split
                staging_images = staging / "images"
                staging_labels = staging / "labels"
                staging_images.mkdir(parents=True, exist_ok=True)
                staging_labels.mkdir(parents=True, exist_ok=True)

                # Copy/link files into expected structure
                import shutil
                img_count = 0
                for img_file in split_images.glob("*.jpg"):
                    label_file = split_labels / f"{img_file.stem}.txt"
                    if label_file.exists():
                        # Only copy if label exists
                        dst_img = staging_images / img_file.name
                        dst_lbl = staging_labels / label_file.name
                        if not dst_img.exists():
                            shutil.copy2(img_file, dst_img)
                        if not dst_lbl.exists():
                            shutil.copy2(label_file, dst_lbl)
                        img_count += 1

                if img_count > 0:
                    configs.append({
                        "path": str(staging),
                        "format": "yolo",
                        "class_names": ["gun"],
                        "prefix": f"ytgdd_{split}",
                    })
                    yt_count += img_count

        # Also include test split images (they have labels in YouTube-GDD structure)
        test_images = yt_images_base / "test"
        test_labels_dir = yt_dir / "YouTube-GDD" / "labels" / "test" if (yt_dir / "YouTube-GDD" / "labels" / "test").exists() else None
        if test_labels_dir is None:
            # Check alternate label locations
            for candidate in [yt_dir / "labels_only_gun" / "test", yt_dir / "YouTube-GDD" / "test" / "labels"]:
                if candidate.exists():
                    test_labels_dir = candidate
                    break

        if test_images.exists() and test_labels_dir and test_labels_dir.exists():
            staging = RAW_DIR / "youtube_gdd_prepared" / "test"
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "images").mkdir(exist_ok=True)
            (staging / "labels").mkdir(exist_ok=True)
            import shutil
            for img_file in test_images.glob("*.jpg"):
                label_file = test_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(img_file, staging / "images" / img_file.name)
                    shutil.copy2(label_file, staging / "labels" / label_file.name)
                    yt_count += 1
            if any((staging / "labels").glob("*.txt")):
                configs.append({
                    "path": str(staging),
                    "format": "yolo",
                    "class_names": ["gun"],
                    "prefix": "ytgdd_test",
                })

        if yt_count > 0:
            print(f"  [OK] YouTube-GDD: {yt_count} images with labels prepared")
        else:
            print(f"  [FAIL] YouTube-GDD: directory exists but can't match images to labels")
            print(f"    Contents: {[p.name for p in yt_dir.iterdir()]}")

    # ── DaSCI Knife ─────────────────────────────────────────────────────
    dasci_dir = RAW_DIR / "dasci_knife"
    if dasci_dir.exists():
        # Check for different possible structures
        # dataset-tools may download as supervisely or other format
        # Check for YOLO format first
        has_yolo = any(dasci_dir.rglob("*.txt")) and any(
            dasci_dir.rglob("*.jpg")
        )

        # Check for COCO format
        coco_json = None
        for f in dasci_dir.rglob("*.json"):
            if "annotation" in f.name.lower() or "instance" in f.name.lower():
                coco_json = f
                break

        # Check for VOC format
        has_voc = any(dasci_dir.rglob("*.xml"))

        if has_yolo:
            configs.append({
                "path": str(dasci_dir),
                "format": "yolo",
                "class_names": ["knife"],
                "prefix": "dasci",
            })
            print(f"  [OK] DaSCI Knife (YOLO format)")
        elif coco_json:
            img_dir = coco_json.parent
            for candidate in [dasci_dir / "images", coco_json.parent / "images"]:
                if candidate.exists():
                    img_dir = candidate
                    break
            configs.append({
                "path": str(img_dir.parent),
                "format": "coco",
                "coco_json": str(coco_json),
                "prefix": "dasci",
            })
            print(f"  [OK] DaSCI Knife (COCO format, json: {coco_json.name})")
        elif has_voc:
            configs.append({
                "path": str(dasci_dir),
                "format": "voc",
                "prefix": "dasci",
            })
            print(f"  [OK] DaSCI Knife (VOC format)")
        else:
            print(f"  [FAIL] DaSCI Knife: can't determine format")
            print(f"    Contents: {[p.name for p in dasci_dir.iterdir()]}")

    return configs


def _parse_roboflow_classes(yaml_path: Path) -> list[str]:
    """Parse class names from a Roboflow data.yaml file."""
    if yaml_path is None or not yaml_path.exists():
        return []

    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, dict):
            # {0: 'class1', 1: 'class2', ...}
            return [names[k] for k in sorted(names.keys())]
        return list(names)
    except ImportError:
        # Fallback: parse yaml manually for 'names' field
        import re
        with open(yaml_path, "r") as f:
            content = f.read()
        # Try to find names: ['class1', 'class2', ...]
        match = re.search(r"names:\s*\[([^\]]+)\]", content)
        if match:
            names_str = match.group(1)
            return [n.strip().strip("'\"") for n in names_str.split(",")]
        # Try names as dict or list format
        names = []
        in_names = False
        for line in content.split("\n"):
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                if line.strip().startswith("-"):
                    names.append(line.strip().lstrip("- ").strip("'\""))
                elif line.strip() and not line.startswith(" "):
                    break
        return names


def main():
    print("=" * 60)
    print("BUILD FINAL WEAPON DETECTION DATASET")
    print("=" * 60)
    print(f"Raw datasets dir:   {RAW_DIR}")
    print(f"Output dir:         {FINAL_DIR}")
    print(f"Target classes:     {CLASSES}")
    print()

    if not RAW_DIR.exists():
        print("ERROR: No raw datasets found!")
        print("Run download_datasets.py first.")
        return

    # Detect available datasets
    print("Detecting downloaded datasets...")
    configs = detect_datasets()

    if not configs:
        print("\nERROR: No usable datasets found!")
        print(f"Check that datasets are properly extracted in {RAW_DIR}/")
        return

    print(f"\nFound {len(configs)} dataset sources.")
    print("\nStarting pipeline...\n")

    # Run the full pipeline
    results = run_full_pipeline(
        source_configs=configs,
        final_output_dir=FINAL_DIR,
        seed=42,
    )

    # Run validation
    print("\nRunning validation...")
    from validate_dataset import generate_report, visualize_samples

    report = generate_report(FINAL_DIR)
    print(report)

    report_path = FINAL_DIR / "validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    visualize_samples(FINAL_DIR, FINAL_DIR / "visualizations")

    print(f"\n{'=' * 60}")
    print("DONE! Dataset is ready for training.")
    print(f"{'=' * 60}")
    print(f"\nDataset location: {FINAL_DIR}")
    print(f"Validation report: {report_path}")
    print(f"\nNext steps:")
    print(f"  1. Review the validation report above")
    print(f"  2. Check sample annotations in {FINAL_DIR}/visualizations/")
    print(f"  3. Zip the dataset: cd {FINAL_DIR.parent} && zip -r weapon_detection_v2.zip weapon_detection_v2/")
    print(f"  4. Upload to Google Drive")
    print(f"  5. Open train_colab.ipynb in Google Colab and train!")


if __name__ == "__main__":
    main()
