"""
Download additional knife-focused datasets to improve knife detection accuracy.

New datasets:
1. Roboflow "Knife" by Sanket Kulkarni (~7K images, knife class)
2. Roboflow "CCTV Knife Detection" by Simuletic (CCTV knife scenes)
3. Roboflow "Knife Dataset" by Porject (~500 images)
4. Roboflow "knife-detection" by srm (~467 images)
5. Roboflow "knife-dataset-new" by workspace (~4K images)

Total: ~12K+ additional knife images

Usage:
    python download_knife_datasets.py --roboflow-key YOUR_API_KEY
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "datasets" / "_raw_knives"


def install_package(package: str):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def download_dataset(api_key: str, workspace: str, project: str, version: int, name: str, expected: str):
    """Download a Roboflow dataset."""
    install_package("roboflow")
    from roboflow import Roboflow

    dest = RAW_DIR / name
    if dest.exists() and any(dest.rglob("*.txt")):
        print(f"  [SKIP] {name} already downloaded")
        return dest

    print(f"\n  Downloading: {name} ({expected})...")
    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        ver = proj.version(version)
        ver.download("yolov8", location=str(dest))
        print(f"  [OK] Downloaded to {dest}")
        return dest
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download knife-focused datasets")
    parser.add_argument("--roboflow-key", type=str, required=True, help="Roboflow API key")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING KNIFE-FOCUSED DATASETS")
    print("=" * 60)

    datasets = [
        # (workspace, project, version, folder_name, description)
        ("sanket-kulkarni", "knife-eydvx", 1, "roboflow_knife_7k", "~7K knife images"),
        ("simuletic", "cctv-knife-detection-dataset-zkkaf", 1, "roboflow_cctv_knife", "CCTV knife scenes"),
        ("porject", "knife-dataset", 1, "roboflow_knife_500", "~500 knife images"),
        ("srm-6ernq", "knife-detection-hwlth", 1, "roboflow_knife_srm", "~467 knife images"),
        ("workspace-zqssx", "knife-dataset-new", 1, "roboflow_knife_4k", "~4K knife images"),
    ]

    results = {}
    for ws, proj, ver, name, desc in datasets:
        results[name] = download_dataset(args.roboflow_key, ws, proj, ver, name, desc)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total_ok = 0
    for name, path in results.items():
        if path and path.exists():
            # Count images
            img_count = sum(1 for _ in path.rglob("*.jpg")) + sum(1 for _ in path.rglob("*.png"))
            print(f"  {name:35s} [OK] {img_count} images")
            total_ok += 1
        else:
            print(f"  {name:35s} [FAIL]")

    print(f"\n  {total_ok}/{len(datasets)} datasets downloaded")
    print(f"  Raw knife data: {RAW_DIR}")
    print(f"\nNext: Run merge_knife_data.py to merge with existing training data")


if __name__ == "__main__":
    main()
