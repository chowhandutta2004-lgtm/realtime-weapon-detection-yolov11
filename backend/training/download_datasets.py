"""
Download and organize datasets for weapon detection training.

Datasets used:
1. Roboflow "Weapon yolo8" (~10K images, 12 classes) — best all-around
2. YouTube-GDD (~5K images, guns) — diverse firearms
3. DaSCI Knife (~2K images, knives) — diverse knife types

Total: ~17K images → after remapping to 4 classes, expect 12-15K usable images.

Usage:
    python download_datasets.py --roboflow-key YOUR_API_KEY
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "datasets" / "_raw"


def install_package(package: str):
    """Install a Python package if not already installed."""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


# ── Dataset 1: Roboflow "Weapon yolo8" ──────────────────────────────────────

def download_roboflow_weapon_yolo8(api_key: str):
    """Download the Weapon yolo8 dataset from Roboflow (~10K images).

    Classes include: grenade, gun, knife, missile, pistol, handgun,
    heavyweapon, rifle, and more.
    """
    install_package("roboflow")
    from roboflow import Roboflow

    print("\n" + "=" * 60)
    print("DOWNLOADING: Roboflow 'Weapon yolo8' (~10K images)")
    print("=" * 60)

    dest = RAW_DIR / "roboflow_weapon_yolo8"
    if dest.exists() and any(dest.rglob("*.txt")):
        print(f"Already downloaded at {dest}, skipping.")
        return dest

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("edi-detection").project("weapon-yolo8")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(dest))
    print(f"Downloaded to {dest}")
    return dest


# ── Dataset 2: Roboflow "YOLO Weapon Detection" ────────────────────────────

def download_roboflow_yolo_weapon(api_key: str):
    """Download the YOLO Weapon Detection dataset (~4.6K images).

    Classes: gunmen, rifle, blunt_object, knife, knife_attacker,
    person, pistol, shot-gun, submachine-gun
    """
    install_package("roboflow")
    from roboflow import Roboflow

    print("\n" + "=" * 60)
    print("DOWNLOADING: Roboflow 'YOLO Weapon Detection' (~4.6K images)")
    print("=" * 60)

    dest = RAW_DIR / "roboflow_yolo_weapon"
    if dest.exists() and any(dest.rglob("*.txt")):
        print(f"Already downloaded at {dest}, skipping.")
        return dest

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("weapon-detect-qbsiw").project("yolo-weapon-detection")
    version = project.version(1)
    dataset = version.download("yolov8", location=str(dest))
    print(f"Downloaded to {dest}")
    return dest


# ── Dataset 3: YouTube-GDD ─────────────────────────────────────────────────

def download_youtube_gdd():
    """Download YouTube-GDD gun detection dataset (~5K images).

    Must be downloaded manually from Google Drive.
    Classes: person, gun (we only use gun)
    """
    print("\n" + "=" * 60)
    print("DATASET: YouTube-GDD (~5K images)")
    print("=" * 60)

    dest = RAW_DIR / "youtube_gdd"
    if dest.exists() and any(dest.rglob("*.txt")):
        print(f"Already exists at {dest}, skipping.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Try gdown for Google Drive download
    try:
        install_package("gdown")
        import gdown

        print("Downloading from Google Drive (this may take a few minutes)...")
        # Google Drive file ID for YouTube-GDD images
        file_id = "1TH6kSx7WoFRrUPbxcDGYBrFrYUI1ReWa"
        zip_path = dest / "youtube_gdd.zip"
        gdown.download(id=file_id, output=str(zip_path), quiet=False)

        if zip_path.exists():
            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)
            zip_path.unlink()
            print(f"Extracted to {dest}")

            # Also download gun-only labels from GitHub
            print("Downloading labels...")
            labels_url = "https://github.com/UCAS-GYX/YouTube-GDD/raw/main/labels_only_gun.zip"
            labels_zip = dest / "labels_only_gun.zip"
            subprocess.run(
                ["curl", "-L", "-o", str(labels_zip), labels_url],
                check=True,
            )
            if labels_zip.exists():
                with zipfile.ZipFile(labels_zip, "r") as zf:
                    zf.extractall(dest)
                labels_zip.unlink()
                print("Labels extracted.")
        return dest

    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n>>> MANUAL DOWNLOAD REQUIRED <<<")
        print("1. Go to: https://drive.google.com/file/d/1TH6kSx7WoFRrUPbxcDGYBrFrYUI1ReWa/view")
        print("2. Download the zip file")
        print(f"3. Extract to: {dest}")
        print("4. Download labels from: https://github.com/UCAS-GYX/YouTube-GDD")
        print(f"5. Put labels in: {dest}/labels_only_gun/")
        print("6. Re-run this script\n")
        return None


# ── Dataset 4: DaSCI Knife ──────────────────────────────────────────────────

def download_dasci_knife():
    """Download DaSCI OD-WeaponDetection knife dataset (~2K images).

    Must be downloaded manually via Google Drive links in the GitHub repo.
    """
    print("\n" + "=" * 60)
    print("DATASET: DaSCI Knife Detection (~2K images)")
    print("=" * 60)

    dest = RAW_DIR / "dasci_knife"
    if dest.exists() and any(dest.rglob("*.txt")):
        print(f"Already exists at {dest}, skipping.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    # Try dataset-tools (dataset ninja) for automated download
    try:
        install_package("dataset-tools")
        print("Downloading via dataset-tools...")
        subprocess.run(
            [
                sys.executable, "-c",
                f"import dataset_tools as dtools; dtools.download(dataset='OD-WeaponDetection: Knife Detection', dst_dir='{dest}')"
            ],
            check=True,
            timeout=600,
        )
        print(f"Downloaded to {dest}")
        return dest
    except Exception as e:
        print(f"dataset-tools download failed: {e}")
        print("\n>>> MANUAL DOWNLOAD REQUIRED <<<")
        print("1. Go to: https://github.com/ari-dasci/OD-WeaponDetection")
        print("2. Follow the Google Drive links in the README for 'Knife Detection'")
        print(f"3. Extract to: {dest}")
        print("4. Re-run this script\n")
        return None


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download weapon detection datasets")
    parser.add_argument(
        "--roboflow-key",
        type=str,
        help="Roboflow API key (get free at https://app.roboflow.com/settings/api)",
    )
    parser.add_argument(
        "--skip-roboflow",
        action="store_true",
        help="Skip Roboflow datasets (if already downloaded)",
    )
    parser.add_argument(
        "--skip-youtube-gdd",
        action="store_true",
        help="Skip YouTube-GDD dataset",
    )
    parser.add_argument(
        "--skip-dasci",
        action="store_true",
        help="Skip DaSCI knife dataset",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Roboflow datasets
    if not args.skip_roboflow:
        if not args.roboflow_key:
            print("=" * 60)
            print("ROBOFLOW API KEY REQUIRED")
            print("=" * 60)
            print("1. Go to https://app.roboflow.com/ and create a free account")
            print("2. Go to Settings → API → copy your Private API Key")
            print("3. Re-run with: python download_datasets.py --roboflow-key YOUR_KEY")
            print("")
            args.roboflow_key = input("Or paste your API key here (Enter to skip): ").strip()

        if args.roboflow_key:
            results["roboflow_weapon_yolo8"] = download_roboflow_weapon_yolo8(args.roboflow_key)
            results["roboflow_yolo_weapon"] = download_roboflow_yolo_weapon(args.roboflow_key)
        else:
            print("Skipping Roboflow datasets (no API key)")

    # YouTube-GDD
    if not args.skip_youtube_gdd:
        results["youtube_gdd"] = download_youtube_gdd()

    # DaSCI Knife
    if not args.skip_dasci:
        results["dasci_knife"] = download_dasci_knife()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, path in results.items():
        status = "✓ Ready" if path and path.exists() else "✗ Manual download needed"
        print(f"  {name:30s} {status}")

    print(f"\nRaw datasets directory: {RAW_DIR}")
    print("\nNext step: Run build_final_dataset.py to merge and prepare for training.")


if __name__ == "__main__":
    main()
