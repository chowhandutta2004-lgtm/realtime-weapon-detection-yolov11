"""Fine-tune YOLOv11s on the weapon detection dataset.

Primary training should be done via train_colab.ipynb (Google Colab with T4 GPU).
This script is kept as a local reference and for environments with CUDA GPUs.
"""

import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

# ── Hyperparameters ─────────────────────────────────────────────────────────
HYPERPARAMS = {
    "model": "yolo11s.pt",
    "data": str(Path(__file__).parent / "dataset.yaml"),
    "epochs": 120,
    "patience": 20,
    "imgsz": 640,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "cos_lr": True,  # cosine annealing
    "warmup_epochs": 5,
    "amp": True,
    "workers": 8,
    # Augmentation
    "mosaic": 1.0,
    "scale": 0.5,
    "degrees": 10.0,
    "mixup": 0.1,
    "fliplr": 0.5,
    "flipud": 0.0,  # no vertical flip — unnatural for weapons
    # Saving
    "save": True,
    "save_period": 10,  # checkpoint every 10 epochs
    "name": "weapon_detector_v2",
}


def train():
    model = YOLO(HYPERPARAMS["model"])

    # Log hyperparams for reproducibility
    log_dir = Path(__file__).parent / "runs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(log_dir / f"hyperparams_{timestamp}.json", "w") as f:
        json.dump(HYPERPARAMS, f, indent=2)
    print(f"Hyperparameters saved to runs/hyperparams_{timestamp}.json")

    # Train
    train_args = {k: v for k, v in HYPERPARAMS.items() if k != "model"}
    results = model.train(**train_args)

    # Copy best weights to models dir
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    target = Path(__file__).parent.parent / "models" / "best.pt"
    target.parent.mkdir(exist_ok=True)
    if best_path.exists():
        import shutil
        shutil.copy2(best_path, target)
        print(f"Best model saved to {target}")

    return results


if __name__ == "__main__":
    train()
