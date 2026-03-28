"""Evaluate the trained weapon detection model.

Reports per-class metrics, confusion matrix, and PR curves.
"""

import json
from pathlib import Path

from ultralytics import YOLO

CLASSES = ["handgun", "long_gun", "knife", "explosive"]

TARGET_METRICS = {
    "mAP50": 0.85,
    "min_recall_per_class": 0.80,
    "min_precision_per_class": 0.85,
}


def evaluate(split: str = "test"):
    """Run evaluation on the specified split.

    Args:
        split: "val" or "test" — which split to evaluate on.
    """
    model_path = Path(__file__).parent.parent / "models" / "best.pt"
    if not model_path.exists():
        print("No fine-tuned model found. Train first with train.py or train_colab.ipynb")
        return None

    model = YOLO(str(model_path))
    data_yaml = str(Path(__file__).parent / "dataset.yaml")

    results = model.val(data=data_yaml, split=split)

    # ── Overall metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({split} set)")
    print("=" * 60)
    print(f"  mAP@50:      {results.box.map50:.4f}")
    print(f"  mAP@50-95:   {results.box.map:.4f}")

    # ── Per-class metrics ───────────────────────────────────────────────
    print(f"\n{'Class':>12s} {'Precision':>10s} {'Recall':>10s} {'mAP@50':>10s} {'mAP@50-95':>10s}")
    print("-" * 55)

    per_class = {}
    for i, cls_name in enumerate(CLASSES):
        if i < len(results.box.ap50):
            p = results.box.p[i] if i < len(results.box.p) else 0
            r = results.box.r[i] if i < len(results.box.r) else 0
            ap50 = results.box.ap50[i]
            ap = results.box.ap[i] if i < len(results.box.ap) else 0
            print(f"  {cls_name:>10s} {p:>10.4f} {r:>10.4f} {ap50:>10.4f} {ap:>10.4f}")
            per_class[cls_name] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "mAP50": round(float(ap50), 4),
                "mAP50-95": round(float(ap), 4),
            }

    # ── Target comparison ───────────────────────────────────────────────
    print(f"\n--- TARGET CHECK ---")
    map50 = results.box.map50
    passed = True

    if map50 >= TARGET_METRICS["mAP50"]:
        print(f"  [OK] mAP@50 = {map50:.4f} >= {TARGET_METRICS['mAP50']}")
    else:
        print(f"  [FAIL] mAP@50 = {map50:.4f} < {TARGET_METRICS['mAP50']}")
        passed = False

    for cls_name, metrics in per_class.items():
        if metrics["recall"] < TARGET_METRICS["min_recall_per_class"]:
            print(f"  [FAIL] {cls_name} recall = {metrics['recall']:.4f} < {TARGET_METRICS['min_recall_per_class']}")
            passed = False
        if metrics["precision"] < TARGET_METRICS["min_precision_per_class"]:
            print(f"  [FAIL] {cls_name} precision = {metrics['precision']:.4f} < {TARGET_METRICS['min_precision_per_class']}")
            passed = False

    if passed:
        print("  All targets met!")
    else:
        print("  Some targets not met. Consider more training data or hyperparameter tuning.")

    # ── Save results ────────────────────────────────────────────────────
    eval_results = {
        "split": split,
        "mAP50": round(float(map50), 4),
        "mAP50-95": round(float(results.box.map), 4),
        "per_class": per_class,
        "targets_met": passed,
    }

    output_path = Path(__file__).parent / "runs" / f"eval_{split}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Confusion matrix and PR curves are auto-saved by ultralytics
    # in the results.save_dir directory
    if results.save_dir:
        print(f"Plots saved to {results.save_dir}")

    return results


def compare_baseline(baseline_path: str | None = None):
    """Compare current model against a baseline model."""
    model_path = Path(__file__).parent.parent / "models" / "best.pt"
    data_yaml = str(Path(__file__).parent / "dataset.yaml")

    if baseline_path is None:
        baseline_path = "yolo11s.pt"  # pretrained baseline

    print(f"Evaluating baseline: {baseline_path}")
    baseline = YOLO(baseline_path)
    baseline_results = baseline.val(data=data_yaml, split="test")

    print(f"\nEvaluating fine-tuned: {model_path}")
    finetuned = YOLO(str(model_path))
    finetuned_results = finetuned.val(data=data_yaml, split="test")

    print("\n" + "=" * 60)
    print("BASELINE vs FINE-TUNED")
    print("=" * 60)
    print(f"  {'Metric':<15s} {'Baseline':>10s} {'Fine-tuned':>10s} {'Δ':>10s}")
    print("-" * 50)

    b_map50 = baseline_results.box.map50
    f_map50 = finetuned_results.box.map50
    print(f"  {'mAP@50':<15s} {b_map50:>10.4f} {f_map50:>10.4f} {f_map50 - b_map50:>+10.4f}")

    b_map = baseline_results.box.map
    f_map = finetuned_results.box.map
    print(f"  {'mAP@50-95':<15s} {b_map:>10.4f} {f_map:>10.4f} {f_map - b_map:>+10.4f}")


if __name__ == "__main__":
    evaluate(split="test")
