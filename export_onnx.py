"""
Export best.pt to ONNX format for faster inference on Intel Arc GPU.
"""

from ultralytics import YOLO
from pathlib import Path

model_path = Path(__file__).parent / "backend" / "models" / "best.pt"
model = YOLO(str(model_path))

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    half=False,
    simplify=True,
    opset=17,
)

print(f"\nONNX model exported to: {model_path.with_suffix('.onnx')}")
print("Move it to backend/models/ if needed.")
