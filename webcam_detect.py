"""
Standalone real-time webcam weapon detection using YOLOv11s.
Press 'q' to quit, 's' to save a screenshot.
"""

import cv2
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "backend" / "models" / "best.pt"
CONFIDENCE = 0.55

LABEL_COLORS = {
    "handgun": (0, 0, 255),
    "long_gun": (0, 0, 180),
    "knife": (0, 140, 255),
    "explosive": (200, 0, 200),
}

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

print("Webcam running. Press 'q' to quit, 's' to save screenshot.")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE, iou=0.4, imgsz=640, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        label = results.names[int(box.cls[0])]
        score = float(box.conf[0])
        color = LABEL_COLORS.get(label.lower(), (0, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Weapon Detection - Press Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"screenshot_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
