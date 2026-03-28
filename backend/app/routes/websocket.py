import base64
import json
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..alerts import add_alert
from ..model import detector

router = APIRouter()

ALERT_COOLDOWN_SEC = 10  # Only create one alert per weapon type per 10 seconds


@router.websocket("/api/detect/live")
async def websocket_detect(ws: WebSocket):
    await ws.accept()
    last_alert_time: dict[str, float] = {}
    # Default confidence from query params, can be overridden per-frame
    current_confidence = None
    conf_param = ws.query_params.get("confidence")
    if conf_param:
        current_confidence = float(conf_param)

    try:
        while True:
            msg = await ws.receive()

            # Handle text messages (confidence updates)
            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                    if "confidence" in payload:
                        current_confidence = float(payload["confidence"])
                except (json.JSONDecodeError, ValueError):
                    pass
                continue

            # Handle binary messages (frames)
            data = msg.get("bytes", b"")
            if not data:
                continue

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await ws.send_text(json.dumps({"error": "Invalid frame"}))
                continue

            annotated, detections = detector.detect_frame(frame, current_confidence)

            # Only create alerts with cooldown to prevent spam
            now = time.time()
            for det in detections:
                label = det["label"]
                last_time = last_alert_time.get(label, 0)
                if now - last_time > ALERT_COOLDOWN_SEC:
                    last_alert_time[label] = now
                    add_alert(label, det["confidence"], frame, det["box"])

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buf).decode()

            await ws.send_text(json.dumps({
                "image": img_b64,
                "detections": detections,
                "weapon_detected": len(detections) > 0,
            }))
    except WebSocketDisconnect:
        pass
