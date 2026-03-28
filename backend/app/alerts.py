import base64
import uuid
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

from .schemas import AlertItem

MAX_ALERTS = 100
_alerts: deque[AlertItem] = deque(maxlen=MAX_ALERTS)


def add_alert(label: str, confidence: float, frame: np.ndarray, box: list[int]):
    x1, y1, x2, y2 = box
    pad = 20
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
    thumbnail = base64.b64encode(buf).decode()

    alert = AlertItem(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        label=label,
        confidence=round(confidence, 3),
        thumbnail=thumbnail,
    )
    _alerts.appendleft(alert)
    return alert


def get_alerts(limit: int = 50) -> list[AlertItem]:
    return list(_alerts)[:limit]


def clear_alerts():
    _alerts.clear()
