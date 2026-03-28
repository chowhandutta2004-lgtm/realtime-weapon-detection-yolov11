import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, UploadFile

from ..alerts import add_alert
from ..model import detector
from ..schemas import Detection, ImageDetectionResponse

router = APIRouter(prefix="/api/detect", tags=["detection"])


@router.post("/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.1, le=1.0),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    annotated, detections = detector.detect_frame(frame, confidence, use_tta=True, deep_knife=True)

    # Generate alerts for detections
    for det in detections:
        add_alert(det["label"], det["confidence"], frame, det["box"])

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode()

    return ImageDetectionResponse(
        image=img_b64,
        detections=[Detection(**d) for d in detections],
        weapon_detected=len(detections) > 0,
    )
