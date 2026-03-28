import asyncio
import base64
import json
import logging
import os
import time
import uuid

import cv2
from fastapi import APIRouter, UploadFile, WebSocket, WebSocketDisconnect

from ..alerts import add_alert
from ..config import settings
from ..model import detector

router = APIRouter()
logger = logging.getLogger(__name__)

# Temp directory for uploaded videos to stream through CCTV pipeline
_TEMP_VIDEO_DIR = os.path.join(settings.UPLOAD_DIR, "cctv_temp")
os.makedirs(_TEMP_VIDEO_DIR, exist_ok=True)

# Map upload IDs to file paths (server-side only)
_upload_paths: dict[str, str] = {}


@router.post("/api/detect/cctv/upload")
async def upload_cctv_video(file: UploadFile):
    """Save an uploaded video and return an ID the CCTV WebSocket can open."""
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    upload_id = uuid.uuid4().hex
    filename = f"{upload_id}{ext}"
    filepath = os.path.join(_TEMP_VIDEO_DIR, filename)

    with open(filepath, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    _upload_paths[upload_id] = filepath
    logger.info(f"CCTV video uploaded: {upload_id}")
    return {"id": upload_id}

ALERT_COOLDOWN_SEC = 10


@router.websocket("/api/detect/cctv")
async def cctv_detect(ws: WebSocket):
    await ws.accept()
    logger.info("CCTV WebSocket accepted")

    cap = None
    is_file = False
    last_alert_time: dict[str, float] = {}
    current_confidence = None
    running = False

    conf_param = ws.query_params.get("confidence")
    if conf_param:
        current_confidence = float(conf_param)

    try:
        while True:
            # If we're streaming from a camera, read frames in a non-blocking loop
            if running and cap is not None and cap.isOpened():
                # Check for incoming messages (non-blocking)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if "text" in msg:
                        payload = json.loads(msg["text"])
                        if payload.get("action") == "disconnect":
                            running = False
                            cap.release()
                            cap = None
                            await ws.send_text(json.dumps({"status": "disconnected"}))
                            continue
                        if "confidence" in payload:
                            current_confidence = float(payload["confidence"])
                except asyncio.TimeoutError:
                    pass
                except (json.JSONDecodeError, ValueError):
                    pass
                except (WebSocketDisconnect, RuntimeError):
                    break

                # Read a frame from the source
                ret, frame = cap.read()
                if not ret:
                    # If it's a video file, loop back to start
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        await ws.send_text(json.dumps({
                            "error": "Lost connection to camera. Check the URL and ensure the camera is online.",
                        }))
                        running = False
                        cap.release()
                        cap = None
                        continue

                # Run detection
                annotated, detections = detector.detect_frame(frame, current_confidence)

                # Alerts with cooldown
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

                # Small yield to keep the event loop responsive
                await asyncio.sleep(0.01)

            else:
                # Waiting for a connect command
                msg = await ws.receive()
                if "text" not in msg:
                    continue

                try:
                    payload = json.loads(msg["text"])
                except (json.JSONDecodeError, ValueError):
                    continue

                if payload.get("action") == "connect":
                    camera_url = payload.get("url", "").strip()
                    if not camera_url:
                        await ws.send_text(json.dumps({
                            "error": "No camera URL provided.",
                        }))
                        continue

                    await ws.send_text(json.dumps({"status": "connecting"}))

                    # Resolve upload IDs to file paths
                    if camera_url in _upload_paths:
                        camera_url = _upload_paths[camera_url]

                    # Check if it's a local file path
                    is_file = os.path.isfile(camera_url)

                    # Try to open the camera stream or video file
                    cap = cv2.VideoCapture(camera_url)

                    # Give network streams a moment to connect
                    if not is_file:
                        await asyncio.sleep(1)

                    if not cap.isOpened():
                        await ws.send_text(json.dumps({
                            "error": "Could not connect to the camera. Verify the URL, ensure the camera is powered on and accessible from this network.",
                        }))
                        cap = None
                        continue

                    running = True
                    logger.info(f"CCTV connected: {camera_url}")
                    await ws.send_text(json.dumps({"status": "connected"}))

                elif payload.get("action") == "disconnect":
                    if cap is not None:
                        cap.release()
                        cap = None
                    running = False
                    await ws.send_text(json.dumps({"status": "disconnected"}))

                elif "confidence" in payload:
                    current_confidence = float(payload["confidence"])

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if cap is not None:
            cap.release()
            logger.info("CCTV stream released.")
