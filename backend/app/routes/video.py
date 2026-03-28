import json
import logging
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..alerts import add_alert
from ..config import settings
from ..model import detector
from ..schemas import VideoDetectionResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/detect", tags=["detection"])

# Get ffmpeg path from imageio-ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_PATH = "ffmpeg"


def _reencode_to_h264(input_path: Path, output_path: Path):
    """Re-encode mp4v video to H.264 for browser playback."""
    temp_path = input_path.with_suffix(".tmp.mp4")
    input_path.rename(temp_path)
    try:
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", str(temp_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        temp_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"H.264 re-encode failed: {e}, falling back to mp4v")
        if not output_path.exists():
            temp_path.rename(output_path)
        else:
            temp_path.unlink(missing_ok=True)


def _save_metadata(video_id: str, filename: str, total_frames: int,
                   frames_with_detections: int, summary: dict):
    """Save video processing metadata as a JSON sidecar file."""
    meta_path = Path(settings.RESULTS_DIR) / f"{video_id}_meta.json"
    meta = {
        "video_id": video_id,
        "original_filename": filename,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "detections_summary": summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def _process_video_generator(input_path: Path, output_path: Path, confidence: float | None, original_filename: str = "video.mp4"):
    """Process video frame by frame, yielding SSE progress events."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid or corrupted video file'})}\n\n"
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Aggressive frame skip based on total frames
    if total_frames <= 100:
        skip = 2
    elif total_frames <= 300:
        skip = 4
    elif total_frames <= 800:
        skip = 6
    elif total_frames <= 2000:
        skip = 10
    else:
        skip = 15

    frames_to_process = max(1, total_frames // skip)

    # Write to a temp file first (mp4v), then re-encode to H.264
    temp_output = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (w, h))

    frames_with_detections = 0
    weapon_tracker: dict[str, dict] = {}
    frame_idx = 0
    last_annotated = None
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            annotated, detections = detector.detect_frame(frame, confidence)
            if detections:
                frames_with_detections += 1
                timestamp_sec = frame_idx / fps
                for det in detections:
                    label = det["label"]
                    if label not in weapon_tracker:
                        weapon_tracker[label] = {
                            "peak_confidence": det["confidence"],
                            "first_seen_sec": timestamp_sec,
                            "last_seen_sec": timestamp_sec,
                            "frame_count": 1,
                        }
                        add_alert(label, det["confidence"], frame, det["box"])
                    else:
                        tracker = weapon_tracker[label]
                        tracker["last_seen_sec"] = timestamp_sec
                        tracker["frame_count"] += 1
                        if det["confidence"] > tracker["peak_confidence"]:
                            tracker["peak_confidence"] = det["confidence"]
            last_annotated = annotated
            processed_count += 1

            pct = min(99, int((processed_count / frames_to_process) * 100))
            yield f"data: {json.dumps({'type': 'progress', 'percent': pct, 'frame': frame_idx, 'total': total_frames})}\n\n"
        else:
            if last_annotated is None:
                last_annotated = frame

        writer.write(last_annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    input_path.unlink(missing_ok=True)

    # Re-encode to H.264 for browser playback
    _reencode_to_h264(temp_output, output_path)

    video_id = output_path.stem.replace("_result", "")
    detections_summary = {}
    for label, info in weapon_tracker.items():
        detections_summary[label] = {
            "peak_confidence": round(info["peak_confidence"], 3),
            "first_seen": _fmt_time(info["first_seen_sec"]),
            "last_seen": _fmt_time(info["last_seen_sec"]),
            "frame_count": info["frame_count"],
        }
    _save_metadata(video_id, original_filename, total_frames, frames_with_detections, detections_summary)

    yield f"data: {json.dumps({'type': 'encoding', 'percent': 99})}\n\n"

    result = {
        "type": "done",
        "video_url": f"/api/detect/video/result/{video_id}",
        "total_frames": total_frames,
        "frames_with_detections": frames_with_detections,
        "detections_summary": detections_summary,
    }
    yield f"data: {json.dumps(result)}\n\n"


@router.post("/video/stream")
async def detect_video_stream(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.1, le=1.0),
):
    """Process video with SSE progress streaming."""
    upload_dir = Path(settings.UPLOAD_DIR)
    results_dir = Path(settings.RESULTS_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(uuid.uuid4())
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = upload_dir / f"{video_id}{ext}"
    output_path = results_dir / f"{video_id}_result.mp4"

    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    return StreamingResponse(
        _process_video_generator(input_path, output_path, confidence, file.filename or "video.mp4"),
        media_type="text/event-stream",
    )


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.1, le=1.0),
):
    """Non-streaming fallback."""
    upload_dir = Path(settings.UPLOAD_DIR)
    results_dir = Path(settings.RESULTS_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(uuid.uuid4())
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = upload_dir / f"{video_id}{ext}"
    output_path = results_dir / f"{video_id}_result.mp4"

    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "Invalid or corrupted video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip = 6 if total_frames > 300 else 3

    temp_output = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (w, h))

    frames_with_detections = 0
    weapon_tracker: dict[str, dict] = {}
    frame_idx = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            annotated, detections = detector.detect_frame(frame, confidence)
            if detections:
                frames_with_detections += 1
                timestamp_sec = frame_idx / fps
                for det in detections:
                    label = det["label"]
                    if label not in weapon_tracker:
                        weapon_tracker[label] = {
                            "peak_confidence": det["confidence"],
                            "first_seen_sec": timestamp_sec,
                            "last_seen_sec": timestamp_sec,
                            "frame_count": 1,
                        }
                        add_alert(label, det["confidence"], frame, det["box"])
                    else:
                        tracker = weapon_tracker[label]
                        tracker["last_seen_sec"] = timestamp_sec
                        tracker["frame_count"] += 1
                        if det["confidence"] > tracker["peak_confidence"]:
                            tracker["peak_confidence"] = det["confidence"]
            last_annotated = annotated
        else:
            if last_annotated is None:
                last_annotated = frame

        writer.write(last_annotated)
        frame_idx += 1

    cap.release()
    writer.release()
    input_path.unlink(missing_ok=True)

    # Re-encode to H.264
    _reencode_to_h264(temp_output, output_path)

    detections_summary = {}
    for label, info in weapon_tracker.items():
        detections_summary[label] = {
            "peak_confidence": round(info["peak_confidence"], 3),
            "first_seen": _fmt_time(info["first_seen_sec"]),
            "last_seen": _fmt_time(info["last_seen_sec"]),
            "frame_count": info["frame_count"],
        }
    _save_metadata(video_id, file.filename or "video.mp4", total_frames, frames_with_detections, detections_summary)

    return VideoDetectionResponse(
        video_url=f"/api/detect/video/result/{video_id}",
        total_frames=total_frames,
        frames_with_detections=frames_with_detections,
        detections_summary=detections_summary,
    )


def _validate_video_id(video_id: str):
    if not re.match(r'^[a-f0-9\-]{36}$', video_id):
        raise HTTPException(400, "Invalid video ID")


@router.get("/video/result/{video_id}")
async def get_result_video(video_id: str):
    _validate_video_id(video_id)
    path = Path(settings.RESULTS_DIR) / f"{video_id}_result.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/video/history")
async def video_history():
    """List all processed videos with their metadata."""
    results_dir = Path(settings.RESULTS_DIR)
    if not results_dir.exists():
        return []

    videos = []
    for meta_file in sorted(results_dir.glob("*_meta.json"), reverse=True):
        try:
            meta = json.loads(meta_file.read_text())
            video_id = meta["video_id"]
            video_path = results_dir / f"{video_id}_result.mp4"
            if not video_path.exists():
                continue

            size_mb = round(video_path.stat().st_size / (1024 * 1024), 1)
            videos.append({
                "video_id": video_id,
                "original_filename": meta.get("original_filename", "video.mp4"),
                "processed_at": meta.get("processed_at", ""),
                "total_frames": meta.get("total_frames", 0),
                "frames_with_detections": meta.get("frames_with_detections", 0),
                "detections_summary": meta.get("detections_summary", {}),
                "video_url": f"/api/detect/video/result/{video_id}",
                "file_size_mb": size_mb,
            })
        except Exception:
            continue

    return videos


@router.delete("/video/history/{video_id}")
async def delete_video(video_id: str):
    """Delete a processed video and its metadata."""
    _validate_video_id(video_id)
    results_dir = Path(settings.RESULTS_DIR)
    video_path = results_dir / f"{video_id}_result.mp4"
    meta_path = results_dir / f"{video_id}_meta.json"

    if not video_path.exists():
        raise HTTPException(404, "Video not found")

    video_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return {"status": "deleted"}
