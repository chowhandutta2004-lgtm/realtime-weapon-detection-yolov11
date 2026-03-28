# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack real-time weapon detection app using YOLOv11s. FastAPI backend + vanilla JS frontend with WebSocket support for live camera streaming. 4-class detection: handgun, long_gun, knife, explosive.

## Dependencies

**Backend** (`backend/requirements.txt`):
- `fastapi` — Web framework
- `uvicorn[standard]` — ASGI server
- `ultralytics>=8.3.0` — YOLOv11 model inference and training
- `opencv-python` — Image/video processing
- `pydantic-settings` — Configuration management
- `python-multipart` — File upload handling
- `aiofiles` — Async file I/O
- `imageio-ffmpeg` — Video codec support (H.264 re-encoding)
- `onnx` — Model export format
- `numpy`, `Pillow` — Array/image utilities

**Frontend**: No dependencies. Vanilla HTML5, CSS3, JavaScript (no npm/node).

**Training**: Google Colab or Kaggle with T4 GPU. Uses `ultralytics` + `torch` (installed in Colab/Kaggle by default).

## Build & Run Commands

```bash
# Install dependencies
cd backend
pip install -r requirements.txt

# Run dev server (serves both API and frontend static files)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Standalone webcam detection (no server needed)
python webcam_detect.py

# Export model to ONNX
python export_onnx.py

# Local training (requires CUDA GPU)
cd backend
python -m training.train
```

Training is done on Google Colab or Kaggle (user has Intel Arc 140V GPU locally, no CUDA). Use `training/train_colab.ipynb` or `training/train_kaggle.ipynb`.

## Architecture

**Backend** (`backend/app/`):
- `main.py` — FastAPI app with lifespan-managed model loading, serves frontend from `frontend/` as static files
- `model.py` — `WeaponDetector` class wrapping ultralytics YOLO. Color-coded labels, dynamic imgsz (640/1280 based on input resolution)
- `config.py` — pydantic-settings config: MODEL_PATH, CONFIDENCE_THRESHOLD (0.25), MAX_UPLOAD_SIZE (100MB), WEBSOCKET_MAX_FPS (15)
- `alerts.py` — In-memory deque (max 100), 10s cooldown per weapon type for deduplication
- `schemas.py` — Pydantic response models

**Routes** (`backend/app/routes/`):
- `image.py` — POST `/api/detect/image` — TTA enabled, returns base64 annotated image
- `video.py` — POST `/api/detect/video/stream` — SSE progress streaming, adaptive frame skipping, H.264 re-encode
- `websocket.py` — WS `/api/detect/live` — Binary frame in, JSON detections out, real-time confidence slider updates

**Frontend** (`frontend/`):
- No build step. Vanilla HTML/CSS/JS
- Modules: `webcamDetection.js`, `imageDetection.js`, `videoDetection.js`, `videoHistory.js`, `alerts.js`, `app.js`
- Tabs: Live Camera, Image Upload, Video Upload, Video History, Alerts

**Training** (`backend/training/`):
- `prepare_dataset.py` — Remaps mixed labels to 4-class taxonomy, creates train/val/test splits
- `dataset.yaml` — YOLO dataset config pointing to `datasets/weapon_detection_v2/`
- `download_datasets.py` / `merge_knife_data.py` — Roboflow dataset acquisition and merging
- `evaluate.py` — Validation metrics (mAP@50, per-class stats)

## Key Technical Details

- Model falls back to pretrained `yolo11s.pt` if fine-tuned `models/best.pt` is missing
- Video processing re-encodes mp4v → H.264 for browser compatibility
- Alerts are in-memory only (lost on restart)
- CORS allows all origins
- Frontend updates confidence threshold in real-time over WebSocket during live detection
- Processed videos saved to `backend/results/` with `{video_id}_meta.json` sidecar files

## Dataset

- **Source**: Multiple Roboflow datasets (weapon + knife collections), downloaded via `download_datasets.py`
- **Format**: YOLO format (images + labels in txt)
- **4 classes**: handgun (0), long_gun (1), knife (2), explosive (3)
- **Structure**: `datasets/weapon_detection_v2/` with `images/train`, `images/val`, `images/test` and matching `labels/` dirs
- **Label remapping**: `prepare_dataset.py` unifies different source label formats into the 4-class taxonomy
- **Knife data**: Merged from 5 separate Roboflow knife datasets via `merge_knife_data.py`
- **Dataset config**: `backend/training/dataset.yaml`

## Weapon Class Colors (BGR for OpenCV)

| Class | Color |
|-------|-------|
| handgun | Red (0,0,255) |
| long_gun | Dark Red (0,0,180) |
| knife | Orange (0,140,255) |
| explosive | Magenta (200,0,200) |
