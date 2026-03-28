# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack real-time weapon detection app using YOLOv11s. FastAPI backend + vanilla JS frontend with WebSocket support for live camera and CCTV streaming. 4-class detection: handgun, long_gun, knife, explosive. Deployed on Hugging Face Spaces via Docker, with cloud-aware frontend that disables live features on free CPU instances.

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
- `onnxruntime` — ONNX inference engine (faster than PyTorch on CPU)
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

# Run with Docker
docker build -t shieldai .
docker run -p 7860:7860 shieldai

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
- `main.py` — FastAPI app with lifespan-managed model loading, serves frontend from `frontend/` as static files, falls back to Docker path `/app/frontend`
- `model.py` — `WeaponDetector` class wrapping ultralytics YOLO:
  - Prefers ONNX model on CPU (faster inference), falls back to `.pt`, then pretrained `yolo11s.pt`
  - Dynamic `imgsz`: 320 (live/fast), 640 (uploads), 1280 (deep_knife mode)
  - Per-class bounding box filters (`MAX_BOX_DIM_RATIO`, `MIN_BOX_AREA`) to reduce false positives
  - `deep_knife` mode: quadrant-crop scanning at 1280px for detecting distant knives
  - IoU-based deduplication across quadrant detections
  - Color-coded bounding box annotations (BGR for OpenCV)
- `config.py` — pydantic-settings config: MODEL_PATH, CONFIDENCE_THRESHOLD (0.65), KNIFE_CONFIDENCE_THRESHOLD (0.30), MIN_BOX_AREA (900), KNIFE_MIN_BOX_AREA (100), MAX_BOX_FRAME_RATIO (0.25), MAX_UPLOAD_SIZE (100MB), WEBSOCKET_MAX_FPS (15)
- `alerts.py` — In-memory deque (max 100), base64 JPEG thumbnails cropped from detection boxes
- `schemas.py` — Pydantic response models: Detection, ImageDetectionResponse, VideoDetectionResponse, AlertItem

**Routes** (`backend/app/routes/`):
- `image.py` — POST `/api/detect/image` — TTA + deep_knife enabled, validates decoded frame, returns base64 annotated image + detection list
- `video.py` — POST `/api/detect/video/stream` — SSE progress streaming, adaptive frame skipping (2-15 based on length), H.264 re-encode via ffmpeg, JSON metadata sidecar files. UUID-validated result retrieval and deletion endpoints. Also: GET `/api/detect/video/result/{video_id}`, GET `/api/detect/video/history`, DELETE `/api/detect/video/history/{video_id}`
- `websocket.py` — WS `/api/detect/live` — Binary JPEG frame in, JSON detections + base64 annotated frame out, real-time confidence slider updates via text messages, 10s alert cooldown per weapon type
- `cctv.py` — WS `/api/detect/cctv` — Connects to RTSP/HTTP camera streams or uploaded video files via `cv2.VideoCapture`. Upload endpoint POST `/api/detect/cctv/upload` returns opaque ID (not server path) for security. Supports connect/disconnect commands, loops video files, handles stream loss gracefully

**Frontend** (`frontend/`):
- No build step. Vanilla HTML/CSS/JS with dark cyber-teal theme
- Modules: `app.js` (tabs, confidence slider, shared helpers), `webcamDetection.js`, `cctvDetection.js`, `uploadDetection.js` (unified image+video), `alerts.js`
- Tabs: Live Cam, CCTV, Upload (images + videos combined)
- Cloud-aware: detects HF Spaces hostname and disables live webcam/CCTV with "Local Only" message + GitHub link
- Toast notifications for all errors (no browser `alert()` calls)

**Training** (`backend/training/`):
- `train.py` — Fine-tunes YOLOv11s, copies best.pt to `backend/models/`
- `train_colab.ipynb` / `train_kaggle.ipynb` — Primary training notebooks (T4 GPU)
- `prepare_dataset.py` — Remaps 50+ source labels to 4-class taxonomy, handles YOLO/COCO/VOC formats, creates stratified 80/10/10 splits
- `build_final_dataset.py` — Auto-detects downloaded datasets and runs full pipeline
- `dataset.yaml` — YOLO dataset config pointing to `datasets/weapon_detection_v2/`
- `download_datasets.py` — Downloads weapon datasets from Roboflow (requires API key)
- `download_knife_datasets.py` — Downloads 5 dedicated knife datasets from Roboflow
- `merge_knife_data.py` — Merges knife datasets into weapon_detection_v2, remaps labels
- `evaluate.py` — Validation metrics (mAP@50, mAP@50-95, per-class precision/recall), target mAP50 >= 0.85
- `validate_dataset.py` — Dataset QA: class balance, annotation quality, duplicate detection

## Key Technical Details

- Model loading priority: ONNX (`best.onnx`) > PyTorch (`best.pt`) > pretrained fallback (`yolo11s.pt`)
- Knife detection uses lower confidence threshold (0.30 vs 0.65) for better recall at distance
- Video processing re-encodes mp4v → H.264 via imageio-ffmpeg for browser compatibility
- Alerts are in-memory only (lost on restart), deque with max 100 entries
- CORS allows all origins (intentional for demo/portfolio)
- Frontend updates confidence threshold in real-time over WebSocket during live detection
- Processed videos saved to `backend/results/` with `{video_id}_meta.json` sidecar files
- Video endpoints validate `video_id` as UUID format to prevent path traversal
- CCTV upload returns opaque ID instead of server filesystem path
- Image upload validates `cv2.imdecode` result before processing
- Video processing validates `cap.isOpened()` before frame loop
- Git LFS tracks `.pt`, `.onnx`, `.mp4`, `.gif` files (configured in `.gitattributes`)

## Dataset

- **Source**: Multiple Roboflow datasets (weapon + knife collections), downloaded via `download_datasets.py` and `download_knife_datasets.py`
- **Format**: YOLO format (images + labels in txt)
- **4 classes**: handgun (0), long_gun (1), knife (2), explosive (3)
- **Structure**: `datasets/weapon_detection_v2/` with `images/train`, `images/val`, `images/test` and matching `labels/` dirs
- **Label remapping**: `prepare_dataset.py` unifies 50+ different source label formats into the 4-class taxonomy
- **Knife data**: Merged from 5 separate Roboflow knife datasets via `merge_knife_data.py`
- **Validation**: `validate_dataset.py` checks class balance, annotation quality, box dimensions
- **Dataset config**: `backend/training/dataset.yaml`

## Weapon Class Colors (BGR for OpenCV)

| Class | Color |
|-------|-------|
| handgun | Red (0,0,255) |
| long_gun | Dark Red (0,0,180) |
| knife | Orange (0,140,255) |
| explosive | Magenta (200,0,200) |

## Deployment

- **Docker**: `Dockerfile` uses `python:3.11-slim`, installs `libgl1`, `libglib2.0-0`, `ffmpeg`. Exposes port 7860 (HF Spaces default)
- **Hugging Face Spaces**: Frontend auto-detects `.hf.space` hostname and disables live features
- **Local**: Runs on port 8000 with `uvicorn --reload`
