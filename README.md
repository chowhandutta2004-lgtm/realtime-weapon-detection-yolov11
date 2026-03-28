# ShieldAI — Real-time Weapon Detection using YOLOv11

Real-time weapon detection system powered by **YOLOv11s**. Detects **handguns, long guns, knives, and explosives** through live webcam, CCTV/IP camera streams, or uploaded images and videos.

https://github.com/user-attachments/assets/d9b31213-03f1-489d-b9d5-cb3bcda0e524

## Features

- **Live Webcam Detection** — Real-time browser-based weapon detection via WebSocket
- **CCTV / IP Camera** — Connect RTSP, HTTP, or DroidCam streams for continuous surveillance
- **File Upload** — Batch process images and videos (up to 10 files) with drag-and-drop
- **4 Weapon Classes** — Handgun, Long Gun, Knife, Explosive with color-coded bounding boxes
- **Threat Alerts** — Instant notifications with cropped evidence thumbnails and 10s cooldown
- **Confidence Control** — Adjust detection sensitivity in real-time with a slider
- **Enhanced Knife Detection** — Quadrant-crop scanning at 1280px for detecting knives at distance
- **H.264 Video Export** — Browser-ready re-encoded detection results via FFmpeg
- **WebSocket Streaming** — Low-latency frame-by-frame analysis with binary frame protocol
- **ONNX Inference** — Optimized CPU inference using ONNX Runtime (faster than PyTorch)
- **100% Local Processing** — No external API calls, everything runs on the server
- **Cloud-Aware UI** — Gracefully disables live features on Hugging Face Spaces with GitHub link

## Weapon Classes

| Class | Color | Description |
|-------|-------|-------------|
| **Handgun** | Red | Pistols, revolvers, compact firearms |
| **Long Gun** | Dark Red | Rifles, shotguns, long-barreled firearms |
| **Knife** | Orange | Bladed weapons — tuned with lower threshold (0.30) for sharper recall at distance |
| **Explosive** | Magenta | Grenades, IEDs, explosive devices |

## How It Works

**1. Capture** — Open your webcam, connect an IP camera, or drag-and-drop any image or video file

**2. Detect** — Every frame is processed through YOLOv11s with per-class confidence thresholds, bounding box dimension filters, and optional quadrant-crop scanning for knives at distance

**3. Alert** — Threats are highlighted with color-coded bounding boxes and logged with cropped evidence

## Tech Stack

**Backend** — Python, FastAPI, Uvicorn, OpenCV, Ultralytics, ONNX Runtime, FFmpeg

**Frontend** — Vanilla HTML5, CSS3, JavaScript (zero dependencies)

**Model** — YOLOv11s fine-tuned on multi-source weapon datasets (Roboflow)

**Streaming** — WebSocket for live detection, SSE for video processing progress

**Deployment** — Docker, Hugging Face Spaces

## Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/chowhandutta2004-lgtm/realtime-weapon-detection-yolov11.git
cd realtime-weapon-detection-yolov11

# Install dependencies
cd backend
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

## Run with Docker

```bash
docker build -t shieldai .
docker run -p 7860:7860 shieldai
# Open http://localhost:7860
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect/image` | POST | Detect weapons in an uploaded image (TTA + deep knife scanning) |
| `/api/detect/video/stream` | POST | Process video with SSE progress streaming and H.264 re-encode |
| `/api/detect/video` | POST | Process video (non-streaming fallback) |
| `/api/detect/video/result/{id}` | GET | Retrieve processed video by UUID |
| `/api/detect/video/history` | GET | List all processed videos with metadata |
| `/api/detect/video/history/{id}` | DELETE | Delete a processed video and its metadata |
| `/api/detect/live` | WebSocket | Live webcam detection (binary JPEG in, JSON out) |
| `/api/detect/cctv` | WebSocket | CCTV/IP camera detection (connect/disconnect commands) |
| `/api/detect/cctv/upload` | POST | Upload video file for CCTV playback (returns opaque ID) |
| `/api/alerts` | GET | List recent threat alerts with thumbnails |
| `/api/alerts` | DELETE | Clear all alerts |

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app with lifespan model loading
│   │   ├── config.py            # Settings (thresholds, paths, limits)
│   │   ├── model.py             # WeaponDetector — YOLOv11s with per-class filtering
│   │   ├── schemas.py           # Pydantic response models
│   │   ├── alerts.py            # In-memory alert manager with thumbnails
│   │   └── routes/
│   │       ├── image.py         # Image upload + deep knife detection
│   │       ├── video.py         # Video processing with SSE + H.264 re-encode
│   │       ├── websocket.py     # Live webcam WebSocket
│   │       └── cctv.py          # CCTV/IP camera WebSocket + video upload
│   ├── models/
│   │   ├── best.pt              # Fine-tuned YOLOv11s weights
│   │   └── best.onnx            # ONNX export for CPU inference
│   └── training/
│       ├── train.py             # Training script (fine-tune YOLOv11s)
│       ├── train_colab.ipynb    # Google Colab training notebook
│       ├── train_kaggle.ipynb   # Kaggle training notebook
│       ├── dataset.yaml         # YOLO dataset config (4 classes)
│       ├── prepare_dataset.py   # Label remapping + stratified splits
│       ├── build_final_dataset.py # Auto-detect and merge all datasets
│       ├── download_datasets.py # Roboflow weapon dataset downloader
│       ├── download_knife_datasets.py # Roboflow knife dataset downloader
│       ├── merge_knife_data.py  # Merge knife data into main dataset
│       ├── evaluate.py          # Validation metrics (mAP, per-class stats)
│       └── validate_dataset.py  # Dataset QA (balance, annotation quality)
├── frontend/
│   ├── index.html               # Single-page app (Live Cam, CCTV, Upload tabs)
│   ├── css/styles.css           # Cyber-teal dark theme
│   └── js/
│       ├── app.js               # Tab switching, confidence slider, shared helpers
│       ├── webcamDetection.js   # Live webcam via WebSocket
│       ├── cctvDetection.js     # CCTV/IP camera streaming
│       ├── uploadDetection.js   # Unified image + video upload
│       └── alerts.js            # Alert panel with polling
├── demo/                        # Demo video and preview GIF
├── Dockerfile                   # Docker deployment (Python 3.11-slim)
└── README.md
```

## Training

The model was trained on multiple Roboflow weapon datasets merged into a 4-class taxonomy. Training was done on Google Colab with T4 GPU using the Ultralytics framework.

**Training pipeline:**
1. Download datasets via `download_datasets.py` and `download_knife_datasets.py` (Roboflow API)
2. Prepare and merge with `prepare_dataset.py` and `merge_knife_data.py` (50+ label remappings)
3. Validate with `validate_dataset.py` (class balance, annotation quality checks)
4. Train with `train_colab.ipynb` or `train_kaggle.ipynb` (YOLOv11s, 120 epochs)
5. Evaluate with `evaluate.py` (mAP@50, per-class precision/recall)

```bash
# Local training (requires CUDA GPU)
cd backend
python -m training.train
```

## Security

- Video endpoint IDs validated as UUID format to prevent path traversal
- CCTV upload returns opaque IDs instead of server filesystem paths
- Image uploads validated after decode (rejects corrupt files)
- Video files validated before processing (checks `VideoCapture.isOpened()`)
- CORS allows all origins (intentional for demo deployment)

## License

MIT

## Author

Built by [chowhandutta](https://github.com/chowhandutta2004-lgtm)
