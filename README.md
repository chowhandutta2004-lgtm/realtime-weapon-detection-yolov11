# ShieldAI — Real-time Weapon Detection using YOLOv11

Real-time weapon detection system powered by **YOLOv11s**. Detects **handguns, long guns, knives, and explosives** through live webcam, CCTV/IP camera streams, or uploaded images and videos.

<p align="center">
  <img src="demo/demo-preview.gif" alt="ShieldAI Demo Preview" width="800">
</p>

> **Full demo video:** [Watch the complete 1-minute detection demo](https://github.com/chowhandutta2004-lgtm/realtime-weapon-detection-yolov11/blob/main/demo/weapon-detection-demo.mp4)

## Live Demo

> **[Try ShieldAI Live](https://huggingface.co/spaces/chowhandutta/realtime-weapon-detection-yolov11)** — open the link, point your webcam, and see it in action.

## Features

- **Live Webcam Detection** — Real-time browser-based weapon detection via WebSocket
- **CCTV / IP Camera** — Connect RTSP, HTTP, or DroidCam streams for continuous surveillance
- **File Upload** — Batch process images and videos with drag-and-drop
- **4 Weapon Classes** — Handgun, Long Gun, Knife, Explosive
- **Threat Alerts** — Instant notifications with cropped evidence thumbnails
- **Confidence Control** — Adjust detection sensitivity in real-time with a slider
- **Enhanced Knife Detection** — Quadrant-crop scanning for detecting knives at distance
- **H.264 Video Export** — Browser-ready re-encoded detection results
- **WebSocket Streaming** — Low-latency frame-by-frame analysis
- **100% Local Processing** — No external API calls, everything runs on the server

## Weapon Classes

| Class | Color | Description |
|-------|-------|-------------|
| **Handgun** | Red | Pistols, revolvers, compact firearms |
| **Long Gun** | Dark Red | Rifles, shotguns, long-barreled firearms |
| **Knife** | Orange | Bladed weapons — tuned with lower threshold for sharper recall at distance |
| **Explosive** | Magenta | Grenades, IEDs, explosive devices |

## How It Works

**1. Capture** — Open your webcam, connect an IP camera, or drag-and-drop any image or video file

**2. Detect** — Every frame is processed through YOLOv11s with per-class confidence thresholds and bounding box filters

**3. Alert** — Threats are highlighted with color-coded bounding boxes and logged with cropped evidence

## Tech Stack

**Backend** — Python, FastAPI, Uvicorn, OpenCV, Ultralytics, FFmpeg

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
| `/api/detect/image` | POST | Detect weapons in an uploaded image |
| `/api/detect/video/stream` | POST | Process video with SSE progress streaming |
| `/api/detect/video/history` | GET | List all processed videos |
| `/api/detect/live` | WebSocket | Live webcam detection stream |
| `/api/detect/cctv` | WebSocket | CCTV/IP camera detection stream |
| `/api/alerts` | GET | List recent threat alerts |

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app with lifespan model loading
│   │   ├── config.py            # Settings (thresholds, paths)
│   │   ├── model.py             # WeaponDetector — YOLOv11s with per-class filtering
│   │   ├── schemas.py           # Pydantic response models
│   │   ├── alerts.py            # In-memory alert manager
│   │   └── routes/
│   │       ├── image.py         # Image upload + deep knife detection
│   │       ├── video.py         # Video processing with SSE + H.264 re-encode
│   │       ├── websocket.py     # Live webcam WebSocket
│   │       └── cctv.py          # CCTV/IP camera WebSocket
│   ├── models/
│   │   └── best.pt              # Fine-tuned YOLOv11s weights
│   └── training/                # Training scripts and notebooks
├── frontend/
│   ├── index.html               # Single-page app
│   ├── css/styles.css           # Cyber-teal dark theme
│   └── js/                      # Modular JS (webcam, cctv, upload, alerts)
├── demo/                        # Demo video and preview GIF
├── Dockerfile                   # Docker deployment
└── README.md
```

## Training

The model was trained on multiple Roboflow weapon datasets merged into a 4-class taxonomy. Training was done on Google Colab with T4 GPU using the Ultralytics framework.

```bash
# Local training (requires CUDA GPU)
cd backend
python -m training.train
```

## License

MIT

## Author

Built by [chowhandutta](https://github.com/chowhandutta2004-lgtm)
