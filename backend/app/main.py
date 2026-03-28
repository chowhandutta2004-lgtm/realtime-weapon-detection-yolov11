import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .alerts import clear_alerts, get_alerts
from .model import detector
from .routes import cctv, image, video, websocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading weapon detection model...")
    detector.load()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Weapon Detection",
    description="Real-time weapon detection using YOLOv11",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(image.router)
app.include_router(video.router)
app.include_router(websocket.router)
app.include_router(cctv.router)


@app.get("/api/alerts")
async def list_alerts(limit: int = 50):
    return get_alerts(limit)


@app.delete("/api/alerts")
async def delete_alerts():
    clear_alerts()
    return {"status": "cleared"}


# Serve frontend — check both local dev and Docker paths
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if not frontend_dir.exists():
    frontend_dir = Path("/app/frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
