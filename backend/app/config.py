from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = str(Path(__file__).resolve().parent.parent / "models" / "best.pt")
    FALLBACK_MODEL: str = "yolo11s.pt"
    CONFIDENCE_THRESHOLD: float = 0.65
    KNIFE_CONFIDENCE_THRESHOLD: float = 0.30  # low bar for knives — maximize recall even at distance
    MIN_BOX_AREA: int = 900            # ignore detections smaller than 30x30 px
    KNIFE_MIN_BOX_AREA: int = 100      # knives can be very thin/small at distance — 10x10 px minimum
    MAX_BOX_FRAME_RATIO: float = 0.25  # ignore detections covering >25% of frame
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "uploads")
    RESULTS_DIR: str = str(Path(__file__).resolve().parent.parent / "results")
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    WEBSOCKET_MAX_FPS: int = 15

    class Config:
        env_file = ".env"


settings = Settings()
