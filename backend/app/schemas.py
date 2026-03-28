from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]


class ImageDetectionResponse(BaseModel):
    image: str  # base64-encoded annotated image
    detections: list[Detection]
    weapon_detected: bool


class VideoDetectionResponse(BaseModel):
    video_url: str
    total_frames: int
    frames_with_detections: int
    detections_summary: dict[str, dict]  # label -> {peak_confidence, first_seen, last_seen, frame_count}


class AlertItem(BaseModel):
    id: str
    timestamp: str
    label: str
    confidence: float
    thumbnail: str  # base64
