from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "yolo11n_candle_video_best.pt"
VIDEO = ROOT / "data" / "raw" / "videos" / "candle_burning_closeup.mp4"
OUTPUT = ROOT / "data" / "processed" / "video"

if not MODEL.exists():
    raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL}")

if not VIDEO.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO}")

detector = YOLO(str(MODEL))

results = detector.predict(
    source=str(VIDEO),
    conf=0.30,
    imgsz=960,
    device="cpu",
    save=True,
    stream=True,
    project=str(OUTPUT),
    name="burning_closeup_yolo",
    exist_ok=True,
    verbose=False,
)

frame_count = 0
detection_count = 0

for result in results:
    frame_count += 1
    if result.boxes is not None:
        detection_count += len(result.boxes)

print(f"Frames processed: {frame_count}")
print(f"Total detections: {detection_count}")
print(f"Annotated video folder: {OUTPUT / 'burning_closeup_yolo'}")