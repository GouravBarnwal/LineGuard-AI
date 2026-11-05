from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "yolo11n_candle_video_best.pt"
VIDEO = ROOT / "data" / "raw" / "videos" / "candle_broken_rim.mp4"
OUTPUT = ROOT / "data" / "processed" / "video"

if not MODEL.exists():
    raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL}")

if not VIDEO.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO}")

detector = YOLO(str(MODEL))

results = detector.track(
    source=str(VIDEO),
    tracker="bytetrack.yaml",
    conf=0.30,
    imgsz=960,
    device="cpu",
    save=True,
    stream=True,
    project=str(OUTPUT),
    name="broken_rim_bytetrack_yolo11n",
    exist_ok=True,
    verbose=False,
)

frame_count = 0
detection_count = 0
frames_with_tracks = 0
track_ids = set()

for result in results:
    frame_count += 1
    boxes = result.boxes

    if boxes is not None and len(boxes):
        detection_count += len(boxes)
        if boxes.id is not None:
            frames_with_tracks += 1
            track_ids.update(int(value) for value in boxes.id.cpu().tolist())

print(f"Frames processed: {frame_count}")
print(f"Detections: {detection_count}")
print(f"Frames with track IDs: {frames_with_tracks}")
print(f"Track IDs seen: {sorted(track_ids)}")
print(f"Annotated video folder: {OUTPUT / 'broken_rim_bytetrack_yolo11n'}")