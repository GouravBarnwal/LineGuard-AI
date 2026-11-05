from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolo11n_candle_video_best.pt"
VIDEO_PATH = ROOT / "data" / "raw" / "videos" / "candle_burning_closeup.mp4"
OUTPUT_DIR = ROOT / "data" / "processed" / "video" / "debug_frames_burning_closeup"
TIMESTAMPS = [0.5, 2.5, 5.0, 7.5, 9.5]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

for seconds in TIMESTAMPS:
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()

    if not ok:
        print(f"{seconds:.1f}s: could not read frame")
        continue

    result = model.predict(
        source=frame,
        conf=0.25,  # Diagnostic threshold; may include false detections.
        imgsz=960,
        device="cpu",
        verbose=False,
    )[0]

    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)

    if count:
        confidences = boxes.conf.cpu().tolist()
        print(
            f"{seconds:.1f}s: {count} detections; "
            f"confidence scores={ [round(value, 4) for value in confidences] }"
        )
    else:
        print(f"{seconds:.1f}s: 0 detections")

    output_path = OUTPUT_DIR / f"frame_{seconds:.1f}s.jpg"
    cv2.imwrite(str(output_path), result.plot())

cap.release()
print(f"Debug frames saved under: {OUTPUT_DIR}")
