from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolo11n_candle_best.pt"
IMAGE_DIR = ROOT / "data" / "processed" / "visa_candle" / "yolo" / "images" / "train"
LABEL_DIR = ROOT / "data" / "processed" / "visa_candle" / "yolo" / "labels" / "train"
OUTPUT_PATH = ROOT / "data" / "processed" / "video" / "debug_training_sample.jpg"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")

# Select the first training image whose label file contains a defect box.
sample_image = None
sample_label = None

for label_path in sorted(LABEL_DIR.glob("*.txt")):
    if not label_path.read_text(encoding="utf-8").strip():
        continue

    matches = list(IMAGE_DIR.glob(label_path.stem + ".*"))
    if matches:
        sample_image = matches[0]
        sample_label = label_path
        break

if sample_image is None:
    raise FileNotFoundError("Could not find a training image with a nonempty label.")

model = YOLO(str(MODEL_PATH))
result = model.predict(
    source=str(sample_image),
    conf=0.001,
    imgsz=960,
    device="cpu",
    verbose=False,
)[0]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.save(filename=str(OUTPUT_PATH))

print(f"Image: {sample_image}")
print(f"Label: {sample_label}")
print(f"YOLO classes: {model.names}")
print(f"Ground-truth boxes: {len(sample_label.read_text(encoding='utf-8').splitlines())}")
print(f"Predicted boxes: {0 if result.boxes is None else len(result.boxes)}")
if result.boxes is not None and len(result.boxes):
    print("Confidence scores:", [round(float(x), 4) for x in result.boxes.conf])
print(f"Annotated sample: {OUTPUT_PATH}")