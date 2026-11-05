from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "unet_candle.onnx"
VIDEO_PATH = Path(
    r"C:\Users\barnw\Downloads\gemini_generated_video_5c550a22.mp4"
)
OUTPUT_PATH = ROOT / "data" / "processed" / "video" / "unet_heatmap_preview.jpg"

INPUT_SIZE = (256, 256)
FRAME_TIME_MS = 5000  # Inspect the 5-second frame


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"U-Net model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Edit VIDEO_PATH at the top to match where you saved the video."
    )

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(str(VIDEO_PATH))
cap.set(cv2.CAP_PROP_POS_MSEC, FRAME_TIME_MS)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read the selected video frame.")

height, width = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, INPUT_SIZE)
tensor = resized.astype(np.float32) / 255.0
tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

logits = session.run([output_name], {input_name: tensor})[0]
probabilities = 1.0 / (1.0 + np.exp(-logits))
probability_map = probabilities[0, 0]

probability_map = cv2.resize(
    probability_map,
    (width, height),
    interpolation=cv2.INTER_LINEAR,
)
heatmap = cv2.applyColorMap(
    np.uint8(np.clip(probability_map, 0, 1) * 255),
    cv2.COLORMAP_JET,
)
overlay = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
comparison = np.hstack([frame, overlay])
cv2.imwrite(str(OUTPUT_PATH), comparison)

print(f"Frame size: {width} x {height}")
print(f"Max anomaly probability: {probability_map.max():.4f}")
print(f"Saved original | heatmap preview to: {OUTPUT_PATH}")