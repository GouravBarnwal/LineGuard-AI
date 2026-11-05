from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "unet_candle.onnx"
VIDEO_PATH = Path(
    r"C:\Users\barnw\Downloads\gemini_generated_video_5c550a22.mp4"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "video"
TEMP_VIDEO = OUTPUT_DIR / "unet_heatmap_temp.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "unet_heatmap_diagnostic.mp4"

INPUT_SIZE = (256, 256)


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"U-Net model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Edit VIDEO_PATH to match where you saved the video."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(TEMP_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width * 2, height),
)
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create video: {TEMP_VIDEO}")

frame_count = 0
max_probability = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, INPUT_SIZE)
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        logits = session.run([output_name], {input_name: tensor})[0]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probability_map = probabilities[0, 0]

        max_probability = max(
            max_probability,
            float(probability_map.max()),
        )

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

        writer.write(np.hstack([frame, overlay]))
        frame_count += 1

finally:
    cap.release()
    writer.release()

if frame_count == 0:
    raise RuntimeError("No video frames were processed.")

subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(TEMP_VIDEO),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(OUTPUT_VIDEO),
    ],
    check=True,
    capture_output=True,
    text=True,
)

TEMP_VIDEO.unlink(missing_ok=True)

print(f"Frames processed: {frame_count}")
print(f"Maximum U-Net probability: {max_probability:.4f}")
print(f"Saved original | heatmap video: {OUTPUT_VIDEO}")