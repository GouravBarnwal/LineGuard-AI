from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "data" / "raw" / "videos"
OUTPUT_DIR = ROOT / "data" / "processed" / "video" / "frames_unreviewed"
SAMPLE_EVERY_SECONDS = 1

video_paths = sorted(VIDEO_DIR.glob("*.mp4"))

if not video_paths:
    raise FileNotFoundError(f"No MP4 videos found in {VIDEO_DIR}")

for video_path in video_paths:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        print(f"Could not read video details: {video_path.name}")
        cap.release()
        continue

    video_output = OUTPUT_DIR / video_path.stem
    video_output.mkdir(parents=True, exist_ok=True)

    duration_seconds = frame_count / fps
    saved = 0

    for second in range(0, int(duration_seconds) + 1, SAMPLE_EVERY_SECONDS):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()

        if not ok:
            continue

        output_path = video_output / f"frame_{second:04d}s.jpg"
        if cv2.imwrite(str(output_path), frame):
            saved += 1

    cap.release()
    print(f"{video_path.name}: saved {saved} frames")

print(f"Frames saved under: {OUTPUT_DIR}")