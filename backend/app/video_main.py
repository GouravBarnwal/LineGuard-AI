import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import cv2
import imageio_ffmpeg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolo11n_candle_video_best.pt"
RUNS_DIR = ROOT / "data" / "processed" / "video" / "api_runs"

MAX_VIDEO_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Process every second frame to reduce YOLO inference work.
VIDEO_FRAME_STRIDE = 2

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
detector = YOLO(str(MODEL_PATH))

app = FastAPI(title="LineGuard Video API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "classes": detector.names,
    }


@app.post("/v1/inspect-video")
def inspect_video(file: UploadFile = File(...)):
    filename = file.filename or "upload.mp4"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload an MP4, AVI, MOV, MKV, or WebM video.",
        )

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    if size == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded video is empty.",
        )

    if size > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Video must be 100 MB or smaller.",
        )

    job_id = uuid4().hex
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"source{extension}"
    annotated_path = job_dir / "annotated.mp4"
    browser_video_path = job_dir / "annotated_h264.mp4"

    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not read this video. Try an MP4 file.",
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            fps = 25.0

        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not read the video dimensions.",
            )

        output_fps = max(1.0, fps / VIDEO_FRAME_STRIDE)

        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )

        if not writer.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Could not create the annotated video.",
            )

        frame_count = 0
        detection_count = 0
        frames_with_tracks = 0
        track_ids: set[int] = set()

        try:
            results = detector.track(
                source=str(input_path),
                tracker="bytetrack.yaml",
                conf=0.10,
                imgsz=640,
                device="cpu",
                stream=True,
                vid_stride=VIDEO_FRAME_STRIDE,
                save=False,
                verbose=False,
            )

            for result in results:
                frame_count += 1
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    detection_count += len(boxes)

                    if boxes.id is not None:
                        frames_with_tracks += 1
                        track_ids.update(
                            int(track_id) for track_id in boxes.id.cpu().tolist()
                        )

                writer.write(result.plot())
        finally:
            writer.release()

        if frame_count == 0:
            raise HTTPException(
                status_code=400,
                detail="No frames could be read from this video.",
            )

        # Convert mp4v to H.264 so browsers can play the result.
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(annotated_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(browser_video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if not browser_video_path.exists() or browser_video_path.stat().st_size == 0:
            raise RuntimeError("H.264 conversion did not produce a video file.")

        input_path.unlink(missing_ok=True)

        return {
            "filename": filename,
            "frames_processed": frame_count,
            "detections": detection_count,
            "frames_with_tracks": frames_with_tracks,
            "track_ids": sorted(track_ids),
            "video_url": f"/outputs/{job_id}/annotated_h264.mp4",
        }

    except HTTPException:
        raise
    except subprocess.CalledProcessError as error:
        ffmpeg_message = error.stderr[-2000:] if error.stderr else str(error)
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not convert the result to browser-compatible video: "
                f"{ffmpeg_message}"
            ),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Video inspection failed: {error}",
        ) from error
    finally:
        file.file.close()


app.mount(
    "/outputs",
    StaticFiles(directory=str(RUNS_DIR)),
    name="outputs",
)
