from pathlib import Path
import subprocess

import imageio_ffmpeg

source = Path(
    r"data\processed\video\api_runs\e173463b5df54c78bf8ee4ae183bd5ed\annotated.mp4"
)
destination = source.with_name("annotated_h264.mp4")

if not source.exists():
    raise FileNotFoundError(f"Video not found: {source.resolve()}")

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run(
    [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ],
    check=True,
)

print(f"Converted video: {destination.resolve()}")
print(f"Output size: {destination.stat().st_size:,} bytes")
