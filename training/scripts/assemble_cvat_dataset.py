from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = Path.home() / "Downloads" / "cvat-manual-label-dataset.zip"
SOURCE_IMAGES = ROOT / "data" / "processed" / "video" / "frames_for_cvat"
OUTPUT = ROOT / "data" / "processed" / "video" / "candle_video_yolo_day5"

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"CVAT ZIP not found: {ZIP_PATH}")
if not SOURCE_IMAGES.exists():
    raise FileNotFoundError(f"Frame folder not found: {SOURCE_IMAGES}")

counts = {
    "train": {"frames": 0, "corrosion": 0, "broken": 0},
    "val": {"frames": 0, "corrosion": 0, "broken": 0},
}

with zipfile.ZipFile(ZIP_PATH) as archive:
    image_names = [
        line.strip()
        for line in archive.read("train.txt").decode("utf-8").splitlines()
        if line.strip()
    ]

    for listed_path in image_names:
        filename = Path(listed_path).name
        stem = Path(filename).stem
        match = re.search(r"_frame_(\d+)s$", stem)
        if not match:
            raise ValueError(f"Unexpected frame name: {filename}")

        # Hold out the last sampled frames from each video for validation.
        split = "val" if int(match.group(1)) >= 8 else "train"

        source_image = SOURCE_IMAGES / filename
        if not source_image.exists():
            raise FileNotFoundError(f"Frame missing locally: {source_image}")

        image_out = OUTPUT / "images" / split / filename
        label_out = OUTPUT / "labels" / split / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_out)

        label_name = f"labels/train/{stem}.txt"
        if label_name in archive.namelist():
            label_text = archive.read(label_name).decode("utf-8").strip()
        else:
            # No label file in CVAT means this frame has no defect boxes.
            label_text = ""

        box_count = 0
        for line in label_text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Bad YOLO row in {label_name}: {line}")

            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
            if class_id not in (0, 1) or any(v < 0 or v > 1 for v in values):
                raise ValueError(f"Invalid YOLO row in {label_name}: {line}")

            counts[split]["corrosion" if class_id == 0 else "broken"] += 1
            box_count += 1

        label_out.write_text(
            label_text + ("\n" if label_text else ""),
            encoding="utf-8",
        )
        counts[split]["frames"] += 1

yaml_text = f"""path: "{OUTPUT.as_posix()}"
train: images/train
val: images/val
names:
  0: Corrosion
  1: Broken
"""
(OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

print(f"Dataset created at: {OUTPUT}")
print(f"Training: {counts['train']}")
print(f"Validation: {counts['val']}")
print(f"Config: {OUTPUT / 'data.yaml'}")