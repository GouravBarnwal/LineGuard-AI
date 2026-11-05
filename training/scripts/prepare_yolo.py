import csv
import shutil
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "splits.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "yolo"


def mask_to_yolo_box(image_path: Path, mask_path: Path) -> str:
    with Image.open(image_path) as image:
        width, height = image.size

    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")
        if mask.size != (width, height):
            raise ValueError(f"Image/mask size mismatch: {image_path.name}")
        bbox = mask.getbbox()

    if bbox is None:
        return ""

    left, top, right, bottom = bbox
    center_x = ((left + right) / 2) / width
    center_y = ((top + bottom) / 2) / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height

    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def main() -> None:
    with SPLITS_CSV.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    for split in ("train", "validation", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = PROJECT_ROOT / row["image_path"]
        image_output = OUTPUT_ROOT / "images" / row["split"] / image_path.name
        label_output = OUTPUT_ROOT / "labels" / row["split"] / f"{image_path.stem}.txt"

        shutil.copy2(image_path, image_output)

        label = ""
        if row["mask_path"]:
            mask_path = PROJECT_ROOT / row["mask_path"]
            label = mask_to_yolo_box(image_path, mask_path)

        label_output.write_text(label, encoding="utf-8")

    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    yaml_path.write_text(
        f'path: "{OUTPUT_ROOT.as_posix()}"\n'
        "train: images/train\n"
        "val: images/validation\n"
        "test: images/test\n"
        "names:\n"
        "  0: defect\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(rows)} images under {OUTPUT_ROOT}")
    print(f"YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()