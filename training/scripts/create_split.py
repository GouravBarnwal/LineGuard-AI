import csv
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "VisA" / "candle" / "Data"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "splits.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def assign_split(
    records: list[dict[str, str]],
    seed: int,
) -> list[dict[str, str]]:
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * 0.70)
    validation_end = train_end + int(len(shuffled) * 0.15)

    for index, record in enumerate(shuffled):
        if index < train_end:
            record["split"] = "train"
        elif index < validation_end:
            record["split"] = "validation"
        else:
            record["split"] = "test"

    return shuffled


def main() -> None:
    normal_dir = DATA_ROOT / "Images" / "Normal"
    anomaly_dir = DATA_ROOT / "Images" / "Anomaly"
    masks_dir = DATA_ROOT / "Masks" / "Anomaly"

    normal_images = image_files(normal_dir)
    anomaly_images = image_files(anomaly_dir)
    masks_by_stem = {path.stem: path for path in image_files(masks_dir)}

    normal_records = [
        {
            "image_path": image.relative_to(PROJECT_ROOT).as_posix(),
            "mask_path": "",
            "defect_label": "0",
        }
        for image in normal_images
    ]

    anomaly_records = []
    for image in anomaly_images:
        mask = masks_by_stem.get(image.stem)
        if mask is None:
            raise SystemExit(f"No mask found for {image.name}")

        anomaly_records.append({
            "image_path": image.relative_to(PROJECT_ROOT).as_posix(),
            "mask_path": mask.relative_to(PROJECT_ROOT).as_posix(),
            "defect_label": "1",
        })

    rows = assign_split(normal_records, seed=42)
    rows.extend(assign_split(anomaly_records, seed=43))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "mask_path", "defect_label", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["defect_label"]) for row in rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")
    for split in ("train", "validation", "test"):
        print(
            f"{split}: normal={counts[(split, '0')]}, "
            f"anomaly={counts[(split, '1')]}"
        )


if __name__ == "__main__":
    main()