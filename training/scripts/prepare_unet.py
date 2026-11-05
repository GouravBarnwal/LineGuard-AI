import csv
from pathlib import Path
import shutil

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "splits.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "unet"


def main() -> None:
    with SPLITS_CSV.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    for split in ("train", "validation", "test"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "masks" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = PROJECT_ROOT / row["image_path"]
        image_output = OUTPUT_ROOT / "images" / row["split"] / image_path.name
        mask_output = OUTPUT_ROOT / "masks" / row["split"] / f"{image_path.stem}.png"

        shutil.copy2(image_path, image_output)

        with Image.open(image_path) as image:
            image_size = image.size

        if row["mask_path"]:
            with Image.open(PROJECT_ROOT / row["mask_path"]) as source_mask:
                mask = source_mask.convert("L")
        else:
            mask = Image.new("L", image_size, 0)

        if mask.size != image_size:
            raise ValueError(f"Image/mask size mismatch: {image_path.name}")

        binary_mask = mask.point(lambda value: 255 if value > 0 else 0)
        binary_mask.save(mask_output)

    print(f"Prepared {len(rows)} images and masks under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()