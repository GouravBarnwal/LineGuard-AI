import csv
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "splits.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "visa_candle" / "label_audit"


def main() -> None:
    with SPLITS_CSV.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    samples = sorted(
        (
            row for row in rows
            if row["split"] == "validation" and row["defect_label"] == "1"
        ),
        key=lambda row: row["image_path"],
    )[:4]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for row in samples:
        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        with Image.open(image_path) as source:
            image = source.convert("RGBA")

        with Image.open(mask_path) as source:
            mask = source.convert("L")

        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {image_path.name}")

        alpha = mask.point(lambda value: 110 if value > 0 else 0)
        red_overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
        red_overlay.putalpha(alpha)
        result = Image.alpha_composite(image, red_overlay)

        bbox = mask.getbbox()
        if bbox:
            draw = ImageDraw.Draw(result)
            draw.rectangle(bbox, outline="lime", width=max(2, image.width // 300))

        output_path = OUTPUT_DIR / f"{image_path.stem}_audit.png"
        result.convert("RGB").save(output_path)
        print(output_path)

    print("Open the saved audit images and check that red covers the defect and green surrounds it.")


if __name__ == "__main__":
    main()