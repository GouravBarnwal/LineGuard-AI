from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "VisA" / "candle" / "Data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    normal_dir = DATA_ROOT / "Images" / "Normal"
    anomaly_dir = DATA_ROOT / "Images" / "Anomaly"
    masks_dir = DATA_ROOT / "Masks" / "Anomaly"

    required_dirs = [normal_dir, anomaly_dir, masks_dir]
    missing_dirs = [path for path in required_dirs if not path.is_dir()]

    if missing_dirs:
        for path in missing_dirs:
            print(f"Missing folder: {path}")
        sys.exit(1)

    normal_images = image_files(normal_dir)
    anomaly_images = image_files(anomaly_dir)
    masks = image_files(masks_dir)
    mask_stems = {path.stem for path in masks}

    missing_masks = [
        image.name for image in anomaly_images if image.stem not in mask_stems
    ]

    print(f"Category: candle")
    print(f"Normal images: {len(normal_images)}")
    print(f"Anomaly images: {len(anomaly_images)}")
    print(f"Anomaly masks: {len(masks)}")

    if not normal_images or not anomaly_images or not masks:
        print("Validation failed: one or more folders contain no images.")
        sys.exit(1)

    if missing_masks:
        print(f"Anomaly images missing masks: {len(missing_masks)}")
        for filename in missing_masks[:10]:
            print(f"  {filename}")
        sys.exit(1)

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()