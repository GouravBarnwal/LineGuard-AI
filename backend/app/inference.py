import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "unet_candle_openvino.xml"
)
THRESHOLD = 0.40
INPUT_SIZE = (256, 256)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"OpenVINO model not found: {MODEL_PATH}")

core = ov.Core()
openvino_model = core.read_model(str(MODEL_PATH))
compiled_model = core.compile_model(openvino_model, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


def inspect_image(image_bytes: bytes) -> dict:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    resized = image.resize(INPUT_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))[None, ...]

    inference_result = compiled_model({input_layer: tensor})
    logits = inference_result[output_layer]

    probabilities = 1.0 / (1.0 + np.exp(-logits))
    mask = probabilities[0, 0] > THRESHOLD
    defect_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    mask_full = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    ).resize(image.size, resample=Image.Resampling.NEAREST)

    red_layer = Image.new("RGBA", image.size, (255, 60, 45, 0))
    red_layer.putalpha(
        mask_full.point(lambda value: 115 if value > 0 else 0)
    )

    overlay = Image.alpha_composite(
        image.convert("RGBA"),
        red_layer,
    ).convert("RGB")

    buffer = BytesIO()
    overlay.save(buffer, format="JPEG", quality=85)

    overlay_data_url = (
        "data:image/jpeg;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii")
    )

    return {
        "width": width,
        "height": height,
        "inspection": (
            "defect_detected" if defect_pixels else "no_defect_detected"
        ),
        "reason": "unet_openvino_cpu",
        "defect_area_percent": round(
            100 * defect_pixels / total_pixels,
            3,
        ),
        "max_defect_probability": round(
            float(probabilities.max()),
            4,
        ),
        "overlay_data_url": overlay_data_url,
    }