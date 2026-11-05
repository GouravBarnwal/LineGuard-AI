from pathlib import Path

import openvino as ov


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = PROJECT_ROOT / "models" / "unet_candle.onnx"
XML_PATH = PROJECT_ROOT / "models" / "unet_candle_openvino.xml"

if not ONNX_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}")

openvino_model = ov.convert_model(str(ONNX_PATH))
ov.save_model(openvino_model, str(XML_PATH), compress_to_fp16=False)

BIN_PATH = XML_PATH.with_suffix(".bin")
print("OpenVINO XML:", XML_PATH, XML_PATH.exists())
print("OpenVINO BIN:", BIN_PATH, BIN_PATH.exists())