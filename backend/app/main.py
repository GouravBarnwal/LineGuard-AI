import os
from io import BytesIO

from .inference import inspect_image as run_model
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="LineGuard AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "LineGuard AI API"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/v1/inspect")
async def inspect_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    content = await file.read(MAX_IMAGE_BYTES + 1)

    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()

        with Image.open(BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    prediction = run_model(content)

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        **prediction,
    }