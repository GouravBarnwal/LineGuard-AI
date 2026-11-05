## Current status

- React frontend supports image and video uploads.
- Image API runs the U-Net model with CPU inference.
- Video API runs YOLO11n detection with ByteTrack tracking.
- Model files are stored in the `models` folder.
- This is a prototype. Results can be inaccurate on unfamiliar videos.

## Run locally

Open three PowerShell terminals from the project folder.

### Terminal 1 — Image API

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\backend
python -m uvicorn app.main:app --reload --port 8000