import { useEffect, useState, type FormEvent } from "react";
import "./App.css";

type HealthResponse = {
  status: string;
  model_loaded: boolean;
};

type InspectionResponse = {
  filename: string;
  width: number;
  height: number;
  inspection: string;
  reason: string;
  defect_area_percent: number;
  max_defect_probability: number;
  overlay_data_url: string;
};

type VideoInspectionResponse = {
  filename: string;
  frames_processed: number;
  detections: number;
  frames_with_tracks: number;
  track_ids: number[];
  video_url: string;
};

type Mode = "image" | "video";

const IMAGE_API =
  import.meta.env.VITE_IMAGE_API_URL || "http://127.0.0.1:8000";

const VIDEO_API =
  import.meta.env.VITE_VIDEO_API_URL || "http://127.0.0.1:8001";
  
function App() {
  const [mode, setMode] = useState<Mode>("image");

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [apiOffline, setApiOffline] = useState(false);

  const [videoApiConnected, setVideoApiConnected] = useState(false);
  const [videoApiOffline, setVideoApiOffline] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<InspectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoResult, setVideoResult] =
    useState<VideoInspectionResponse | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoUploading, setVideoUploading] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${IMAGE_API}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Image API health check failed");
        return response.json() as Promise<HealthResponse>;
      })
      .then(setHealth)
      .catch((caught: unknown) => {
        if (!(caught instanceof Error && caught.name === "AbortError")) {
          setApiOffline(true);
        }
      });

    fetch(`${VIDEO_API}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("Video API health check failed");
        return response.json() as Promise<HealthResponse>;
      })
      .then((data) => setVideoApiConnected(data.status === "ok"))
      .catch((caught: unknown) => {
        if (!(caught instanceof Error && caught.name === "AbortError")) {
          setVideoApiOffline(true);
        }
      });

    return () => controller.abort();
  }, []);

  const imageApiConnected = health?.status === "ok";

  async function handleImageUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;

    setUploading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${IMAGE_API}/v1/inspect`, {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as InspectionResponse & {
        detail?: string;
      };

      if (!response.ok) {
        throw new Error(data.detail ?? "Image inspection failed.");
      }

      setResult(data);
    } catch (caught: unknown) {
      setError(
        caught instanceof Error ? caught.message : "Image inspection failed.",
      );
    } finally {
      setUploading(false);
    }
  }

  async function handleVideoUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!videoFile) return;

    setVideoUploading(true);
    setVideoError(null);
    setVideoResult(null);

    const formData = new FormData();
    formData.append("file", videoFile);

    try {
      const response = await fetch(`${VIDEO_API}/v1/inspect-video`, {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as VideoInspectionResponse & {
        detail?: string;
      };

      if (!response.ok) {
        throw new Error(data.detail ?? "Video inspection failed.");
      }

      setVideoResult(data);
    } catch (caught: unknown) {
      setVideoError(
        caught instanceof Error ? caught.message : "Video inspection failed.",
      );
    } finally {
      setVideoUploading(false);
    }
  }

  const annotatedVideoUrl = videoResult
    ? `${VIDEO_API}${videoResult.video_url.replace(
        "/annotated.mp4",
        "/annotated_h264.mp4",
      )}`
    : "";

  return (
    <main className="page">
      <header className="header">
        <div className="brand-icon">LG</div>
        <div>
          <strong>
            LineGuard <span>AI</span>
          </strong>
          <small>Industrial vision inspection</small>
        </div>
        <span className="prototype">PROTOTYPE</span>
      </header>

      <section className="hero">
        <p className="eyebrow">QUALITY CONTROL / VISION SYSTEM</p>
        <h1>
          Catch defects
          <br />
          <span>before they ship.</span>
        </h1>
        <p className="intro">
          A computer vision workspace for inspecting products and locating
          visible defects.
        </p>
      </section>

      <section className="status-grid">
        <article className="card">
          <p className="eyebrow">IMAGE API</p>
          <h2 className={imageApiConnected ? "online" : "offline"}>
            {imageApiConnected ? "Connected" : apiOffline ? "Offline" : "Checking…"}
          </h2>
          <p>U-Net image inspection · port 8000</p>
        </article>

        <article className="card">
          <p className="eyebrow">VIDEO API</p>
          <h2 className={videoApiConnected ? "online" : "offline"}>
            {videoApiConnected
              ? "Connected"
              : videoApiOffline
                ? "Offline"
                : "Checking…"}
          </h2>
          <p>YOLO11 + ByteTrack · port 8001</p>
        </article>
      </section>

      <section className="inspection">
        <p className="eyebrow">INSPECTION WORKSPACE</p>

        <div className="mode-switch" role="tablist" aria-label="Inspection type">
          <button
            className={`mode-button ${mode === "image" ? "active" : ""}`}
            type="button"
            role="tab"
            aria-selected={mode === "image"}
            onClick={() => setMode("image")}
          >
            Upload image
          </button>
          <button
            className={`mode-button ${mode === "video" ? "active" : ""}`}
            type="button"
            role="tab"
            aria-selected={mode === "video"}
            onClick={() => setMode("video")}
          >
            Upload video
          </button>
        </div>

        {mode === "image" ? (
          <>
            <h2>Inspect a product image</h2>
            <p className="upload-hint">JPEG, PNG, or WebP · 10 MB maximum</p>

            <form className="upload-form" onSubmit={handleImageUpload}>
              <label className="file-choice">
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(event) => {
                    setFile(event.target.files?.[0] ?? null);
                    setResult(null);
                    setError(null);
                  }}
                />
                {file ? file.name : "Choose an image"}
              </label>

              <button
                className="inspect-button"
                type="submit"
                disabled={!file || !imageApiConnected || uploading}
              >
                {uploading ? "Inspecting…" : "Inspect image"}
              </button>
            </form>

            {error && (
              <p className="upload-error" role="alert">
                {error}
              </p>
            )}

            {result && (
              <div className="result-panel" role="status">
                <strong>
                  {result.inspection === "defect_detected"
                    ? "Possible defect detected"
                    : "No defect detected"}
                </strong>
                <p>
                  {result.filename} · {result.width} × {result.height} px
                </p>
                <p>
                  Predicted defect area: {result.defect_area_percent}% · Max
                  probability: {(result.max_defect_probability * 100).toFixed(1)}%
                </p>
                <img
                  src={result.overlay_data_url}
                  alt="Red overlay showing the predicted defect area"
                  className="inspection-overlay"
                />
              </div>
            )}
          </>
        ) : (
          <>
            <h2>Inspect a product video</h2>
            <p className="upload-hint">
              MP4, AVI, MOV, MKV, or WebM · 100 MB maximum
            </p>

            <form className="upload-form" onSubmit={handleVideoUpload}>
              <label className="file-choice">
                <input
                  type="file"
                  accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm"
                  onChange={(event) => {
                    setVideoFile(event.target.files?.[0] ?? null);
                    setVideoResult(null);
                    setVideoError(null);
                  }}
                />
                {videoFile ? videoFile.name : "Choose a video"}
              </label>

              <button
                className="inspect-button"
                type="submit"
                disabled={!videoFile || !videoApiConnected || videoUploading}
              >
                {videoUploading ? "Inspecting video…" : "Inspect video"}
              </button>
            </form>

            {videoError && (
              <p className="upload-error" role="alert">
                {videoError}
              </p>
            )}

            {videoResult && (
              <div className="result-panel" role="status">
                <strong>Video inspection complete</strong>
                <p>
                  {videoResult.filename} · {videoResult.frames_processed} frames
                  processed
                </p>
                <p>
                  {videoResult.detections} detections ·{" "}
                  {videoResult.frames_with_tracks} frames with tracks · Track
                  IDs: {videoResult.track_ids.join(", ") || "none"}
                </p>
                <video
                  className="inspection-video"
                  controls
                  preload="metadata"
                  src={annotatedVideoUrl}
                >
                  Your browser does not support video playback.
                </video>
              </div>
            )}
          </>
        )}
      </section>
    </main>
  );
}

export default App;