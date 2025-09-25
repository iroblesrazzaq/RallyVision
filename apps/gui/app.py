#!/usr/bin/env python3
import os
import uuid
import threading
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from tennis_tracker.extraction.pose_extractor import PoseExtractor
from tennis_tracker.preprocessing.data_preprocessor import DataPreprocessor
from tennis_tracker.features.feature_engineer import FeatureEngineer
from tennis_tracker.infer import (
    load_model_from_checkpoint,
    run_windowed_inference_average,
    hysteresis_threshold,
    extract_segments_from_binary,
    write_segments_csv,
)
from tennis_tracker.segmentation.segment import segment_video, load_intervals

import numpy as np
import joblib
import av
import scipy.ndimage


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
JOBS_DIR = ROOT_DIR / "jobs"
MODELS_DIR = ROOT_DIR / "models"
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path="/",
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

JobDict = Dict[str, Any]
jobs_lock = threading.Lock()
jobs: Dict[str, JobDict] = {}


def _new_job_state(job_id: str) -> JobDict:
    return {
        "id": job_id,
        "status": "in_progress",
        "error": None,
        "cancelled": False,
        "steps": {
            "pose": {"status": "waiting", "progress": 0},
            "preprocess": {"status": "waiting", "progress": 0},
            "feature": {"status": "waiting", "progress": 0},
            "inference": {"status": "waiting", "progress": 0},
            "output": {"status": "waiting", "progress": 0},
        },
        "paths": {"upload": None, "raw_npz": None, "preprocessed_npz": None, "features_npz": None, "csv": None, "video": None, "job_dir": str(JOBS_DIR / job_id)},
        "thread": None,
    }


def _set_step(job: JobDict, step: str, status: str, progress: int) -> None:
    job["steps"][step]["status"] = status
    job["steps"][step]["progress"] = int(max(0, min(100, progress)))


def _check_cancel(job: JobDict) -> None:
    if job.get("cancelled"):
        job["status"] = "cancelled"
        raise RuntimeError("Job cancelled")


def _run_pipeline(job_id: str, yolo_model_filename: str = "yolov8s-pose.pt", conf_thresh: float = 0.25) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return
    try:
        job_dir = Path(job["paths"]["job_dir"]) ; job_dir.mkdir(parents=True, exist_ok=True)
        upload_path = Path(job["paths"]["upload"])  # set during upload

        _check_cancel(job) ; _set_step(job, "pose", "in_progress", 1)
        extractor = PoseExtractor(model_path=yolo_model_filename)
        # determine duration
        video_duration = 0.0
        try:
            with av.open(str(upload_path)) as container:
                stream = container.streams.video[0]
                if getattr(stream, "duration", None) and getattr(stream, "time_base", None):
                    video_duration = float(stream.duration * stream.time_base)
        except Exception:
            video_duration = 0.0
        if not video_duration or video_duration <= 0:
            video_duration = 99999.0
        def pose_progress(frac: float) -> None:
            _set_step(job, "pose", "in_progress", int(1 + max(0.0, min(1.0, frac)) * 98))
        raw_npz_path = extractor.extract_pose_data(str(upload_path), float(conf_thresh), 0, int(video_duration), 15, None, progress_callback=pose_progress)
        job["paths"]["raw_npz"] = raw_npz_path
        _set_step(job, "pose", "completed", 100)

        _check_cancel(job) ; _set_step(job, "preprocess", "in_progress", 1)
        pre = DataPreprocessor(save_court_masks=False)
        preprocessed_npz = str(job_dir / "preprocessed.npz")
        _set_step(job, "preprocess", "in_progress", 10)
        if not pre.preprocess_single_video(raw_npz_path, str(upload_path), preprocessed_npz, overwrite=True):
            raise RuntimeError("Preprocessing failed")
        job["paths"]["preprocessed_npz"] = preprocessed_npz
        _set_step(job, "preprocess", "completed", 100)

        _check_cancel(job) ; _set_step(job, "feature", "in_progress", 1)
        fe = FeatureEngineer()
        features_npz = str(job_dir / "features.npz")
        _set_step(job, "feature", "in_progress", 10)
        if not fe.create_features_from_preprocessed(preprocessed_npz, features_npz, overwrite=True):
            raise RuntimeError("Feature engineering failed")
        job["paths"]["features_npz"] = features_npz
        _set_step(job, "feature", "completed", 100)

        _check_cancel(job) ; _set_step(job, "inference", "in_progress", 1)
        scaler_path = str(ROOT_DIR / "data" / "seq_len_300" / "scaler.joblib")
        model_ckpt = str(ROOT_DIR / "checkpoints" / "seq_len300" / "best_model.pth")
        data = np.load(features_npz) ; feature_vectors = data["features"]
        scaler = joblib.load(scaler_path) ; normalized_features = scaler.transform(feature_vectors)
        model, device = load_model_from_checkpoint(model_ckpt, return_logits=False)
        def infer_progress(frac: float) -> None:
            _set_step(job, "inference", "in_progress", int(1 + max(0.0, min(1.0, frac)) * 94))
        avg_probs = run_windowed_inference_average(model, device, normalized_features, sequence_length=300, overlap=150, progress_callback=infer_progress)
        _set_step(job, "inference", "in_progress", 60)
        smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs, sigma=1.5)
        binary_pred = hysteresis_threshold(smoothed_probs, low=0.45, high=0.80, min_duration=int(0.5 * 15))
        csv_path = str(job_dir / "segments.csv")
        segments = extract_segments_from_binary(binary_pred)
        write_segments_csv(segments, csv_path, fps=15.0, overwrite=True)
        job["paths"]["csv"] = csv_path
        _set_step(job, "inference", "completed", 100)

        _check_cancel(job) ; _set_step(job, "output", "in_progress", 1)
        video_out_path = str(job_dir / "segmented.mp4")
        try:
            intervals = load_intervals(csv_path)
            if intervals:
                _set_step(job, "output", "in_progress", 40)
                segment_video(str(upload_path), intervals, video_out_path)
        except Exception:
            video_out_path = None
        job["paths"]["video"] = video_out_path
        _set_step(job, "output", "completed", 100)
        job["status"] = "completed"
    except Exception as e:
        if job.get("status") != "cancelled":
            job["status"] = "failed"
            job["error"] = str(e)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/upload-and-start", methods=["POST"])
def upload_and_start():
    if "video" not in request.files:
        return jsonify({"error": "Missing file field 'video'"}), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    filename = secure_filename(file.filename)
    if not (filename.lower().endswith(".mp4")):
        return jsonify({"error": "Only MP4 files are supported at this time"}), 400
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id ; job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / "input.mp4" ; file.save(str(upload_path))
    state = _new_job_state(job_id)
    state["paths"]["upload"] = str(upload_path)
    worker = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    state["thread"] = worker
    with jobs_lock:
        jobs[job_id] = state
    worker.start()
    return jsonify({"job_id": job_id}), 200


@app.route("/api/progress/<job_id>", methods=["GET"])
def get_progress(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify({"status": job["status"], "steps": job["steps"], "error": job.get("error")}), 200


@app.route("/api/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    job["cancelled"] = True
    return jsonify({"status": "cancelled"}), 200


@app.route("/api/download/video/<job_id>", methods=["GET"])
def download_video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    video_path = job["paths"].get("video")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video not available"}), 404
    return send_file(video_path, as_attachment=True, download_name=f"{job_id}_segmented.mp4")


@app.route("/api/download/csv/<job_id>", methods=["GET"])
def download_csv(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    csv_path = job["paths"].get("csv")
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "CSV not available"}), 404
    return send_file(csv_path, as_attachment=True, download_name=f"{job_id}_segments.csv")


def main() -> int:
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


