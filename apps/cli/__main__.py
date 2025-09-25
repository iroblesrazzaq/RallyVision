#!/usr/bin/env python3
import argparse
import os
import sys

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
from tennis_tracker.segmentation.segment import segment_video

import numpy as np
import joblib
import scipy.ndimage


def run(args: argparse.Namespace) -> int:
    os.makedirs(args.output_dir, exist_ok=True)
    # Step 1: Pose extraction (skip if exists and not overwrite)
    pose_extractor = PoseExtractor(model_path=args.yolo_model)
    raw_npz = args.raw_npz or None
    if raw_npz is None:
        # create default raw path
        raw_npz = pose_extractor.extract_pose_data(
            video_path=args.video,
            confidence_threshold=float(args.conf),
            start_time_seconds=int(args.start_time),
            duration_seconds=int(args.duration),
            target_fps=int(args.fps),
            annotations_csv=None,
        )
    # Step 2: Preprocess
    preprocessed_npz = os.path.join(args.output_dir, "preprocessed.npz")
    pre = DataPreprocessor(save_court_masks=False)
    pre.preprocess_single_video(raw_npz, args.video, preprocessed_npz, overwrite=bool(args.overwrite))
    # Step 3: Features
    features_npz = os.path.join(args.output_dir, "features.npz")
    fe = FeatureEngineer()
    fe.create_features_from_preprocessed(preprocessed_npz, features_npz, overwrite=bool(args.overwrite))
    # Step 4: Inference
    data = np.load(features_npz)
    features = data["features"]
    scaler = joblib.load(args.scaler_path)
    features = scaler.transform(features)
    model, device = load_model_from_checkpoint(args.model_path, return_logits=False)
    avg_probs = run_windowed_inference_average(model, device, features, sequence_length=int(args.seq_len), overlap=int(args.overlap))
    smoothed_probs = scipy.ndimage.gaussian_filter1d(avg_probs.astype(np.float32), sigma=float(args.sigma))
    min_duration_frames = int(round(max(0.0, float(args.min_dur_sec)) * float(args.fps)))
    binary_pred = hysteresis_threshold(smoothed_probs, low=float(args.low), high=float(args.high), min_duration=min_duration_frames)
    segments = extract_segments_from_binary(binary_pred)
    csv_out = os.path.join(args.output_dir, "segments.csv")
    write_segments_csv(segments, csv_out, fps=float(args.fps), overwrite=bool(args.overwrite))
    # Step 5: Segment video (optional)
    if args.segment_video:
        video_out = os.path.join(args.output_dir, "segmented.mp4")
        from tennis_tracker.segmentation.segment import load_intervals
        intervals = load_intervals(csv_out)
        if intervals:
            segment_video(args.video, intervals, video_out)
    print("Done.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Tennis Tracker end-to-end CLI")
    p.add_argument("--video", required=True, help="Path to input MP4 video")
    p.add_argument("--output-dir", required=True, help="Directory to store outputs")
    p.add_argument("--model-path", required=True, help="Path to LSTM .pth file")
    p.add_argument("--scaler-path", required=True, help="Path to StandardScaler .joblib")
    p.add_argument("--yolo-model", default="yolov8s-pose.pt", help="YOLO pose weights file name in models/")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--seq-len", type=int, default=300)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--low", type=float, default=0.45)
    p.add_argument("--high", type=float, default=0.8)
    p.add_argument("--min-dur-sec", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--start-time", type=int, default=0)
    p.add_argument("--duration", type=int, default=999999)
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    p.add_argument("--raw-npz", default=None, help="Optional: point to existing raw pose npz to skip extraction")
    p.add_argument("--segment-video", action="store_true", help="Also write segmented MP4")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())


