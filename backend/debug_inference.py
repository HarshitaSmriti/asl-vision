import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd
import torch

from backend.inference import ASLInferenceEngine
from backend.landmark_detector import HolisticLandmarkDetector
from backend.preprocessing import (
    extract_frame_landmarks,
    normalize_frame,
    process_landmarks_sequence,
    SEQUENCE_LENGTH,
    FINAL_FEATURE_DIM,
    POSE_LANDMARK_COUNT,
    FACE_SELECTED_INDICES,
    FACE_LANDMARK_COUNT,
    HAND_LANDMARK_COUNT
)

def debug_inference_on_video(video_path: str, confidence_threshold: float = 0.0):
    """
    Runs debug inference on a local video file using the exact production pipeline:
    Video -> MediaPipe Holistic -> 74 Landmarks -> Body/Hand Normalization (348) ->
    64-frame Resampling -> Velocity Features (696) -> ASLTransformer -> Top-5 Probabilities.
    """
    print("=" * 70)
    print(f"DEBUG INFERENCE ON VIDEO: {video_path}")
    print("=" * 70)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    detector = HolisticLandmarkDetector()
    engine = ASLInferenceEngine(confidence_threshold=confidence_threshold)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {total_frames} total frames, {fps:.1f} FPS")

    raw_frames_348 = []
    frame_idx = 0
    hands_detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        feat_348, lm_dict = detector.process_frame(frame)
        raw_frames_348.append(feat_348)
        
        if lm_dict.get("left_hand_present") or lm_dict.get("right_hand_present"):
            hands_detected_count += 1

    cap.release()
    print(f"Processed {len(raw_frames_348)} frames | Frames with hands detected: {hands_detected_count}/{len(raw_frames_348)}")

    if len(raw_frames_348) == 0:
        print("Error: No frames extracted from video.")
        return

    sample_696 = process_landmarks_sequence(raw_frames_348)
    print(f"\nExtracted Feature Tensor Shape : {sample_696.shape} (Expected: 64, 696)")
    print(f"Base Feature Dims (0..347)     : min={sample_696[:, :348].min():.4f}, max={sample_696[:, :348].max():.4f}")
    print(f"Velocity Dims (348..695)       : min={sample_696[:, 348:].min():.4f}, max={sample_696[:, 348:].max():.4f}")

    result = engine.predict_sample(sample_696, top_k=5)
    
    print("\n" + "=" * 70)
    print("NEURAL MODEL PREDICTION (100% Raw Softmax Output)")
    print("=" * 70)
    print(f"Frame Window: 1 (Full Sequence)")
    print(f"Valid Frames: {len(raw_frames_348)} -> Resampled to 64")
    print(f"Input Shape : [1, {SEQUENCE_LENGTH}, {FINAL_FEATURE_DIM}]")
    print("\nTop 5 Predictions:")
    for rank, p in enumerate(result["top_predictions"], 1):
        print(f"  {rank}. {p['class']:<20s} : {p['confidence']:.4f} ({p['confidence']*100:.2f}%)")

    print("\nFinal Output:")
    print(f"  Recognized Sign : {result['prediction']}")
    print(f"  Confidence      : {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print("=" * 70)
    return result

def debug_inference_on_parquet(parquet_path: str, confidence_threshold: float = 0.0):
    """
    Runs debug inference on a raw MediaPipe parquet file (Google ASL format).
    """
    print("=" * 70)
    print(f"DEBUG INFERENCE ON PARQUET: {parquet_path}")
    print("=" * 70)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    engine = ASLInferenceEngine(confidence_threshold=confidence_threshold)
    df_pq = pd.read_parquet(parquet_path)
    frames = sorted(df_pq['frame'].unique())
    print(f"Total Landmark Frames in Parquet: {len(frames)}")

    frame_features_list = []
    for f_idx in frames:
        frame_df = df_pq[df_pq['frame'] == f_idx]
        
        # Pose
        pose_df = frame_df[frame_df['type'] == 'pose'].sort_values('landmark_index')
        pose_arr = np.zeros((POSE_LANDMARK_COUNT, 3), dtype=np.float32)
        for _, row in pose_df.iterrows():
            l_idx = int(row['landmark_index'])
            if l_idx < POSE_LANDMARK_COUNT:
                x, y, z = row['x'], row['y'], row['z']
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    pose_arr[l_idx] = [x, y, z]
                    
        # Face
        face_df = frame_df[frame_df['type'] == 'face']
        face_arr = np.zeros((FACE_LANDMARK_COUNT, 3), dtype=np.float32)
        for idx, lm_idx in enumerate(FACE_SELECTED_INDICES):
            row = face_df[face_df['landmark_index'] == lm_idx]
            if len(row) > 0:
                x, y, z = row['x'].values[0], row['y'].values[0], row['z'].values[0]
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    face_arr[idx] = [x, y, z]
                    
        # Left hand
        lh_df = frame_df[frame_df['type'] == 'left_hand'].sort_values('landmark_index')
        lh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
        lh_present = False
        for _, row in lh_df.iterrows():
            l_idx = int(row['landmark_index'])
            if l_idx < HAND_LANDMARK_COUNT:
                x, y, z = row['x'], row['y'], row['z']
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    lh_arr[l_idx] = [x, y, z]
                    lh_present = True
                    
        # Right hand
        rh_df = frame_df[frame_df['type'] == 'right_hand'].sort_values('landmark_index')
        rh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
        rh_present = False
        for _, row in rh_df.iterrows():
            l_idx = int(row['landmark_index'])
            if l_idx < HAND_LANDMARK_COUNT:
                x, y, z = row['x'], row['y'], row['z']
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    rh_arr[l_idx] = [x, y, z]
                    rh_present = True
                    
        f_feat = normalize_frame(pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present)
        frame_features_list.append(f_feat)

    sample_696 = process_landmarks_sequence(frame_features_list)
    result = engine.predict_sample(sample_696, top_k=5)

    print("\n" + "=" * 70)
    print("NEURAL MODEL PREDICTION (100% Raw Softmax Output)")
    print("=" * 70)
    print(f"Valid Frames: {len(frames)} -> Resampled to 64")
    print(f"Input Shape : [1, {SEQUENCE_LENGTH}, {FINAL_FEATURE_DIM}]")
    print("\nTop 5 Predictions:")
    for rank, p in enumerate(result["top_predictions"], 1):
        print(f"  {rank}. {p['class']:<20s} : {p['confidence']:.4f} ({p['confidence']*100:.2f}%)")

    print("\nFinal Output:")
    print(f"  Recognized Sign : {result['prediction']}")
    print(f"  Confidence      : {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print("=" * 70)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug ASL Neural Inference on Video or Parquet")
    parser.add_argument("--file", type=str, default="data/test_samples/1000061708.parquet", help="Path to video or parquet file")
    parser.add_argument("--threshold", type=float, default=0.0, help="Confidence threshold")
    args = parser.parse_args()

    if args.file.endswith(".parquet"):
        debug_inference_on_parquet(args.file, args.threshold)
    else:
        debug_inference_on_video(args.file, args.threshold)
