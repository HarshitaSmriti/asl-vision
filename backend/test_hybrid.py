import os
import sys
import glob
import csv
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import load_asl_model
from backend.hybrid_inference import HybridASLInferenceEngine
from backend.preprocessing import (
    normalize_frame,
    resample_sequence,
    compute_velocity_features,
    POSE_LANDMARK_COUNT,
    FACE_SELECTED_INDICES,
    HAND_LANDMARK_COUNT
)

def load_parquet_sample(parquet_path: str) -> np.ndarray:
    """Loads and preprocesses a parquet file into (64, 696) array."""
    df = pd.read_parquet(parquet_path)
    frames = df['frame'].unique()
    frames.sort()
    
    frame_features = []
    for f in frames:
        f_df = df[df['frame'] == f]
        
        # Pose
        pose_df = f_df[f_df['type'] == 'pose']
        pose_arr = np.zeros((POSE_LANDMARK_COUNT, 3), dtype=np.float32)
        for _, row in pose_df.iterrows():
            idx = int(row['landmark_index'])
            if idx < POSE_LANDMARK_COUNT:
                x = row['x'] if not np.isnan(row['x']) else 0.0
                y = row['y'] if not np.isnan(row['y']) else 0.0
                z = row['z'] if not np.isnan(row['z']) else 0.0
                pose_arr[idx] = [x, y, z]
                
        # Face
        face_df = f_df[f_df['type'] == 'face']
        face_arr = np.zeros((len(FACE_SELECTED_INDICES), 3), dtype=np.float32)
        for out_idx, lm_idx in enumerate(FACE_SELECTED_INDICES):
            row = face_df[face_df['landmark_index'] == lm_idx]
            if len(row) > 0:
                r = row.iloc[0]
                x = r['x'] if not np.isnan(r['x']) else 0.0
                y = r['y'] if not np.isnan(r['y']) else 0.0
                z = r['z'] if not np.isnan(r['z']) else 0.0
                face_arr[out_idx] = [x, y, z]
                
        # Left Hand
        lh_df = f_df[f_df['type'] == 'left_hand']
        lh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
        lh_present = False
        if len(lh_df) > 0 and not lh_df['x'].isna().all():
            lh_present = True
            for _, row in lh_df.iterrows():
                idx = int(row['landmark_index'])
                if idx < HAND_LANDMARK_COUNT:
                    x = row['x'] if not np.isnan(row['x']) else 0.0
                    y = row['y'] if not np.isnan(row['y']) else 0.0
                    z = row['z'] if not np.isnan(row['z']) else 0.0
                    lh_arr[idx] = [x, y, z]
                    
        # Right Hand
        rh_df = f_df[f_df['type'] == 'right_hand']
        rh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
        rh_present = False
        if len(rh_df) > 0 and not rh_df['x'].isna().all():
            rh_present = True
            for _, row in rh_df.iterrows():
                idx = int(row['landmark_index'])
                if idx < HAND_LANDMARK_COUNT:
                    x = row['x'] if not np.isnan(row['x']) else 0.0
                    y = row['y'] if not np.isnan(row['y']) else 0.0
                    z = row['z'] if not np.isnan(row['z']) else 0.0
                    rh_arr[idx] = [x, y, z]
                    
        feat_348 = normalize_frame(pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present)
        frame_features.append(feat_348)
        
    seq = np.array(frame_features, dtype=np.float32)
    resampled = resample_sequence(seq, target_length=64)
    sample_696 = compute_velocity_features(resampled)
    return sample_696

def make_hand(wrist: np.ndarray, shape: str = "open_palm") -> np.ndarray:
    """Generates an anatomically structured 21-landmark hand array."""
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist  # 0: Wrist
    
    # Joint definitions: (mcp, pip, dip, tip)
    # Thumb: 1, 2, 3, 4
    # Index: 5, 6, 7, 8
    # Middle: 9, 10, 11, 12
    # Ring: 13, 14, 15, 16
    # Pinky: 17, 18, 19, 20
    
    if shape == "open_palm":
        # All fingers extended outward/upward
        hand[1] = wrist + [-0.02, -0.02, 0.0]
        hand[2] = wrist + [-0.03, -0.03, 0.0]
        hand[3] = wrist + [-0.04, -0.04, 0.0]
        hand[4] = wrist + [-0.05, -0.05, 0.0] # Thumb tip
        
        for f_idx, mcp_x in enumerate([-0.02, 0.0, 0.02, 0.04]):
            b = 5 + f_idx * 4
            hand[b] = wrist + [mcp_x, -0.04, 0.0]     # MCP
            hand[b+1] = wrist + [mcp_x, -0.07, 0.0]   # PIP
            hand[b+2] = wrist + [mcp_x, -0.09, 0.0]   # DIP
            hand[b+3] = wrist + [mcp_x, -0.11, 0.0]   # TIP
            
    elif shape == "fist":
        # All fingers curled down into palm
        hand[1] = wrist + [-0.02, -0.01, 0.0]
        hand[2] = wrist + [-0.02, -0.02, 0.0]
        hand[3] = wrist + [-0.01, -0.02, 0.0]
        hand[4] = wrist + [0.0, -0.02, 0.0]
        
        for f_idx, mcp_x in enumerate([-0.02, 0.0, 0.02, 0.04]):
            b = 5 + f_idx * 4
            hand[b] = wrist + [mcp_x, -0.03, 0.0]     # MCP
            hand[b+1] = wrist + [mcp_x, -0.04, 0.01]  # PIP
            hand[b+2] = wrist + [mcp_x, -0.03, 0.02]  # DIP (curled)
            hand[b+3] = wrist + [mcp_x, -0.02, 0.01]  # TIP (curled into palm)
            
    elif shape == "pinch":
        # Thumb (4) and Index (8) touching, others folded
        hand[1] = wrist + [-0.02, -0.02, 0.0]
        hand[2] = wrist + [-0.02, -0.04, 0.0]
        hand[3] = wrist + [-0.01, -0.05, 0.0]
        hand[4] = wrist + [0.0, -0.05, 0.0]       # Thumb tip
        
        # Index
        hand[5] = wrist + [-0.01, -0.03, 0.0]
        hand[6] = wrist + [-0.01, -0.05, 0.0]
        hand[7] = wrist + [-0.005, -0.055, 0.0]
        hand[8] = wrist + [0.0, -0.05, 0.0]       # Index tip (touches thumb tip)
        
        # Middle, Ring, Pinky folded
        for f_idx, mcp_x in enumerate([0.01, 0.02, 0.03]):
            b = 9 + f_idx * 4
            hand[b] = wrist + [mcp_x, -0.03, 0.0]
            hand[b+1] = wrist + [mcp_x, -0.04, 0.01]
            hand[b+2] = wrist + [mcp_x, -0.03, 0.02]
            hand[b+3] = wrist + [mcp_x, -0.02, 0.01]
            
    return hand

def generate_synthetic_everyday_sign(sign_name: str) -> np.ndarray:
    """Generates a geometrically valid synthetic sample for everyday sign rule verification."""
    frames = []
    T = 64
    for t in range(T):
        prog = t / (T - 1)
        pose_arr = np.zeros((25, 3), dtype=np.float32)
        pose_arr[11] = [0.4, 0.5, 0.0]  # Left shoulder
        pose_arr[12] = [0.6, 0.5, 0.0]  # Right shoulder
        pose_arr[0] = [0.5, 0.2, 0.0]   # Nose
        
        face_arr = np.zeros((7, 3), dtype=np.float32)
        face_arr[0] = [0.5, 0.28, 0.0]  # mouth center
        face_arr[3] = [0.5, 0.32, 0.0]  # chin
        
        lh_arr = np.zeros((21, 3), dtype=np.float32)
        lh_present = False
        rh_present = True
        
        if sign_name == "thank you":
            # Right hand open palm starts at chin (0.5, 0.32) and moves forward/down (0.5, 0.46)
            wrist = np.array([0.5, 0.32 + 0.14 * prog, 0.1 * prog], dtype=np.float32)
            rh_arr = make_hand(wrist, "open_palm")
                
        elif sign_name == "hello":
            # Right hand open palm near forehead (0.58, 0.18) waving side-to-side (oscillating in x)
            wave_x = 0.58 + 0.08 * np.sin(prog * 4 * np.pi)
            wrist = np.array([wave_x, 0.18, 0.0], dtype=np.float32)
            rh_arr = make_hand(wrist, "open_palm")
                
        elif sign_name == "yes":
            # Fist nodding up and down vertically
            nod_y = 0.45 + 0.06 * np.sin(prog * 6 * np.pi)
            wrist = np.array([0.55, nod_y, 0.0], dtype=np.float32)
            rh_arr = make_hand(wrist, "fist")
                
        elif sign_name == "no":
            # Index + middle finger tapping thumb (pinch oscillation)
            tap_y = 0.42 + 0.04 * np.sin(prog * 4 * np.pi)
            wrist = np.array([0.55, tap_y, 0.0], dtype=np.float32)
            rh_arr = make_hand(wrist, "pinch")
                
        else:
            wrist = np.array([0.55, 0.5, 0.0], dtype=np.float32)
            rh_arr = make_hand(wrist, "open_palm")
                
        feat_348 = normalize_frame(pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present)
        frames.append(feat_348)
        
    seq = np.array(frames, dtype=np.float32)
    sample_696 = compute_velocity_features(seq)
    return sample_696

def run_hybrid_evaluation_suite():
    print("=" * 70)
    print("RUNNING HYBRID ASL RECOGNITION SYSTEM TEST SUITE")
    print("=" * 70)

    engine_hybrid = HybridASLInferenceEngine(hybrid_mode=True)
    engine_pure = HybridASLInferenceEngine(hybrid_mode=False)

    results_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "hybrid_test_results.csv")
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    csv_rows = []

    # 1. Evaluate on Real Parquet Test Samples
    parquet_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_samples", "*.parquet"))[:30]
    print(f"\n1. Evaluating on {len(parquet_files)} Local Test Parquets...")

    match_count = 0
    for idx, p_file in enumerate(parquet_files):
        filename = os.path.basename(p_file)
        sample_696 = load_parquet_sample(p_file)

        # Pure neural prediction
        res_pure = engine_pure.predict_sample(sample_696, top_k=5)
        # Hybrid prediction
        res_hybrid = engine_hybrid.predict_sample(sample_696, top_k=5)

        neural_top1 = res_pure.get("raw_top_class", "N/A")
        hybrid_top1 = res_hybrid.get("raw_top_class", "N/A")
        source = res_hybrid.get("source", "N/A")
        neural_conf = res_hybrid.get("model_confidence", 0.0)
        rule_compat = res_hybrid.get("rule_compatibility", 0.0)
        final_rel = res_hybrid.get("final_reliability", 0.0)

        csv_rows.append({
            "test_type": "parquet_file",
            "sample_id": filename,
            "neural_prediction": neural_top1,
            "hybrid_prediction": hybrid_top1,
            "source": source,
            "neural_confidence": f"{neural_conf*100:.2f}%",
            "rule_compatibility": f"{rule_compat*100:.2f}%",
            "final_reliability": f"{final_rel*100:.2f}%",
            "status": res_hybrid.get("status", "unknown")
        })

        if idx < 5:
            print(f"  [{filename}] Neural: {neural_top1} ({neural_conf*100:.1f}%) | Hybrid: {hybrid_top1} [Source: {source}] (Rel: {final_rel*100:.1f}%)")

    # 2. Evaluate on Everyday Kinematic Signs
    everyday_signs_to_test = ["thank you", "hello", "yes", "no"]
    print(f"\n2. Evaluating on Everyday Kinematic Gestures: {everyday_signs_to_test}...")

    for sign in everyday_signs_to_test:
        sample_696 = generate_synthetic_everyday_sign(sign)
        res_hybrid = engine_hybrid.predict_sample(sample_696, top_k=5)
        
        hybrid_top1 = res_hybrid.get("raw_top_class", "N/A")
        source = res_hybrid.get("source", "N/A")
        neural_conf = res_hybrid.get("model_confidence", 0.0)
        rule_compat = res_hybrid.get("rule_compatibility", 0.0)
        final_rel = res_hybrid.get("final_reliability", 0.0)

        csv_rows.append({
            "test_type": "everyday_gesture_synthetic",
            "sample_id": sign,
            "neural_prediction": "N/A (not in 95)",
            "hybrid_prediction": hybrid_top1,
            "source": source,
            "neural_confidence": f"{neural_conf*100:.2f}%",
            "rule_compatibility": f"{rule_compat*100:.2f}%",
            "final_reliability": f"{final_rel*100:.2f}%",
            "status": res_hybrid.get("status", "unknown")
        })
        print(f"  [Target: '{sign}'] -> Hybrid Output: '{hybrid_top1}' | Source: '{source}' | Rule Match: {rule_compat*100:.1f}%")

    # 3. Write results to CSV
    fieldnames = [
        "test_type", "sample_id", "neural_prediction", "hybrid_prediction",
        "source", "neural_confidence", "rule_compatibility", "final_reliability", "status"
    ]
    with open(results_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved sanity evaluation log to: {results_csv_path}")
    print("=" * 70)
    print("HYBRID RECOGNITION TEST SUITE COMPLETED SUCCESSFULLY.")
    print("=" * 70)

if __name__ == "__main__":
    run_hybrid_evaluation_suite()
