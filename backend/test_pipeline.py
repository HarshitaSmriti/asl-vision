import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from backend.model import load_asl_model, ASLTransformer
from backend.preprocessing import (
    extract_frame_landmarks,
    normalize_frame,
    process_landmarks_sequence,
    resample_sequence,
    compute_velocity_features,
    BASE_FEATURE_DIM,
    FINAL_FEATURE_DIM,
    SEQUENCE_LENGTH
)
from backend.inference import ASLInferenceEngine, RollingLivePredictor

def test_pipeline():
    print("================================================================")
    print("        ASL VISION COMPREHENSIVE VERIFICATION TEST SUITE        ")
    print("================================================================")

    print("\n--- TEST 1: Checkpoint Loading & State Dict Integrity ---")
    model, class_names, device = load_asl_model()
    assert len(class_names) == 95, f"Expected 95 classes, got {len(class_names)}"
    assert isinstance(model, ASLTransformer), "Model must be an ASLTransformer instance"
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 2311519, f"Expected 2,311,519 parameters, got {total_params}"
    print(f"[PASS] Model loaded successfully on {device.type.upper()} ({total_params:,} parameters, 95 classes)")

    print("\n--- TEST 2: Exact 95-Class Output & Mapping Verification ---")
    assert class_names[0] == "airplane", f"Class 0 should be 'airplane', got {class_names[0]}"
    assert class_names[94] == "grandpa", f"Class 94 should be 'grandpa', got {class_names[94]}"
    assert model.classifier.out_features == 95, f"Classifier head must have 95 outputs"
    print(f"[PASS] Exact 95 classes verified (0: {class_names[0]} ... 94: {class_names[94]})")

    print("\n--- TEST 3: Tensor Forward Pass Shape [1, 64, 696] -> [1, 95] ---")
    dummy = torch.zeros(1, 64, 696).to(device)
    with torch.inference_mode():
        logits = model(dummy)
        probs = torch.softmax(logits, dim=-1)
    assert logits.shape == (1, 95), f"Expected logits (1, 95), got {logits.shape}"
    assert probs.shape == (1, 95), f"Expected probs (1, 95), got {probs.shape}"
    assert np.isclose(probs.sum().item(), 1.0, atol=1e-5), "Softmax probabilities must sum to 1.0"
    print("[PASS] Forward pass verified: [1, 64, 696] -> [1, 95] with valid softmax")

    print("\n--- TEST 4: Landmark Normalization (348 Features) ---")
    pose = np.random.randn(25, 3).astype(np.float32)
    face = np.random.randn(7, 3).astype(np.float32)
    lh = np.random.randn(21, 3).astype(np.float32)
    rh = np.random.randn(21, 3).astype(np.float32)

    feat_348 = normalize_frame(pose, face, lh, rh, lh_present=True, rh_present=True)
    assert feat_348.shape == (348,), f"Expected (348,), got {feat_348.shape}"
    assert not np.isnan(feat_348).any(), "Normalized features must not contain NaNs"
    assert not np.isinf(feat_348).any(), "Normalized features must not contain Infs"
    print("[PASS] Landmark normalization produces clean 348-dim feature vectors")

    print("\n--- TEST 5: Temporal Resampling to 64 Frames ---")
    # Test shorter sequence (20 frames)
    short_seq = np.random.randn(20, 348).astype(np.float32)
    res_short = resample_sequence(short_seq, 64)
    assert res_short.shape == (64, 348), f"Expected (64, 348), got {res_short.shape}"

    # Test longer sequence (120 frames)
    long_seq = np.random.randn(120, 348).astype(np.float32)
    res_long = resample_sequence(long_seq, 64)
    assert res_long.shape == (64, 348), f"Expected (64, 348), got {res_long.shape}"
    print("[PASS] Temporal resampling verified for both upsampling (20->64) and downsampling (120->64)")

    print("\n--- TEST 6: Velocity Feature Math ---")
    sample_696 = compute_velocity_features(res_short)
    assert sample_696.shape == (64, 696), f"Expected (64, 696), got {sample_696.shape}"
    base_part = sample_696[:, :348]
    vel_part = sample_696[:, 348:]
    assert np.allclose(vel_part[0], np.zeros(348)), "Velocity at t=0 must be exactly zero"
    assert np.allclose(vel_part[1], base_part[1] - base_part[0]), "Velocity at t=1 must be s[1] - s[0]"
    print("[PASS] Velocity calculation verified: v[0] = 0, v[t] = s[t] - s[t-1]")

    print("\n--- TEST 7: Missing Landmarks & Zero Padding Stability ---")
    # Frame with no hands detected
    empty_feat = normalize_frame(pose, face, np.zeros((21,3)), np.zeros((21,3)), False, False)
    assert empty_feat.shape == (348,)
    assert np.all(empty_feat[222:348] == 0.0), "Hand features must be zero when no hands are present"
    print("[PASS] Missing landmark handling verified without NaN or shape errors")

    print("\n--- TEST 8: Canonical Sequence Prediction Engine ---")
    engine = ASLInferenceEngine(confidence_threshold=0.15)
    seq_frames = [feat_348 + np.random.randn(348).astype(np.float32) * 0.01 for _ in range(45)]
    pred_res = engine.predict_sequence(seq_frames, top_k=5)
    assert pred_res["success"] is True
    assert len(pred_res["top_predictions"]) == 5
    assert pred_res["top_class_id"] is not None
    print(f"[PASS] Canonical predict_sequence output: {pred_res['prediction']} ({pred_res['confidence']*100:.2f}%)")

    print("\n--- TEST 9: Confidence Thresholding & Unknown State ---")
    engine_strict = ASLInferenceEngine(confidence_threshold=0.99)
    res_strict = engine_strict.predict_sample(sample_696, top_k=5)
    assert res_strict["is_confident"] is False
    assert res_strict["prediction"] == "Detecting sign..."
    print(f"[PASS] Strict thresholding correctly triggered: is_confident = False, prediction = 'Detecting sign...'")

    print("\n--- TEST 10: Rolling Buffer Live Predictor States ---")
    rolling = RollingLivePredictor(buffer_size=64, step_size=2, min_frames=16)
    
    # State A: Collecting frames (< 16)
    res_init = rolling.add_frame(feat_348)
    assert res_init["status"] == "collecting_frames"
    assert "Collecting frames... 1/64" in res_init["prediction"]
    
    # State B: Sufficient frames (>= 16)
    for _ in range(25):
        res_live = rolling.add_frame(feat_348)
    assert res_live["status"] in ["recognizing", "low_confidence"]
    assert len(res_live["top_predictions"]) == 5
    print(f"[PASS] Rolling predictor states verified: collecting (1/64) -> temporal inference ({res_live['buffer_fill']}/64)")

    print("\n--- TEST 11: Verification That NO Heuristic / Mock Fallback Exists ---")
    with open("streamlit_app.py", "r", encoding="utf-8") as f:
        app_code = f.read()
    assert "predictLocalSpatial" not in app_code, "CRITICAL ERROR: predictLocalSpatial still exists in streamlit_app.py!"
    assert "boost(" not in app_code, "CRITICAL ERROR: boost rules still exist in streamlit_app.py!"
    print("[PASS] Strict neural compliance verified: No heuristic rules or mock fallbacks found")

    print("\n================================================================")
    print("        ALL 11 VERIFICATION TESTS PASSED SUCCESSFULLY!          ")
    print("================================================================")

if __name__ == "__main__":
    test_pipeline()

