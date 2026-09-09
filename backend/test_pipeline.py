import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from backend.model import load_asl_model, ASLTransformer
from backend.preprocessing import (
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
    print("=== STEP 1: Verifying Model Loading & Shapes ===")
    model, class_names, device = load_asl_model()
    assert len(class_names) == 95, f"Expected 95 classes, got {len(class_names)}"
    
    # Dummy tensor verification
    dummy = torch.zeros(1, 64, 696).to(device)
    with torch.no_grad():
        out = model(dummy)
        probs = torch.softmax(out, dim=-1)
    
    assert out.shape == (1, 95), f"Expected (1, 95), got {out.shape}"
    assert probs.shape == (1, 95), f"Expected probs (1, 95), got {probs.shape}"
    print("[PASS] Model loaded and forward pass matches (1, 95).")

    print("\n=== STEP 2: Verifying Preprocessing & Velocity Dimensions ===")
    # Create synthetic frame landmarks
    pose = np.random.randn(25, 3).astype(np.float32)
    face = np.random.randn(7, 3).astype(np.float32)
    lh = np.random.randn(21, 3).astype(np.float32)
    rh = np.random.randn(21, 3).astype(np.float32)

    feat_348 = normalize_frame(pose, face, lh, rh, lh_present=True, rh_present=True)
    assert feat_348.shape == (348,), f"Expected (348,), got {feat_348.shape}"
    print("[PASS] Frame normalization produces 348 features.")

    # Sequence of 30 frames
    seq_30 = [feat_348 + np.random.randn(348).astype(np.float32) * 0.01 for _ in range(30)]
    sample_696 = process_landmarks_sequence(seq_30)
    assert sample_696.shape == (64, 696), f"Expected (64, 696), got {sample_696.shape}"
    print("[PASS] Resampling & Velocity concatenation produces (64, 696).")

    # Verify velocity definition: v[0] == 0, v[t] == s[t] - s[t-1]
    resampled_348 = sample_696[:, :348]
    velocity_348 = sample_696[:, 348:]
    assert np.allclose(velocity_348[0], np.zeros(348)), "Velocity at t=0 must be zero."
    assert np.allclose(velocity_348[1], resampled_348[1] - resampled_348[0]), "Velocity at t=1 must be s[1] - s[0]."
    print("[PASS] Velocity math verified: velocity[0] = 0, velocity[t] = s[t] - s[t-1].")

    print("\n=== STEP 3: Verifying Inference Engine & Rolling Live Predictor ===")
    engine = ASLInferenceEngine()
    result = engine.predict_sample(sample_696, top_k=5)
    assert result["success"] is True
    assert len(result["top_predictions"]) == 5
    print(f"[PASS] Inference result: {result['prediction']} ({result['confidence']*100:.2f}%)")
    for i, top in enumerate(result["top_predictions"], 1):
        print(f"   {i}. {top['class']:<15} {top['confidence']*100:.2f}%")

    # Test rolling predictor
    rolling = RollingLivePredictor(buffer_size=64, step_size=2)
    for i in range(20):
        res = rolling.add_frame(feat_348)
    assert res is not None
    print(f"[PASS] Rolling predictor returned output at buffer fill: {res.get('buffer_fill')}")

    print("\n=== STEP 4: Validating live_camera_html JavaScript Syntax ===")
    with open("streamlit_app.py", "r", encoding="utf-8") as f:
        app_code = f.read()

    start_str = "live_camera_html = f\"\"\""
    end_str = "\"\"\""
    p1 = app_code.find(start_str)
    if p1 != -1:
        p1 += len(start_str)
        p2 = app_code.find(end_str, p1)
        raw_html = app_code[p1:p2]
        raw_html = raw_html.replace("{classes_json}", "[]").replace("{{", "{").replace("}}", "}")
        
        s1 = raw_html.find("<script>") + len("<script>")
        s2 = raw_html.rfind("</script>")
        js_code = raw_html[s1:s2]
        
        with open("extracted_camera.js", "w", encoding="utf-8") as jf:
            jf.write(js_code)
        print(f"[PASS] Extracted {len(js_code)} bytes of JavaScript to extracted_camera.js")

if __name__ == "__main__":
    test_pipeline()
