import os
import sys
import numpy as np
import pandas as pd
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.hybrid_inference import HybridRollingLivePredictor, HybridASLInferenceEngine
from backend.verify_hybrid_system import load_parquet_sample, generate_controlled_gesture

def test_poc_signs():
    print("=" * 80)
    print("LIVE POC RECOGNITION VALIDATION TEST")
    print("=" * 80)

    predictor = HybridRollingLivePredictor(buffer_size=64, step_size=2, min_frames=24, hybrid_mode=True)
    
    test_cases = [
        {"name": "duck", "type": "95_class", "file": "data/test_samples/1002091184.parquet"},
        {"name": "brother", "type": "95_class", "file": "data/test_samples/1001379621.parquet"},
        {"name": "go", "type": "95_class", "file": "data/test_samples/1000061708.parquet"},
        {"name": "apple", "type": "95_class", "synthetic": "apple"},
        {"name": "airplane", "type": "95_class", "synthetic": "airplane"},
        {"name": "hello", "type": "everyday", "synthetic": "hello"},
        {"name": "thank you", "type": "everyday", "synthetic": "thank you"},
        {"name": "stop", "type": "everyday", "synthetic": "stop"},
        {"name": "where", "type": "everyday", "synthetic": "where"}
    ]

    results_table = []

    for tc in test_cases:
        expected = tc["name"]
        predictor.reset()

        if "file" in tc and os.path.exists(tc["file"]):
            sample_696 = load_parquet_sample(tc["file"])
        else:
            sample_696 = generate_controlled_gesture(expected)

        # Unpack 64 frames (348 base) and stream them frame-by-frame
        frames_348 = sample_696[:, :348]

        last_res = None
        for f_idx in range(64):
            last_res = predictor.add_frame(frames_348[f_idx])

        pred_sign = last_res.get("prediction", "Detecting sign...")
        conf = last_res.get("confidence", 0.0) * 100
        source = last_res.get("source", "N/A")
        is_conf = last_res.get("is_confident", False)

        # Check match
        is_match = (pred_sign.lower() == expected.lower())
        status_str = "PASS" if is_match else "MISMATCH"

        results_table.append({
            "Expected": expected,
            "Predicted": pred_sign,
            "Source": source,
            "Confidence": f"{conf:.1f}%",
            "Result": status_str
        })

        print(f"[{status_str:<8s}] Expected: '{expected:<12s}' -> Predicted: '{pred_sign:<12s}' | Source: [{source}] (Conf: {conf:.1f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY OF POC SIGN DEMO")
    print("=" * 80)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    test_poc_signs()
