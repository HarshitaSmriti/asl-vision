import collections
import numpy as np
from typing import Dict, List, Any, Optional
from backend.inference import ASLInferenceEngine
from backend.gesture_rules import GestureRuleEngine, GeometricFeatureExtractor
from backend.preprocessing import process_landmarks_sequence, SEQUENCE_LENGTH, FINAL_FEATURE_DIM

class HybridASLInferenceEngine:
    """
    Hybrid ASL Recognition Engine:
    Combines the 95-class ASLTransformer neural model with a geometric gesture interpretation layer.
    Features:
      - Toggleable HYBRID_MODE (zero regression when disabled).
      - Reranks / verifies top-K candidates from the 95-class model using kinematic rules.
      - Discovers everyday signs (e.g. hello, thank you, please, sorry, yes, no, stop, help) not in 95-class model.
      - Emits explicit sources ('95_class_model', 'everyday_gesture_layer', 'uncertain').
      - Outputs separate neural confidence and rule compatibility metrics.
    """

    def __init__(self, hybrid_mode: bool = True, confidence_threshold: float = 0.20):
        self.neural_engine = ASLInferenceEngine(confidence_threshold=confidence_threshold)
        self.rule_engine = GestureRuleEngine()
        self.hybrid_mode = hybrid_mode
        self.confidence_threshold = confidence_threshold
        self.class_names = self.neural_engine.class_names

    def predict_sample(self, sample_696: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Main hybrid prediction pipeline for a single (64, 696) tensor.
        """
        if sample_696.shape != (SEQUENCE_LENGTH, FINAL_FEATURE_DIM):
            raise ValueError(f"Expected shape ({SEQUENCE_LENGTH}, {FINAL_FEATURE_DIM}), got {sample_696.shape}")

        # 1. Run pure neural model forward pass
        neural_res = self.neural_engine.predict_sample(sample_696, top_k=top_k)
        
        if not neural_res.get("hand_detected", False) or neural_res.get("status") == "no_hands":
            neural_res["hybrid_mode"] = self.hybrid_mode
            neural_res["source"] = "none"
            neural_res["rule_compatibility"] = 0.0
            neural_res["final_reliability"] = 0.0
            neural_res["debug_telemetry"] = {}
            return neural_res

        # If hybrid mode is disabled, return pure neural results directly
        if not self.hybrid_mode:
            neural_res["hybrid_mode"] = False
            neural_res["source"] = "95_class_model"
            neural_res["rule_compatibility"] = 1.0
            neural_res["final_reliability"] = neural_res.get("confidence", 0.0)
            neural_res["debug_telemetry"] = {}
            return neural_res

        # 2. Extract geometric and kinematic telemetry
        kinematics = GeometricFeatureExtractor.extract_kinematics(sample_696)

        # 3. Check for Everyday Gestures (signs not in the 95-class model)
        everyday_matches = self.rule_engine.evaluate_everyday_signs(kinematics)
        top_everyday = everyday_matches[0] if len(everyday_matches) > 0 else None

        # 4. Evaluate rule compatibility for 95-class Top-K predictions
        raw_top_preds = neural_res.get("top_predictions", [])
        reranked_preds = []

        for item in raw_top_preds:
            cls_name = item["class"]
            p_conf = item["confidence"]
            c_score = self.rule_engine.score_95class_compatibility(cls_name, kinematics)
            # Blended score: 65% neural probability + 35% geometric compatibility
            blended_score = float(0.65 * p_conf + 0.35 * (c_score * p_conf * 1.5))
            reranked_preds.append({
                "class_id": item["class_id"],
                "class": cls_name,
                "confidence": p_conf,
                "rule_compatibility": c_score,
                "blended_score": blended_score
            })

        # Sort neural candidates by blended score
        reranked_preds.sort(key=lambda x: x["blended_score"], reverse=True)

        top_neural = reranked_preds[0] if len(reranked_preds) > 0 else None
        top_neural_conf = top_neural["confidence"] if top_neural else 0.0
        top_neural_compat = top_neural["rule_compatibility"] if top_neural else 0.0

        # Decision Strategy:
        # A. If everyday sign has very strong kinematic evidence (>=0.82) and neural 95-class model is not confident (<0.40)
        #    OR everyday sign matches >=0.90 with near-zero 95-class compatibility:
        if top_everyday and (
            (top_everyday["score"] >= 0.82 and top_neural_conf < 0.40) or
            (top_everyday["score"] >= 0.90 and top_neural_compat < 0.30)
        ):
            final_sign = top_everyday["sign"]
            source = "everyday_gesture_layer"
            rule_compat = top_everyday["score"]
            model_conf = top_neural_conf
            final_reliability = top_everyday["score"]
            status = "everyday_gesture"
            is_confident = True
        # B. If neural top-1 has sufficient confidence and reasonable compatibility
        elif top_neural and top_neural_conf >= self.confidence_threshold:
            final_sign = top_neural["class"]
            source = "95_class_model"
            rule_compat = top_neural_compat
            model_conf = top_neural_conf
            final_reliability = float(0.70 * top_neural_conf + 0.30 * top_neural_compat)
            status = "recognizing"
            is_confident = True
        # C. Ambiguous / Low confidence
        else:
            final_sign = "Uncertain"
            source = "uncertain"
            rule_compat = top_neural_compat if top_neural else 0.0
            model_conf = top_neural_conf
            final_reliability = model_conf
            status = "low_confidence"
            is_confident = False

        # Format debug telemetry for UI / API
        debug_telemetry = {
            "dominant_hand": kinematics.get("dom_hand", "none"),
            "both_hands_active": kinematics.get("both_hands", False),
            "hand_shape": kinematics.get("hand_shape", "unknown"),
            "displacement_magnitude": round(kinematics.get("disp_mag", 0.0), 3),
            "near_chin": kinematics.get("near_chin", False),
            "near_forehead": kinematics.get("near_forehead", False),
            "y_oscillations": kinematics.get("zero_crossings_y", 0),
            "x_oscillations": kinematics.get("zero_crossings_x", 0)
        }

        return {
            "success": True,
            "prediction": final_sign if is_confident else "Detecting sign...",
            "raw_top_class": final_sign,
            "source": source,
            "confidence": model_conf,
            "model_confidence": model_conf,
            "rule_compatibility": rule_compat,
            "final_reliability": final_reliability,
            "is_confident": is_confident,
            "hybrid_mode": self.hybrid_mode,
            "top_predictions": reranked_preds,
            "everyday_matches": everyday_matches[:3],
            "debug_telemetry": debug_telemetry,
            "status": status,
            "hand_detected": True
        }

    def predict_sequence(self, frames_348: List[np.ndarray], top_k: int = 5) -> Dict[str, Any]:
        """Predicts from list of 348-dim feature frames."""
        sample_696 = process_landmarks_sequence(frames_348)
        return self.predict_sample(sample_696, top_k=top_k)


class HybridRollingLivePredictor:
    """
    Maintains a rolling buffer of 348-dim landmark frames for real-time live video streams.
    Integrates HybridASLInferenceEngine with toggleable hybrid mode.
    """
    def __init__(self, buffer_size: int = 64, step_size: int = 2, min_frames: int = 16, hybrid_mode: bool = True):
        self.engine = HybridASLInferenceEngine(hybrid_mode=hybrid_mode)
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.min_frames = min_frames
        self.frame_buffer = collections.deque(maxlen=buffer_size)
        self.frame_count = 0
        self.last_prediction: Optional[Dict[str, Any]] = None

    def add_frame(self, frame_features_348: np.ndarray, hybrid_mode: Optional[bool] = None) -> Dict[str, Any]:
        """
        Appends a new frame (348 features) to the buffer and predicts.
        """
        if hybrid_mode is not None:
            self.engine.hybrid_mode = hybrid_mode

        self.frame_buffer.append(frame_features_348)
        self.frame_count += 1

        curr_fill = len(self.frame_buffer)
        if curr_fill < self.min_frames:
            return {
                "success": True,
                "status": "collecting_frames",
                "buffer_fill": curr_fill,
                "buffer_target": self.buffer_size,
                "prediction": f"Collecting frames... {curr_fill}/{self.buffer_size}",
                "confidence": 0.0,
                "model_confidence": 0.0,
                "rule_compatibility": 0.0,
                "final_reliability": 0.0,
                "source": "collecting",
                "is_confident": False,
                "top_predictions": [],
                "hand_detected": True
            }

        # Throttle inference to every step_size frames for responsiveness
        if self.frame_count % self.step_size != 0 and self.last_prediction is not None:
            return self.last_prediction

        frames_list = list(self.frame_buffer)
        sample_696 = process_landmarks_sequence(frames_list)
        
        result = self.engine.predict_sample(sample_696, top_k=5)
        result["buffer_fill"] = curr_fill
        result["buffer_target"] = self.buffer_size

        self.last_prediction = result
        return result

    def reset(self):
        self.frame_buffer.clear()
        self.frame_count = 0
        self.last_prediction = None

