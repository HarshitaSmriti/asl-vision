import collections
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from backend.model import load_asl_model, ASLTransformer
from backend.preprocessing import process_landmarks_sequence, SEQUENCE_LENGTH, BASE_FEATURE_DIM, FINAL_FEATURE_DIM

class ASLInferenceEngine:
    """
    Authoritative inference engine for ASLTransformer (95 classes).
    Handles tensor conversion, forward pass, softmax probabilities, and top-K extraction.
    """
    def __init__(self, confidence_threshold: float = 0.20, smoothing_window: int = 5):
        self.model, self.class_names, self.device = load_asl_model()
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.num_classes = len(self.class_names)

    def predict_sample(self, sample_696: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Authoritative single-sample prediction method.
        Input: (64, 696) float32 numpy array.
        Output: Dictionary with predictions, top-5 classes, confidence, and status.
        """
        if sample_696.shape != (SEQUENCE_LENGTH, FINAL_FEATURE_DIM):
            raise ValueError(f"Expected input shape ({SEQUENCE_LENGTH}, {FINAL_FEATURE_DIM}), got {sample_696.shape}")

        # Check hand landmark presence in local feature channels
        # Left Hand (222:285) & Right Hand (285:348) local coordinates
        hand_features = sample_696[:, 222:348]
        has_hands = np.max(np.abs(hand_features)) > 1e-4

        if not has_hands:
            return {
                "success": True,
                "prediction": "Position hands in view",
                "is_confident": False,
                "raw_top_class": None,
                "top_class_id": None,
                "confidence": 0.0,
                "top_predictions": [],
                "status": "no_hands",
                "hand_detected": False
            }

        tensor_in = torch.from_numpy(sample_696).unsqueeze(0).to(self.device)  # (1, 64, 696)
        
        with torch.inference_mode():
            logits = self.model(tensor_in)  # (1, 95)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (95,)

        top_indices = np.argsort(probs)[::-1][:top_k]
        
        top_predictions = [
            {
                "class_id": int(idx),
                "class": self.class_names.get(int(idx), f"class_{idx}"),
                "confidence": float(probs[idx])
            }
            for idx in top_indices
        ]

        top_class_id = int(top_indices[0])
        top_class = top_predictions[0]["class"]
        top_conf = float(probs[top_class_id])

        confident = top_conf >= self.confidence_threshold

        return {
            "success": True,
            "prediction": top_class if confident else "Detecting sign...",
            "is_confident": confident,
            "raw_top_class": top_class,
            "top_class_id": top_class_id,
            "confidence": top_conf,
            "top_predictions": top_predictions,
            "all_probabilities": probs,
            "status": "recognizing" if confident else "low_confidence",
            "hand_detected": True
        }

    def predict_sequence(self, frames_348: List[np.ndarray], top_k: int = 5) -> Dict[str, Any]:
        """
        Canonical sequence prediction method.
        Accepts any number of 348-dim landmark frames, resamples to 64, computes velocity, and predicts.
        """
        sample_696 = process_landmarks_sequence(frames_348)
        return self.predict_sample(sample_696, top_k=top_k)

class RollingLivePredictor:
    """
    Maintains a rolling buffer of 348-dim landmark frames for real-time live video streams.
    Provides raw temporal windowing (EMA smoothing disabled for pure raw neural testing).
    """
    def __init__(self, buffer_size: int = 64, step_size: int = 2, min_frames: int = 16, smoothing_alpha: float = 0.0):
        self.engine = ASLInferenceEngine()
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.min_frames = min_frames
        self.smoothing_alpha = smoothing_alpha
        self.frame_buffer = collections.deque(maxlen=buffer_size)
        self.frame_count = 0
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.smoothed_probs: Optional[np.ndarray] = None

    def add_frame(self, frame_features_348: np.ndarray) -> Dict[str, Any]:
        """
        Appends a new frame (348 features) to the buffer.
        Returns live prediction status.
        """
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
                "is_confident": False,
                "top_predictions": [],
                "hand_detected": True
            }

        # Throttle inference to every step_size frames for responsiveness
        if self.frame_count % self.step_size != 0 and self.last_prediction is not None:
            return self.last_prediction

        # Prepare temporal sequence
        frames_list = list(self.frame_buffer)
        sample_696 = process_landmarks_sequence(frames_list)
        
        result = self.engine.predict_sample(sample_696, top_k=5)
        
        if result.get("hand_detected", False) and "all_probabilities" in result:
            raw_probs = result["all_probabilities"]
            if self.smoothing_alpha <= 0.0 or self.smoothed_probs is None:
                self.smoothed_probs = raw_probs.copy()
            else:
                self.smoothed_probs = (1.0 - self.smoothing_alpha) * self.smoothed_probs + self.smoothing_alpha * raw_probs

            # Re-derive top-5 from smoothed probabilities
            top_indices = np.argsort(self.smoothed_probs)[::-1][:5]
            smoothed_top = [
                {
                    "class_id": int(idx),
                    "class": self.engine.class_names.get(int(idx), f"class_{idx}"),
                    "confidence": float(self.smoothed_probs[idx])
                }
                for idx in top_indices
            ]
            top_class = smoothed_top[0]["class"]
            top_conf = smoothed_top[0]["confidence"]
            confident = top_conf >= self.engine.confidence_threshold

            result["top_predictions"] = smoothed_top
            result["confidence"] = top_conf
            result["raw_top_class"] = top_class
            result["prediction"] = top_class if confident else "Detecting sign..."
            result["is_confident"] = confident
            result["status"] = "recognizing" if confident else "low_confidence"

        result["buffer_fill"] = curr_fill
        result["buffer_target"] = self.buffer_size

        self.last_prediction = result
        return result

    def reset(self):
        self.frame_buffer.clear()
        self.frame_count = 0
        self.last_prediction = None
        self.smoothed_probs = None
