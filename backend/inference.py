import collections
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from backend.model import load_asl_model, ASLTransformer
from backend.preprocessing import process_landmarks_sequence, SEQUENCE_LENGTH, BASE_FEATURE_DIM

class ASLInferenceEngine:
    """
    Manages model execution, rolling temporal buffers, prediction smoothing, and top-K extraction.
    """
    def __init__(self, confidence_threshold: float = 0.20, smoothing_window: int = 5):
        self.model, self.class_names, self.device = load_asl_model()
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.prediction_history = collections.deque(maxlen=smoothing_window)

    def predict_sample(self, sample_696: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs model inference on a (64, 696) numpy array.
        """
        if sample_696.shape != (SEQUENCE_LENGTH, 696):
            raise ValueError(f"Expected shape ({SEQUENCE_LENGTH}, 696), got {sample_696.shape}")

        # Check if hands are present in the temporal sequence
        # Left Hand (222:285) & Right Hand (285:348) local coordinates
        hand_features = sample_696[:, 222:348]
        has_hands = np.max(np.abs(hand_features)) > 1e-4

        if not has_hands:
            return {
                "success": True,
                "prediction": "No hands in view",
                "is_confident": False,
                "raw_top_class": None,
                "confidence": 0.0,
                "top_predictions": []
            }

        tensor_in = torch.from_numpy(sample_696).unsqueeze(0).to(self.device)  # (1, 64, 696)
        
        with torch.no_grad():
            logits = self.model(tensor_in)  # (1, 95)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (95,)

        top_indices = np.argsort(probs)[::-1][:top_k]
        
        top_predictions = [
            {
                "class": self.class_names.get(int(idx), f"class_{idx}"),
                "confidence": float(probs[idx])
            }
            for idx in top_indices
        ]

        top_class = top_predictions[0]["class"]
        top_conf = top_predictions[0]["confidence"]

        # Check threshold
        confident = top_conf >= self.confidence_threshold

        return {
            "success": True,
            "prediction": top_class if confident else "Detecting sign...",
            "is_confident": confident,
            "raw_top_class": top_class,
            "confidence": top_conf,
            "top_predictions": top_predictions
        }

class RollingLivePredictor:
    """
    Maintains a rolling buffer of 348-dim landmark frames for real-time live webcam streams.
    """
    def __init__(self, buffer_size: int = 64, step_size: int = 3, min_frames: int = 16):
        self.engine = ASLInferenceEngine()
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.min_frames = min_frames
        self.frame_buffer = collections.deque(maxlen=buffer_size)
        self.frame_count = 0
        self.last_prediction: Optional[Dict[str, Any]] = None
        self.history_probs = collections.deque(maxlen=5)

    def add_frame(self, frame_features_348: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Appends a new frame (348 features) to the buffer.
        If sufficient frames exist and it's time to infer (throttled by step_size), returns prediction.
        """
        self.frame_buffer.append(frame_features_348)
        self.frame_count += 1

        if len(self.frame_buffer) < self.min_frames:
            return {
                "status": "collecting_frames",
                "buffer_fill": len(self.frame_buffer),
                "buffer_target": self.buffer_size,
                "prediction": "Collecting frames...",
                "confidence": 0.0,
                "top_predictions": []
            }

        # Throttle inference to every step_size frames to preserve CPU/GPU responsiveness
        if self.frame_count % self.step_size != 0 and self.last_prediction is not None:
            return self.last_prediction

        # Prepare temporal sequence
        frames_list = list(self.frame_buffer)
        sample_696 = process_landmarks_sequence(frames_list)
        
        result = self.engine.predict_sample(sample_696, top_k=5)
        result["buffer_fill"] = len(self.frame_buffer)
        result["buffer_target"] = self.buffer_size
        result["status"] = "recognizing" if result["is_confident"] else "detecting"

        self.last_prediction = result
        return result

    def reset(self):
        self.frame_buffer.clear()
        self.frame_count = 0
        self.last_prediction = None
        self.history_probs.clear()
