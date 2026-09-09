import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
from backend.preprocessing import (
    extract_frame_landmarks,
    normalize_frame,
    process_landmarks_sequence,
    BASE_FEATURE_DIM,
    FINAL_FEATURE_DIM,
    SEQUENCE_LENGTH
)

class HolisticLandmarkDetector:
    """
    Wrapper around MediaPipe Holistic solution for video and image landmark extraction.
    """
    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Processes a single BGR image and returns (348_features, landmark_points_dict).
        """
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_image)

        pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present = extract_frame_landmarks(
            results.pose_landmarks,
            results.face_landmarks,
            results.left_hand_landmarks,
            results.right_hand_landmarks
        )

        features_348 = normalize_frame(pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present)

        landmarks_dict = {
            "pose": pose_arr.tolist(),
            "face": face_arr.tolist(),
            "left_hand": lh_arr.tolist() if lh_present else [],
            "right_hand": rh_arr.tolist() if rh_present else [],
            "left_hand_present": lh_present,
            "right_hand_present": rh_present
        }

        return features_348, landmarks_dict

    def process_video_path(self, video_path: str, max_frames: int = 300) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Processes an entire video file, extracting landmarks from each frame.
        Returns:
            features_696: (64, 696) tensor ready for model inference
            landmarks_timeline: list of landmark dicts per sampled frame
            fps: video FPS
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_348 = []
        landmarks_timeline = []

        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            feat_348, lm_dict = self.process_frame(frame)
            frames_348.append(feat_348)
            landmarks_timeline.append(lm_dict)
            frame_count += 1

        cap.release()

        if len(frames_348) == 0:
            raise ValueError("No valid frames could be decoded from the video.")

        sample_696 = process_landmarks_sequence(frames_348)
        return sample_696, landmarks_timeline, fps

    def close(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()
