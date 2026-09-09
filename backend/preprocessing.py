import numpy as np
import scipy.interpolate
from typing import List, Optional, Tuple, Dict, Any

# Landmark configuration
POSE_LANDMARK_COUNT = 25  # Pose landmarks 0..24
FACE_SELECTED_INDICES = [0, 13, 14, 17, 37, 267, 269]  # 7 mouth/lips keypoints
FACE_LANDMARK_COUNT = len(FACE_SELECTED_INDICES)
HAND_LANDMARK_COUNT = 21
TOTAL_LANDMARKS = POSE_LANDMARK_COUNT + FACE_LANDMARK_COUNT + HAND_LANDMARK_COUNT + HAND_LANDMARK_COUNT  # 74
BASE_FEATURE_DIM = 348  # 74*3 (body norm) + 21*3 (lh local) + 21*3 (rh local)
FINAL_FEATURE_DIM = 696  # 348 base + 348 velocity
SEQUENCE_LENGTH = 64

def extract_frame_landmarks(
    pose_landmarks: Optional[Any],
    face_landmarks: Optional[Any],
    left_hand_landmarks: Optional[Any],
    right_hand_landmarks: Optional[Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """
    Extracts raw coordinates (x, y, z) for 74 landmarks from MediaPipe Holistic output.
    Returns:
        pose_arr: (25, 3)
        face_arr: (7, 3)
        lh_arr: (21, 3)
        rh_arr: (21, 3)
        lh_present: bool
        rh_present: bool
    """
    pose_arr = np.zeros((POSE_LANDMARK_COUNT, 3), dtype=np.float32)
    if pose_landmarks is not None and hasattr(pose_landmarks, 'landmark'):
        for i in range(POSE_LANDMARK_COUNT):
            if i < len(pose_landmarks.landmark):
                lm = pose_landmarks.landmark[i]
                pose_arr[i] = [lm.x, lm.y, lm.z]

    face_arr = np.zeros((FACE_LANDMARK_COUNT, 3), dtype=np.float32)
    if face_landmarks is not None and hasattr(face_landmarks, 'landmark'):
        for idx, lm_idx in enumerate(FACE_SELECTED_INDICES):
            if lm_idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[lm_idx]
                face_arr[idx] = [lm.x, lm.y, lm.z]

    lh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
    lh_present = False
    if left_hand_landmarks is not None and hasattr(left_hand_landmarks, 'landmark'):
        lh_present = True
        for i in range(HAND_LANDMARK_COUNT):
            if i < len(left_hand_landmarks.landmark):
                lm = left_hand_landmarks.landmark[i]
                lh_arr[i] = [lm.x, lm.y, lm.z]

    rh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
    rh_present = False
    if right_hand_landmarks is not None and hasattr(right_hand_landmarks, 'landmark'):
        rh_present = True
        for i in range(HAND_LANDMARK_COUNT):
            if i < len(right_hand_landmarks.landmark):
                lm = right_hand_landmarks.landmark[i]
                rh_arr[i] = [lm.x, lm.y, lm.z]

    return pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present

def normalize_frame(
    pose_arr: np.ndarray,
    face_arr: np.ndarray,
    lh_arr: np.ndarray,
    rh_arr: np.ndarray,
    lh_present: bool,
    rh_present: bool
) -> np.ndarray:
    """
    Normalizes a single frame of 74 landmarks into 348 features:
    1. Body normalization: shoulder midpoint & shoulder scale -> (74, 3) -> 222 features
    2. Left hand local normalization: wrist relative & hand scale -> (21, 3) -> 63 features
    3. Right hand local normalization: wrist relative & hand scale -> (21, 3) -> 63 features
    Total = 222 + 63 + 63 = 348 features.
    """
    # 1. Body normalization
    # Left shoulder = idx 11, Right shoulder = idx 12 in pose
    left_shoulder = pose_arr[11]
    right_shoulder = pose_arr[12]
    
    # Check if shoulders are detected
    has_shoulders = (np.linalg.norm(left_shoulder) > 1e-4) and (np.linalg.norm(right_shoulder) > 1e-4)
    if has_shoulders:
        midpoint = (left_shoulder + right_shoulder) / 2.0
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        body_scale = shoulder_dist if shoulder_dist > 1e-3 else 1.0
    else:
        # Fallback to mean of non-zero pose landmarks or default
        non_zero = pose_arr[np.linalg.norm(pose_arr, axis=-1) > 1e-4]
        if len(non_zero) > 0:
            midpoint = np.mean(non_zero, axis=0)
            body_scale = 1.0
        else:
            midpoint = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            body_scale = 1.0

    # Stack all 74 landmarks for body normalization
    all_74 = np.concatenate([pose_arr, face_arr, lh_arr, rh_arr], axis=0)  # (74, 3)
    
    # Mask non-zero landmarks so missing points remain zero
    is_valid_point = np.linalg.norm(all_74, axis=-1) > 1e-5  # (74,)
    body_norm_74 = np.zeros_like(all_74)
    body_norm_74[is_valid_point] = (all_74[is_valid_point] - midpoint) / body_scale
    body_features = body_norm_74.flatten()  # 74 * 3 = 222

    # 2. Left Hand Local Normalization (21 * 3 = 63)
    lh_features = np.zeros((HAND_LANDMARK_COUNT * 3,), dtype=np.float32)
    if lh_present:
        lh_wrist = lh_arr[0]
        # Hand scale: distance between wrist (0) and middle MCP (9) or max span
        mcp = lh_arr[9]
        hand_dist = np.linalg.norm(lh_wrist - mcp)
        lh_scale = hand_dist if hand_dist > 1e-3 else np.max(np.linalg.norm(lh_arr - lh_wrist, axis=-1))
        lh_scale = lh_scale if lh_scale > 1e-3 else 1.0
        
        lh_local = (lh_arr - lh_wrist) / lh_scale
        lh_features = lh_local.flatten()

    # 3. Right Hand Local Normalization (21 * 3 = 63)
    rh_features = np.zeros((HAND_LANDMARK_COUNT * 3,), dtype=np.float32)
    if rh_present:
        rh_wrist = rh_arr[0]
        mcp = rh_arr[9]
        hand_dist = np.linalg.norm(rh_wrist - mcp)
        rh_scale = hand_dist if hand_dist > 1e-3 else np.max(np.linalg.norm(rh_arr - rh_wrist, axis=-1))
        rh_scale = rh_scale if rh_scale > 1e-3 else 1.0
        
        rh_local = (rh_arr - rh_wrist) / rh_scale
        rh_features = rh_local.flatten()

    # Concatenate all 348 features
    frame_features = np.concatenate([body_features, lh_features, rh_features], axis=0)  # (348,)
    return frame_features

def resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Resamples a temporal sequence of shape (T, feature_dim) to (target_length, feature_dim).
    Uses linear interpolation along the time dimension.
    """
    T, D = sequence.shape
    if T == target_length:
        return sequence.astype(np.float32)
    
    if T == 1:
        # Stationary repetition
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    orig_indices = np.linspace(0, 1, T)
    target_indices = np.linspace(0, 1, target_length)
    
    interpolator = scipy.interpolate.interp1d(orig_indices, sequence, axis=0, kind='linear', fill_value='extrapolate')
    resampled = interpolator(target_indices)
    return resampled.astype(np.float32)

def compute_velocity_features(sequence: np.ndarray) -> np.ndarray:
    """
    Computes frame-to-frame velocity features and concatenates with original sample:
    sample: (64, 348)
    velocity[0] = zeros(348)
    velocity[t] = sample[t] - sample[t-1]
    final_sample = concatenate([sample, velocity], axis=-1) -> (64, 696)
    """
    T, D = sequence.shape
    velocity = np.zeros_like(sequence)
    if T > 1:
        velocity[1:] = sequence[1:] - sequence[:-1]
    
    final_sample = np.concatenate([sequence, velocity], axis=-1)  # (64, 696)
    return final_sample.astype(np.float32)

def process_landmarks_sequence(frames_348: List[np.ndarray]) -> np.ndarray:
    """
    Takes a list/array of 348-dim frame features, resamples to 64 frames, and computes velocity.
    Returns: (64, 696) numpy float32 array ready for ASLTransformer input.
    """
    if len(frames_348) == 0:
        # Empty fallback
        seq = np.zeros((SEQUENCE_LENGTH, BASE_FEATURE_DIM), dtype=np.float32)
    else:
        seq = np.array(frames_348, dtype=np.float32)
    
    resampled = resample_sequence(seq, target_length=SEQUENCE_LENGTH)
    sample_696 = compute_velocity_features(resampled)
    return sample_696

def process_client_landmarks_dict(data: Dict[str, Any]) -> np.ndarray:
    """
    Processes client-side landmark payload sent via WebSocket / JSON:
    data = {
      'pose': [[x,y,z], ...], # 25 or 33 points
      'face': [[x,y,z], ...], # 7 points or full face
      'left_hand': [[x,y,z], ...], # 21 points
      'right_hand': [[x,y,z], ...] # 21 points
    }
    Returns: (348,) normalized frame feature vector.
    """
    pose_raw = data.get('pose') or []
    face_raw = data.get('face') or []
    lh_raw = data.get('left_hand') or []
    rh_raw = data.get('right_hand') or []

    pose_arr = np.zeros((POSE_LANDMARK_COUNT, 3), dtype=np.float32)
    for i in range(min(len(pose_raw), POSE_LANDMARK_COUNT)):
        pt = pose_raw[i]
        if pt and len(pt) >= 2:
            pose_arr[i] = [pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0]

    face_arr = np.zeros((FACE_LANDMARK_COUNT, 3), dtype=np.float32)
    if len(face_raw) == FACE_LANDMARK_COUNT:
        for i in range(FACE_LANDMARK_COUNT):
            pt = face_raw[i]
            if pt and len(pt) >= 2:
                face_arr[i] = [pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0]
    elif len(face_raw) > FACE_LANDMARK_COUNT:
        for idx, lm_idx in enumerate(FACE_SELECTED_INDICES):
            if lm_idx < len(face_raw):
                pt = face_raw[lm_idx]
                if pt and len(pt) >= 2:
                    face_arr[idx] = [pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0]

    lh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
    lh_present = len(lh_raw) > 0
    for i in range(min(len(lh_raw), HAND_LANDMARK_COUNT)):
        pt = lh_raw[i]
        if pt and len(pt) >= 2:
            lh_arr[i] = [pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0]

    rh_arr = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
    rh_present = len(rh_raw) > 0
    for i in range(min(len(rh_raw), HAND_LANDMARK_COUNT)):
        pt = rh_raw[i]
        if pt and len(pt) >= 2:
            rh_arr[i] = [pt[0], pt[1], pt[2] if len(pt) > 2 else 0.0]

    return normalize_frame(pose_arr, face_arr, lh_arr, rh_arr, lh_present, rh_present)
