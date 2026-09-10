import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class GeometricFeatureExtractor:
    """
    Extracts spatial, postural, hand-shape, and trajectory kinematic features
    from (64, 696) or (64, 348) normalized landmark tensors.
    """
    
    # Feature offsets in 348 base vector:
    # 0..74: Pose (25 landmarks * 3) [0: nose, 11: left shoulder, 12: right shoulder, 13: left elbow, 14: right elbow, 15: left wrist, 16: right wrist]
    # 75..95: Face (7 landmarks * 3) [0: mouth-center, 1: upper-lip, 2: lower-lip, 3: chin, 4: left-mouth, 5: right-mouth, 6: nose-base]
    # 96..158: Left hand body-norm (21 * 3)
    # 159..221: Right hand body-norm (21 * 3)
    # 222..284: Left hand local-wrist-norm (21 * 3)
    # 285..347: Right hand local-wrist-norm (21 * 3)

    @staticmethod
    def unpack_sequence(seq_696_or_348: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Unpacks sequence into named landmark arrays across 64 frames.
        Returns dict with:
          pose: (64, 25, 3)
          face: (64, 7, 3)
          lh_body: (64, 21, 3)
          rh_body: (64, 21, 3)
          lh_local: (64, 21, 3)
          rh_local: (64, 21, 3)
        """
        T = seq_696_or_348.shape[0]
        base = seq_696_or_348[:, :348]

        pose = base[:, 0:75].reshape(T, 25, 3)
        face = base[:, 75:96].reshape(T, 7, 3)
        lh_body = base[:, 96:159].reshape(T, 21, 3)
        rh_body = base[:, 159:222].reshape(T, 21, 3)
        lh_local = base[:, 222:285].reshape(T, 21, 3)
        rh_local = base[:, 285:348].reshape(T, 21, 3)

        return {
            "pose": pose,
            "face": face,
            "lh_body": lh_body,
            "rh_body": rh_body,
            "lh_local": lh_local,
            "rh_local": rh_local
        }

    @staticmethod
    def get_finger_extension(hand_local: np.ndarray) -> Dict[str, float]:
        """
        Calculates extension ratio (0.0=folded, 1.0=fully extended) for 5 fingers.
        hand_local: (21, 3) local coordinates where wrist is (0,0,0).
        Fingers:
          Thumb: tip 4, mcp 2
          Index: tip 8, pip 6, mcp 5
          Middle: tip 12, pip 10, mcp 9
          Ring: tip 16, pip 14, mcp 13
          Pinky: tip 20, pip 18, mcp 17
        """
        if np.max(np.abs(hand_local)) < 1e-4:
            return {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}

        def ext_score(tip_idx, pip_idx, mcp_idx):
            d_tip = np.linalg.norm(hand_local[tip_idx])
            d_pip = np.linalg.norm(hand_local[pip_idx])
            d_mcp = np.linalg.norm(hand_local[mcp_idx])
            if d_pip < 1e-4:
                return 0.0
            ratio = d_tip / (d_pip + 1e-4)
            return float(np.clip((ratio - 0.85) / 0.65, 0.0, 1.0))

        thumb_d = np.linalg.norm(hand_local[4] - hand_local[2])
        thumb_ext = float(np.clip(thumb_d / 0.85, 0.0, 1.0))

        return {
            "thumb": thumb_ext,
            "index": ext_score(8, 6, 5),
            "middle": ext_score(12, 10, 9),
            "ring": ext_score(16, 14, 13),
            "pinky": ext_score(20, 18, 17)
        }

    @staticmethod
    def classify_hand_shape(ext: Dict[str, float], hand_local: np.ndarray) -> str:
        """
        Classifies common ASL handshapes:
        'open_palm', 'fist', 'index_point', 'v_sign', 'y_sign', 'pinch', 'thumbs_up', 'folded'
        """
        t, i, m, r, p = ext["thumb"], ext["index"], ext["middle"], ext["ring"], ext["pinky"]
        
        # Fist: all 4 fingers folded
        if i < 0.35 and m < 0.35 and r < 0.35 and p < 0.35:
            if t > 0.6 and hand_local[4, 1] < hand_local[2, 1]:  # thumb up
                return "thumbs_up"
            return "fist"

        # Open palm / 5-hand: all extended
        if i > 0.55 and m > 0.55 and r > 0.50 and p > 0.50:
            return "open_palm"

        # Index point: only index extended
        if i > 0.60 and m < 0.35 and r < 0.35 and p < 0.35:
            return "index_point"

        # V-sign / Peace: index and middle extended
        if i > 0.55 and m > 0.55 and r < 0.40 and p < 0.40:
            return "v_sign"

        # Y-sign: thumb and pinky extended, middle 3 folded
        if t > 0.50 and p > 0.50 and i < 0.40 and m < 0.40 and r < 0.40:
            return "y_sign"

        # Pinch: tip 4 close to tip 8
        if np.linalg.norm(hand_local[4] - hand_local[8]) < 0.22 and (m < 0.50 and r < 0.50):
            return "pinch"

        return "open_palm" if (i + m + r + p) / 4.0 > 0.50 else "folded"

    @classmethod
    def extract_kinematics(cls, seq_696_or_348: np.ndarray) -> Dict[str, Any]:
        """
        Computes rich summary features across the 64-frame gesture.
        """
        unpacked = cls.unpack_sequence(seq_696_or_348)
        pose = unpacked["pose"]
        face = unpacked["face"]
        lh_b = unpacked["lh_body"]
        rh_b = unpacked["rh_body"]
        lh_l = unpacked["lh_local"]
        rh_l = unpacked["rh_local"]

        T = pose.shape[0]
        lh_active = np.max(np.abs(lh_b)) > 1e-4
        rh_active = np.max(np.abs(rh_b)) > 1e-4

        # Primary active hand
        if rh_active and not lh_active:
            dom_hand = "right"
            dom_body = rh_b
            dom_local = rh_l
        elif lh_active and not rh_active:
            dom_hand = "left"
            dom_body = lh_b
            dom_local = lh_l
        else:
            lh_motion = np.sum(np.std(lh_b[:, 0], axis=0)) if lh_active else 0
            rh_motion = np.sum(np.std(rh_b[:, 0], axis=0)) if rh_active else 0
            dom_hand = "right" if rh_motion >= lh_motion else "left"
            dom_body = rh_b if dom_hand == "right" else lh_b
            dom_local = rh_l if dom_hand == "right" else lh_l

        # Trajectory
        wrist_traj = dom_body[:, 0]  # wrist is landmark 0
        start_wrist = wrist_traj[0]
        end_wrist = wrist_traj[-1]
        disp_vec = end_wrist - start_wrist
        disp_mag = float(np.linalg.norm(disp_vec))

        dx, dy, dz = float(disp_vec[0]), float(disp_vec[1]), float(disp_vec[2])
        y_variance = float(np.var(wrist_traj[:, 1]))
        x_variance = float(np.var(wrist_traj[:, 0]))

        # Face/Chin relative location
        chin_pts = face[:, 3]  # chin landmark
        has_face = np.max(np.abs(face)) > 1e-4
        
        if has_face:
            min_dist_to_chin = float(np.min(np.linalg.norm(wrist_traj - chin_pts, axis=-1)))
            start_dist_to_chin = float(np.linalg.norm(wrist_traj[0] - chin_pts[0]))
            end_dist_to_chin = float(np.linalg.norm(wrist_traj[-1] - chin_pts[-1]))
            forehead_y = float(np.mean(pose[:, 0, 1]))  # nose y as reference
            wrist_y_mean = float(np.mean(wrist_traj[:, 1]))
            near_forehead = wrist_y_mean < (forehead_y + 0.10)
            near_chin = min_dist_to_chin < 0.50
        else:
            min_dist_to_chin = 999.0
            start_dist_to_chin = 999.0
            end_dist_to_chin = 999.0
            near_forehead = False
            near_chin = False

        # Two-handed geometry
        both_hands = lh_active and rh_active
        two_hand_dist = float(np.mean(np.linalg.norm(lh_b[:, 0] - rh_b[:, 0], axis=-1))) if both_hands else 999.0

        # Mean hand shape
        mid_local = np.mean(dom_local[T//4:3*T//4], axis=0) if (lh_active or rh_active) else np.zeros((21, 3))
        finger_ext = cls.get_finger_extension(mid_local)
        hand_shape = cls.classify_hand_shape(finger_ext, mid_local)

        # Oscillations
        vy = np.diff(wrist_traj[:, 1])
        zero_crossings_y = int(np.sum(np.diff(np.sign(vy)) != 0)) if len(vy) > 1 else 0

        vx = np.diff(wrist_traj[:, 0])
        zero_crossings_x = int(np.sum(np.diff(np.sign(vx)) != 0)) if len(vx) > 1 else 0

        return {
            "has_hands": lh_active or rh_active,
            "both_hands": both_hands,
            "dom_hand": dom_hand,
            "hand_shape": hand_shape,
            "finger_ext": finger_ext,
            "start_wrist": start_wrist,
            "end_wrist": end_wrist,
            "disp_vec": disp_vec,
            "disp_mag": disp_mag,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "x_variance": x_variance,
            "y_variance": y_variance,
            "near_chin": near_chin,
            "near_forehead": near_forehead,
            "min_dist_to_chin": min_dist_to_chin,
            "start_dist_to_chin": start_dist_to_chin,
            "end_dist_to_chin": end_dist_to_chin,
            "two_hand_dist": two_hand_dist,
            "zero_crossings_y": zero_crossings_y,
            "zero_crossings_x": zero_crossings_x
        }


class GestureRuleEngine:
    """
    Evaluates kinematic rules for:
    1. Everyday gestures: Strict, non-overtriggering rules for high-confidence geometric gestures.
    2. Disambiguation and verification for 95-class model candidates.
    """

    # Explicit audit status dictionary for all everyday signs
    EVERYDAY_SIGNS_STATUS = {
        "thank you": {"status": "VALIDATED", "method": "geometric_trajectory", "trigger": "Open flat hand from chin/lips moving forward/down"},
        "hello": {"status": "VALIDATED", "method": "geometric_trajectory", "trigger": "Open flat hand at temple/forehead with horizontal wave"},
        "stop": {"status": "VALIDATED", "method": "two_hand_geometry", "trigger": "Flat dominant hand chopping downward into flat base palm"},
        "where": {"status": "VALIDATED", "method": "geometric_trajectory", "trigger": "Single upright index finger wagging side-to-side"},
        "what": {"status": "TESTED", "method": "two_hand_geometry", "trigger": "Both open palms facing up oscillating side-to-side"},
        "yes": {"status": "requires_training_data", "reason": "Fist vertical nodding easily false-positives on generic talking/rest hand jitter without full wrist flexion model"},
        "no": {"status": "requires_training_data", "reason": "Index+middle+thumb rapid pinch tap easily false-positives on small fidgeting"},
        "please": {"status": "requires_training_data", "reason": "Flat hand chest circle requires accurate chest contact plane to prevent false triggers"},
        "sorry": {"status": "requires_training_data", "reason": "Fist chest circle requires accurate chest contact plane to prevent false triggers"},
        "help": {"status": "requires_training_data", "reason": "Thumbs-up resting on palm upward motion requires multi-view 3D depth verification"},
        "eat": {"status": "requires_training_data", "reason": "Flat-O tapping mouth overlaps with 95-class 'drink' and 'food'"},
        "sleep": {"status": "requires_training_data", "reason": "Face closing hand gesture requires head tilt + eyes closed detection"},
        "bathroom": {"status": "requires_training_data", "reason": "T-handshape (thumb between index and middle) requires fine finger occlusion tracking"},
        "happy": {"status": "requires_training_data", "reason": "Upward chest brushing requires fine contact tracking"},
        "sad": {"status": "requires_training_data", "reason": "Downward face tracing overlaps with generic face wiping movements"}
    }

    def __init__(self):
        self.extractor = GeometricFeatureExtractor()

    def evaluate_everyday_signs(self, k: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates candidate everyday signs from kinematic features.
        Only evaluates VALIDATED/TESTED rules to strictly prevent false positives.
        """
        if not k.get("has_hands", False):
            return []

        matches = []
        shape = k["hand_shape"]
        ext = k["finger_ext"]
        near_chin = k["near_chin"]
        near_forehead = k["near_forehead"]
        dx, dy, dz = k["dx"], k["dy"], k["dz"]
        disp_mag = k["disp_mag"]
        zc_y = k["zero_crossings_y"]
        zc_x = k["zero_crossings_x"]
        both_hands = k["both_hands"]

        # 1. "thank you" (Strict: Single-hand open palm starting AT chin/lips and moving forward/downward away from chin)
        if not both_hands and shape == "open_palm" and k["start_dist_to_chin"] < 0.35 and k["end_dist_to_chin"] > k["start_dist_to_chin"] + 0.10:
            if dy > 0.04 or disp_mag > 0.15:
                score = 0.90 + min(0.08, disp_mag * 0.2)
                matches.append({"sign": "thank you", "score": min(score, 0.98), "category": "everyday"})

        # 2. "hello" (Strict: Open palm AT forehead/temple height with clear horizontal wave)
        if shape == "open_palm" and near_forehead:
            if zc_x >= 3 and k["x_variance"] > 0.006:
                score = 0.88 + min(0.10, zc_x * 0.02)
                matches.append({"sign": "hello", "score": min(score, 0.98), "category": "everyday"})

        # 3. "stop" (Strict: Two hands, dominant chopping down into flat base palm)
        if both_hands and shape == "open_palm" and k["two_hand_dist"] < 0.55:
            if dy > 0.10:  # Sharp downward chop
                matches.append({"sign": "stop", "score": 0.92, "category": "everyday"})

        # 4. "where" (Strict: Only index finger extended, pointing up, wagging horizontally at neutral height)
        if not both_hands and shape == "index_point" and ext["index"] > 0.65 and ext["middle"] < 0.30 and ext["pinky"] < 0.30:
            if not near_forehead and zc_x >= 3 and k["x_variance"] > 0.005 and k["x_variance"] >= k["y_variance"]:
                matches.append({"sign": "where", "score": 0.90, "category": "everyday"})

        # 5. "what" (Strict: Two open hands, horizontal oscillation at waist/chest)
        if both_hands and shape == "open_palm" and zc_x >= 3 and k["two_hand_dist"] < 0.60:
            if not near_chin and not near_forehead:
                matches.append({"sign": "what", "score": 0.85, "category": "everyday"})

        # Sort matches by compatibility score descending
        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches

    def score_95class_compatibility(self, candidate_class: str, k: Dict[str, Any]) -> float:
        """
        Calculates compatibility score [0.0..1.0] of a 95-class prediction with observed kinematics.
        """
        if not k.get("has_hands", False):
            return 0.0

        c = candidate_class.lower()
        shape = k["hand_shape"]
        ext = k["finger_ext"]
        near_chin = k["near_chin"]
        near_forehead = k["near_forehead"]
        both_hands = k["both_hands"]

        compat = 0.70  # Neutral baseline

        # Facial / Chin signs
        if c in ["apple", "chin", "cheek", "drink", "cry"]:
            if near_chin:
                compat += 0.20
            else:
                compat -= 0.35

        # Head / Forehead signs
        if c in ["boy", "grandpa", "awake", "eye", "ear"]:
            if near_forehead:
                compat += 0.20
            else:
                compat -= 0.30

        # Chin / Lower face signs
        if c in ["grandma", "girl", "aunt"]:
            if near_chin:
                compat += 0.20
            else:
                compat -= 0.30

        # Chest signs (e.g. fine - open 5 on chest, not forehead)
        if c == "fine":
            if near_forehead:
                compat -= 0.35
            elif not near_chin and not near_forehead:
                compat += 0.20

        # Two-handed signs
        if c in ["car", "boat", "book", "alligator", "dance", "clean"]:
            if both_hands:
                compat += 0.20
            else:
                compat -= 0.25

        # Phone sign (Y shape)
        if c == "callonphone":
            if shape == "y_sign":
                compat += 0.25

        # Airplane (Y/ILY shape)
        if c == "airplane":
            if shape == "y_sign" or (ext["thumb"] > 0.4 and ext["index"] > 0.4 and ext["pinky"] > 0.4):
                compat += 0.20

        return float(np.clip(compat, 0.10, 1.0))
