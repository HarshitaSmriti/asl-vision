import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class GeometricFeatureExtractor:
    """
    Extracts spatial, postural, hand-shape, and trajectory kinematic features
    from (64, 696) or (64, 348) normalized landmark tensors.
    """
    
    # Feature offsets in 348 base vector:
    # 0..74: Pose (25 landmarks * 3)
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
            # Extended if tip is significantly farther from wrist than pip & mcp
            if d_pip < 1e-4:
                return 0.0
            ratio = d_tip / (d_pip + 1e-4)
            return float(np.clip((ratio - 0.8) / 0.7, 0.0, 1.0))

        thumb_d = np.linalg.norm(hand_local[4] - hand_local[2])
        thumb_ext = float(np.clip(thumb_d / 0.8, 0.0, 1.0))

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
        'open_palm', 'fist', 'index_point', 'v_sign', 'y_sign', 'pinch', 'thumbs_up', 't_sign', 'unknown'
        """
        t, i, m, r, p = ext["thumb"], ext["index"], ext["middle"], ext["ring"], ext["pinky"]
        
        # Fist / A-shape: all 4 fingers folded
        if i < 0.35 and m < 0.35 and r < 0.35 and p < 0.35:
            if t > 0.6 and hand_local[4, 1] < hand_local[2, 1]:  # thumb up
                return "thumbs_up"
            return "fist"

        # Open palm / 5-hand: all extended
        if i > 0.6 and m > 0.6 and r > 0.6 and p > 0.6:
            return "open_palm"

        # Index point: only index extended
        if i > 0.6 and m < 0.35 and r < 0.35 and p < 0.35:
            return "index_point"

        # V-sign / Peace: index and middle extended
        if i > 0.6 and m > 0.6 and r < 0.4 and p < 0.4:
            return "v_sign"

        # Y-sign: thumb and pinky extended, middle 3 folded
        if t > 0.5 and p > 0.5 and i < 0.4 and m < 0.4 and r < 0.4:
            return "y_sign"

        # Pinch / O-sign: tip 4 close to tip 8
        if np.linalg.norm(hand_local[4] - hand_local[8]) < 0.25:
            return "pinch"

        return "open_palm" if (i + m + r + p) / 4.0 > 0.5 else "folded"

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

        # Primary active hand (default to right hand if both or right active)
        if rh_active and not lh_active:
            dom_hand = "right"
            dom_body = rh_b
            dom_local = rh_l
        elif lh_active and not rh_active:
            dom_hand = "left"
            dom_body = lh_b
            dom_local = lh_l
        else:
            # Check which hand has more motion
            lh_motion = np.sum(np.std(lh_b[:, 0], axis=0)) if lh_active else 0
            rh_motion = np.sum(np.std(rh_b[:, 0], axis=0)) if rh_active else 0
            dom_hand = "right" if rh_motion >= lh_motion else "left"
            dom_body = rh_b if dom_hand == "right" else lh_b
            dom_local = rh_l if dom_hand == "right" else lh_l

        # Wrist position trajectory (T, 3)
        wrist_traj = dom_body[:, 0]  # wrist is index 0
        
        # Net displacement vector & magnitude
        start_wrist = wrist_traj[0]
        end_wrist = wrist_traj[-1]
        mid_wrist = wrist_traj[T // 2]
        disp_vec = end_wrist - start_wrist
        disp_mag = float(np.linalg.norm(disp_vec))

        # Vertical / Horizontal / Depth motion
        dx, dy, dz = float(disp_vec[0]), float(disp_vec[1]), float(disp_vec[2])
        y_variance = float(np.var(wrist_traj[:, 1]))
        x_variance = float(np.var(wrist_traj[:, 0]))

        # Location relative to Face / Chin / Torso
        # Chin is face landmark 3 (or face landmark 2 lower lip)
        chin_pts = face[:, 3]
        has_face = np.max(np.abs(face)) > 1e-4
        
        if has_face:
            dist_to_chin = float(np.mean(np.linalg.norm(wrist_traj - chin_pts, axis=-1)))
            min_dist_to_chin = float(np.min(np.linalg.norm(wrist_traj - chin_pts, axis=-1)))
            
            # Nose / Forehead approx (pose 0 or face 0)
            forehead_y = float(np.mean(pose[:, 0, 1]))
            wrist_y_mean = float(np.mean(wrist_traj[:, 1]))
            near_forehead = wrist_y_mean < (forehead_y + 0.15)
            near_chin = min_dist_to_chin < 0.65
        else:
            dist_to_chin = 999.0
            min_dist_to_chin = 999.0
            near_forehead = False
            near_chin = False

        # Two-handedness
        both_hands = lh_active and rh_active
        two_hand_dist = float(np.mean(np.linalg.norm(lh_b[:, 0] - rh_b[:, 0], axis=-1))) if both_hands else 999.0

        # Mean hand shape across middle frames (20..44)
        mid_local = np.mean(dom_local[T//4:3*T//4], axis=0) if (lh_active or rh_active) else np.zeros((21, 3))
        finger_ext = cls.get_finger_extension(mid_local)
        hand_shape = cls.classify_hand_shape(finger_ext, mid_local)

        # Repetition / Oscillation detection (for nodding 'yes', shaking 'no', circular 'please')
        # Check zero-crossings of velocity
        vy = np.diff(wrist_traj[:, 1])
        zero_crossings_y = int(np.sum(np.diff(np.sign(vy)) != 0))

        vx = np.diff(wrist_traj[:, 0])
        zero_crossings_x = int(np.sum(np.diff(np.sign(vx)) != 0))

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
            "two_hand_dist": two_hand_dist,
            "zero_crossings_y": zero_crossings_y,
            "zero_crossings_x": zero_crossings_x
        }


class GestureRuleEngine:
    """
    Evaluates kinematic rules for:
    1. Everyday gestures (not in the 95-class model): hello, thank you, please, sorry, yes, no, stop, help, what, where, eat, sleep.
    2. Verification & disambiguation for 95-class model candidates.
    """

    def __init__(self):
        self.extractor = GeometricFeatureExtractor()

    def evaluate_everyday_signs(self, k: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates candidate everyday signs from kinematic features.
        Returns list of matched everyday signs sorted by compatibility score.
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

        # 1. "thank you" (Flat open hand starts near chin/mouth and moves forward/downward toward viewer)
        if (shape == "open_palm" or ext["index"] > 0.5) and near_chin:
            # Movement: moving away from face (forward/downward, dy > -0.1 and disp_mag > 0.15)
            score = 0.5
            if k["min_dist_to_chin"] < 0.45:
                score += 0.25
            if dy > 0.05 or disp_mag > 0.20:
                score += 0.20
            if shape == "open_palm":
                score += 0.05
            matches.append({"sign": "thank you", "score": min(score, 0.98), "category": "everyday"})

        # 2. "hello" (Open palm near forehead/temple, waving or moving outward)
        if shape == "open_palm" and (near_forehead or k["start_wrist"][1] < 0.25 or k["min_dist_to_chin"] < 0.85):
            score = 0.60
            if zc_x >= 2:  # waving motion
                score += 0.30
            elif abs(dx) > 0.12:  # salute/outward motion
                score += 0.25
            matches.append({"sign": "hello", "score": min(score, 0.96), "category": "everyday"})

        # 3. "yes" (Fist nodding up and down vertically)
        if shape in ["fist", "thumbs_up", "folded"]:
            if zc_y >= 2 and k["y_variance"] > 0.003:
                score = 0.75 + min(0.20, zc_y * 0.05)
                matches.append({"sign": "yes", "score": min(score, 0.98), "category": "everyday"})

        # 4. "no" (Index + middle finger tapping thumb, or horizontal shake)
        if shape in ["pinch", "v_sign", "open_palm", "folded"]:
            if (ext["index"] > 0.35 and ext["middle"] > 0.35 and ext["pinky"] < 0.45) or shape == "pinch":
                if zc_y >= 2 or zc_x >= 2:
                    score = 0.80 + min(0.18, (zc_y + zc_x) * 0.04)
                    matches.append({"sign": "no", "score": min(score, 0.96), "category": "everyday"})

        # 5. "please" (Flat open hand rubbing circular motion on chest)
        if shape == "open_palm" and not near_chin and not near_forehead:
            if zc_x >= 2 and zc_y >= 2 and k["x_variance"] > 0.005 and k["y_variance"] > 0.005:
                score = 0.82
                matches.append({"sign": "please", "score": score, "category": "everyday"})

        # 6. "sorry" (Fist rubbing circular motion on chest)
        if shape == "fist" and not near_chin and not near_forehead:
            if zc_x >= 2 and zc_y >= 2:
                score = 0.85
                matches.append({"sign": "sorry", "score": score, "category": "everyday"})

        # 7. "stop" (Flat dominant hand chopping down into flat base hand)
        if both_hands and shape == "open_palm" and k["two_hand_dist"] < 0.4:
            if dy > 0.10:
                score = 0.84
                matches.append({"sign": "stop", "score": score, "category": "everyday"})

        # 8. "help" (Thumbs up / fist on open base hand moving upward)
        if both_hands and shape in ["thumbs_up", "fist"] and k["two_hand_dist"] < 0.4:
            if dy < -0.08:  # moving upward
                score = 0.86
                matches.append({"sign": "help", "score": score, "category": "everyday"})

        # 9. "what" (Both hands open, palms up, shaking slightly side to side)
        if both_hands and shape == "open_palm" and zc_x >= 3:
            score = 0.80
            matches.append({"sign": "what", "score": score, "category": "everyday"})

        # 10. "where" (Single index finger pointing up, shaking side to side)
        if shape == "index_point" and zc_x >= 3:
            score = 0.85
            matches.append({"sign": "where", "score": score, "category": "everyday"})

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
        disp_mag = k["disp_mag"]
        zc_y = k["zero_crossings_y"]

        # Default neutral compatibility score
        compat = 0.70

        # Facial / Head signs
        if c in ["apple", "chin", "cheek", "drink", "eat", "food", "cry"]:
            if near_chin:
                compat += 0.25
            else:
                compat -= 0.40

        if c in ["boy", "cap", "grandpa", "awake", "eye", "ear"]:
            if near_forehead:
                compat += 0.25
            else:
                compat -= 0.30

        if c in ["grandma", "girl", "aunt"]:
            if near_chin:
                compat += 0.25
            else:
                compat -= 0.35

        # Two-handed signs
        if c in ["car", "boat", "book", "alligator", "dance", "clean", "shoes"]:
            if both_hands:
                compat += 0.20
            else:
                compat -= 0.25

        # Drinking motion (hand to mouth)
        if c == "drink":
            if near_chin and (shape in ["fist", "thumbs_up", "pinch", "open_palm"]):
                compat += 0.15

        # Call on phone (Y-shape near ear/face)
        if c == "callonphone":
            if shape == "y_sign" or (ext["thumb"] > 0.5 and ext["pinky"] > 0.5):
                compat += 0.30

        # Airplane (I-L-Y or Y shape moving)
        if c == "airplane":
            if shape == "y_sign" or (ext["thumb"] > 0.4 and ext["index"] > 0.4 and ext["pinky"] > 0.4):
                compat += 0.25

        # Fish (flat hand swimming motion)
        if c == "fish":
            if shape == "open_palm" and k["zero_crossings_x"] >= 2:
                compat += 0.20

        return float(np.clip(compat, 0.05, 1.0))
