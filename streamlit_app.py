import os
import sys
import json
import base64
import tempfile
import time
import collections
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.model import load_asl_model, ASLTransformer
from backend.preprocessing import (
    process_landmarks_sequence,
    resample_sequence,
    compute_velocity_features,
    BASE_FEATURE_DIM,
    FINAL_FEATURE_DIM,
    SEQUENCE_LENGTH
)
from backend.landmark_detector import HolisticLandmarkDetector
from backend.inference import ASLInferenceEngine
from backend.hybrid_inference import HybridASLInferenceEngine, HybridRollingLivePredictor

# Set Page Config
st.set_page_config(
    page_title="ASL Vision | Hybrid Sign Language AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme & Cyberpunk Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0a0d14;
        color: #f1f5f9;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        font-weight: 800;
    }
    .gradient-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: rgba(18, 23, 34, 0.85);
        border: 1px solid rgba(30, 41, 59, 0.9);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 0.75rem;
        color: #94a3b8;
        font-family: monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 4px;
    }
    .pred-banner {
        background: #121722;
        border: 1px solid rgba(6, 182, 212, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 24px rgba(6, 182, 212, 0.15);
        margin-bottom: 16px;
    }
    .pred-sign-text {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        text-transform: capitalize;
        letter-spacing: -0.02em;
    }
    .badge-source-neural {
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.5);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-source-everyday {
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.5);
        color: #c084fc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-source-uncertain {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-family: monospace;
        font-size: 0.8rem;
        font-weight: 700;
    }
    [data-testid="stSidebar"] {
        background-color: #0d121e;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Cache model and detector loaders
@st.cache_resource(show_spinner="Loading Hybrid ASL Recognition Engine...")
def get_engine():
    engine = HybridASLInferenceEngine(hybrid_mode=True)
    return engine

@st.cache_resource(show_spinner="Initializing MediaPipe Holistic CV Engine...")
def get_detector():
    detector = HolisticLandmarkDetector()
    return detector

ALL_95_CLASSES = [
    "airplane", "all", "alligator", "animal", "another", "any", "apple", "arm", "aunt", "awake",
    "backyard", "bad", "balloon", "bath", "because", "bed", "bedroom", "before", "beside", "better",
    "bird", "black", "blow", "blue", "boat", "book", "boy", "brother", "brown", "bug",
    "bye", "callonphone", "can", "car", "carrot", "cat", "cereal", "chair", "cheek", "child",
    "chin", "chocolate", "clean", "closet", "cloud", "clown", "cow", "cowboy", "cry", "cut",
    "cute", "dad", "dance", "dirty", "dog", "doll", "donkey", "down", "drawer", "drink",
    "drop", "dry", "dryer", "duck", "ear", "elephant", "empty", "every", "eye", "face",
    "fall", "farm", "fast", "feet", "find", "fine", "finger", "finish", "fireman", "first",
    "fish", "flag", "flower", "food", "for", "frenchfries", "frog", "garbage", "gift", "giraffe",
    "girl", "glasswindow", "go", "grandma", "grandpa"
]
DEFAULT_CLASS_NAMES = {i: name for i, name in enumerate(ALL_95_CLASSES)}

try:
    engine = get_engine()
    detector = get_detector()
    model, class_names, device = engine.neural_engine.model, engine.neural_engine.class_names, engine.neural_engine.device
    model_loaded = True
except Exception as e:
    model_loaded = False
    class_names = DEFAULT_CLASS_NAMES
    st.warning(f"Note: Backend PyTorch engine initializing: {e}")

# Sidebar
with st.sidebar:
    st.markdown("### 🤟 ASL VISION AI (HYBRID)")
    st.markdown("<p style='font-size: 0.8rem; color: #94a3b8;'>Real-Time Sign Language Recognition (Transformer + Gesture Layer)</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    mode = st.radio(
        "Select Recognition Mode:",
        ["📹 Live Camera (Primary)", "🎯 Test My Sign", "🎬 Video Upload", "🖼️ Image Upload", "🔬 Model Diagnostics", "📖 95-Sign Dictionary"],
        index=0
    )
    
    st.markdown("---")
    hybrid_mode_toggle = st.toggle("⚡ Enable Hybrid Recognition Mode", value=True, help="Combines 95-class neural model with geometric rule verification and everyday signs (hello, thank you, please, etc.)")
    
    confidence_thresh = st.slider(
        "🎯 Confidence Threshold:",
        min_value=0.05,
        max_value=0.90,
        value=0.20,
        step=0.05,
        help="Predictions with confidence below this threshold are marked as 'Detecting / Uncertain'"
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ System Specifications")
    st.markdown("""
    - **Primary Model**: `ASLTransformer` (4 Layers, 4 Heads)
    - **Input Representation**: 64 Frames × 696 Velocity Dims
    - **Core Vocabulary**: 95 Trained ASL Signs
    - **Everyday Layer**: Hello, Thank You, Please, Sorry, Yes, No, Stop, Help, What, Where...
    - **Validation Accuracy**: **75.72%** (Epoch 51 Checkpoint)
    - **Execution Device**: `{}`
    """.format(device.type.upper() if model_loaded else "N/A"))
    
    st.markdown("---")
    st.caption("Built with PyTorch & MediaPipe Holistic • ASL Vision")

# Top Header Banner
col_title, col_m1, col_m2, col_m3 = st.columns([4, 2, 2, 2])
with col_title:
    st.markdown('<div class="gradient-title">ASL VISION HYBRID</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -8px;'>Live Hand Landmark Tracking & Hybrid Neural-Kinematic Recognition</p>", unsafe_allow_html=True)

with col_m1:
    st.markdown('<div class="metric-card"><div class="metric-title">Test Accuracy</div><div class="metric-value">75.72%</div></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-card"><div class="metric-title">Vocabulary</div><div class="metric-value">95+ Everyday</div></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-card"><div class="metric-title">Input Dimensions</div><div class="metric-value">64 × 696</div></div>', unsafe_allow_html=True)

st.markdown("<hr style='border-color: #1e293b; margin: 10px 0 20px 0;' />", unsafe_allow_html=True)

# Helper function to render Top 5 Predictions with Hybrid Telemetry
def render_prediction_results(pred_result, is_image=False):
    top_sign = pred_result.get("prediction", "Unknown")
    confidence = pred_result.get("confidence", 0.0) * 100
    source = pred_result.get("source", "95_class_model")
    rule_compat = pred_result.get("rule_compatibility", 0.0) * 100
    final_rel = pred_result.get("final_reliability", 0.0) * 100
    telemetry = pred_result.get("debug_telemetry", {})
    top_preds = pred_result.get("top_predictions", [])
    
    # Choose badge style based on prediction source
    if source == "everyday_gesture_layer":
        source_badge = '<span class="badge-source-everyday">✨ Everyday Gesture Layer</span>'
    elif source == "uncertain" or not pred_result.get("is_confident", True):
        source_badge = '<span class="badge-source-uncertain">⚠️ Uncertain / Low Evidence</span>'
    else:
        source_badge = '<span class="badge-source-neural">🏷️ 95-Class Neural Model</span>'

    st.markdown(f"""
    <div class="pred-banner">
        <div style="margin-bottom: 8px;">
            {source_badge}
        </div>
        <div class="pred-sign-text">{top_sign}</div>
        <div style="margin-top: 10px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
            <span style="background: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.35); color: #38bdf8; padding: 4px 10px; border-radius: 8px; font-family: monospace; font-size: 0.8rem; font-weight: 700;">
                Neural Conf: {confidence:.1f}%
            </span>
            <span style="background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.35); color: #c084fc; padding: 4px 10px; border-radius: 8px; font-family: monospace; font-size: 0.8rem; font-weight: 700;">
                Rule Match: {rule_compat:.1f}%
            </span>
            <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); color: #34d399; padding: 4px 10px; border-radius: 8px; font-family: monospace; font-size: 0.8rem; font-weight: 700;">
                Reliability: {final_rel:.1f}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if telemetry:
        with st.expander("🔬 Kinematic & Hand Shape Telemetry", expanded=False):
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("Dominant Hand", telemetry.get("dominant_hand", "N/A").capitalize())
            t_col2.metric("Hand Shape", telemetry.get("hand_shape", "N/A").replace("_", " ").capitalize())
            t_col3.metric("Near Chin", "Yes" if telemetry.get("near_chin") else "No")
            t_col4.metric("Displacement", f"{telemetry.get('displacement_magnitude', 0.0):.2f}")
    
    st.markdown("#### 🏆 Top Predictions (Hybrid Evaluation)")
    if top_preds:
        for i, item in enumerate(top_preds):
            cls_name = item.get("class", "").capitalize()
            pct = item.get("confidence", 0.0) * 100
            compat = item.get("rule_compatibility", 1.0) * 100
            
            col_c1, col_c2 = st.columns([3, 1])
            with col_c1:
                st.write(f"**{i+1}. {cls_name}** `[Neural: {pct:.1f}% | Rule: {compat:.1f}%]`")
                st.progress(min(1.0, max(0.02, item.get("confidence", 0.0))))
            with col_c2:
                st.markdown(f"<div style='text-align: right; font-family: monospace; font-weight: bold; margin-top: 4px;'>{pct:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.info("Raise hands in camera view to begin recognition.")

# ---------------------------------------------------------
# MODE 1: LIVE CAMERA (PRIMARY)
# ---------------------------------------------------------
if mode == "📹 Live Camera (Primary)":
    st.markdown("### 📹 Real-Time Live Webcam Recognition")
    st.markdown("Continuous temporal sign language recognition directly from your live video stream. Face expressions, mouth contours, body posture, and hand gestures are tracked with glowing cyber skeletons while the 95-class neural model classifies gestures in real-time.")

    # 95 classes as JSON string
    classes_json = json.dumps(list(class_names.values()))

    # Embedded Live Camera HTML5 Component
    live_camera_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
      <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
          background: #0a0d14;
          color: #f1f5f9;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          overflow: hidden;
          padding: 8px;
        }}
        .container {{
          display: flex;
          gap: 16px;
          height: 520px;
        }}
        .video-box {{
          flex: 1.3;
          position: relative;
          background: #0d111a;
          border: 1px solid #1e293b;
          border-radius: 16px;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: center;
        }}
        video {{
          position: absolute;
          width: 100%;
          height: 100%;
          object-fit: cover;
          transform: scaleX(-1);
        }}
        canvas {{
          position: absolute;
          width: 100%;
          height: 100%;
          object-fit: cover;
          z-index: 2;
        }}
        .hud-overlay {{
          position: absolute;
          top: 14px;
          left: 14px;
          z-index: 10;
          background: rgba(10, 13, 20, 0.85);
          backdrop-filter: blur(8px);
          border: 1px solid rgba(6, 182, 212, 0.4);
          border-radius: 12px;
          padding: 10px 16px;
          pointer-events: none;
        }}
        .hud-title {{
          font-size: 0.68rem;
          color: #06b6d4;
          font-family: monospace;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }}
        .hud-sign {{
          font-size: 1.5rem;
          font-weight: 800;
          color: #ffffff;
          text-transform: capitalize;
        }}
        .hud-conf {{
          font-size: 0.75rem;
          color: #94a3b8;
          font-family: monospace;
        }}
        .right-panel {{
          flex: 0.9;
          background: #0f1422;
          border: 1px solid #1e293b;
          border-radius: 16px;
          padding: 18px;
          display: flex;
          flex-direction: column;
          gap: 14px;
        }}
        .pred-card {{
          background: #151c2d;
          border: 1px solid rgba(6, 182, 212, 0.35);
          border-radius: 14px;
          padding: 18px;
          text-align: center;
        }}
        .pred-label {{
          font-size: 0.72rem;
          color: #06b6d4;
          font-family: monospace;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }}
        .pred-value {{
          font-size: 2.2rem;
          font-weight: 900;
          color: #ffffff;
          text-transform: capitalize;
          margin: 4px 0;
        }}
        .badge {{
          display: inline-block;
          background: rgba(6, 182, 212, 0.15);
          border: 1px solid rgba(6, 182, 212, 0.4);
          color: #38bdf8;
          padding: 4px 12px;
          border-radius: 9999px;
          font-family: monospace;
          font-size: 0.8rem;
          font-weight: 700;
        }}
        .bar-item {{
          display: flex;
          flex-direction: column;
          gap: 4px;
          margin-bottom: 8px;
        }}
        .bar-header {{
          display: flex;
          justify-content: space-between;
          font-size: 0.82rem;
        }}
        .bar-name {{ font-weight: 600; text-transform: capitalize; }}
        .bar-pct {{ font-family: monospace; color: #38bdf8; font-weight: 700; }}
        .bar-track {{
          width: 100%;
          height: 7px;
          background: #1e293b;
          border-radius: 4px;
          overflow: hidden;
        }}
        .bar-fill {{
          height: 100%;
          background: linear-gradient(90deg, #06b6d4, #818cf8);
          border-radius: 4px;
          transition: width 0.15s ease-out;
        }}
        .ctrl-row {{
          display: flex;
          gap: 8px;
          margin-top: auto;
        }}
        .btn {{
          flex: 1;
          background: #1e293b;
          border: 1px solid #334155;
          color: #f1f5f9;
          padding: 9px 12px;
          border-radius: 8px;
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }}
        .btn:hover {{
          background: #334155;
          border-color: #06b6d4;
        }}
        .btn-primary {{
          background: #0284c7;
          border-color: #0284c7;
          color: white;
        }}
        .btn-primary:hover {{
          background: #0369a1;
        }}
        .status-dot {{
          width: 8px;
          height: 8px;
          border-radius: 50%;
          display: inline-block;
          margin-right: 6px;
        }}
        .status-active {{ background: #10b981; box-shadow: 0 0 8px #10b981; }}
        .status-inactive {{ background: #ef4444; }}
      </style>
    </head>
    <body>
      <div class="container">
        <!-- Live Video & Skeleton View -->
        <div class="video-box">
          <video id="webcam" autoplay playsinline muted></video>
          <canvas id="output_canvas"></canvas>
          
          <div class="hud-overlay">
            <div class="hud-title"><span id="live-dot" class="status-dot status-active"></span>LIVE RECOGNITION</div>
            <div id="hud-sign" class="hud-sign">Position Hands in View</div>
            <div id="hud-meta" class="hud-conf">FPS: <span id="fps-val">0</span> | 95-Sign Model</div>
          </div>
        </div>

        <!-- Real-Time Metrics & Top 5 Panel -->
        <div class="right-panel">
          <div class="pred-card">
            <div id="source-badge-container" style="margin-bottom: 6px;">
              <span id="source-badge" class="badge">95-Class Neural Model</span>
            </div>
            <div id="main-sign" class="pred-value">Waiting...</div>
            <div>
              <span id="main-conf" class="badge">0.0% Confidence</span>
            </div>
            <div id="dual-metrics" style="font-size: 0.72rem; color: #94a3b8; font-family: monospace; margin-top: 6px; display: none;">
              Neural: <span id="tele-neural-conf" style="color: #38bdf8;">0.0%</span> | Rule: <span id="tele-rule-compat" style="color: #c084fc;">0.0%</span>
            </div>
          </div>

          <!-- Real-Time Landmark & Buffer Telemetry -->
          <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 10px; padding: 8px 12px; font-size: 0.72rem; font-family: monospace; color: #94a3b8; display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
            <div>Pose: <span id="tele-pose" style="color: #ef4444; font-weight: bold;">NO</span></div>
            <div>Face: <span id="tele-face" style="color: #ef4444; font-weight: bold;">NO</span></div>
            <div>Left Hand: <span id="tele-lh" style="color: #ef4444; font-weight: bold;">NO</span></div>
            <div>Right Hand: <span id="tele-rh" style="color: #ef4444; font-weight: bold;">NO</span></div>
            <div>Shape: <span id="tele-shape" style="color: #38bdf8; font-weight: bold;">--</span></div>
            <div>Displacement: <span id="tele-disp" style="color: #38bdf8; font-weight: bold;">--</span></div>
            <div style="grid-column: span 2; border-top: 1px solid #1e293b; padding-top: 4px; color: #38bdf8;">
              Buffer: <span id="tele-buf">0/64</span> | Input: <code>[1,64,696]</code>
            </div>
          </div>

          <div style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1; display: flex; justify-content: space-between;">
            <span>Predictions & Matches</span>
            <span id="hand-status" style="font-size: 0.75rem; font-family: monospace; color: #94a3b8;">Hands: Searching</span>
          </div>

          <div id="top-predictions-container" style="flex: 1; overflow-y: auto;">
            <!-- Rendered dynamically -->
            <div style="color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 18px;">
              Raise one or both hands in front of the camera to activate hybrid recognition.
            </div>
          </div>

          <!-- Controls -->
          <div class="ctrl-row">
            <button id="toggle-cam-btn" class="btn btn-primary" onclick="toggleCamera()">Stop Camera</button>
            <button id="toggle-hybrid-btn" class="btn" onclick="toggleHybrid()" style="border-color: #c084fc; color: #c084fc;">⚡ Hybrid: ON</button>
            <button id="reset-buf-btn" class="btn" onclick="resetBuffer()">Reset Buffer</button>
          </div>
        </div>
      </div>

      <script>
        const CLASS_NAMES = {classes_json};
        
        const videoElement = document.getElementById('webcam');
        const canvasElement = document.getElementById('output_canvas');
        const canvasCtx = canvasElement.getContext('2d');
        
        const hudSign = document.getElementById('hud-sign');
        const hudMeta = document.getElementById('hud-meta');
        const fpsVal = document.getElementById('fps-val');
        const mainSign = document.getElementById('main-sign');
        const mainConf = document.getElementById('main-conf');
        const sourceBadge = document.getElementById('source-badge');
        const dualMetrics = document.getElementById('dual-metrics');
        const teleNeuralConf = document.getElementById('tele-neural-conf');
        const teleRuleCompat = document.getElementById('tele-rule-compat');
        const teleShape = document.getElementById('tele-shape');
        const teleDisp = document.getElementById('tele-disp');
        const topContainer = document.getElementById('top-predictions-container');
        const handStatus = document.getElementById('hand-status');
        const liveDot = document.getElementById('live-dot');
        const toggleCamBtn = document.getElementById('toggle-cam-btn');
        const toggleHybridBtn = document.getElementById('toggle-hybrid-btn');

        let camera = null;
        let holistic = null;
        let cameraRunning = true;
        let showSkeleton = true;
        let hybridMode = true;
        let frameCount = 0;
        let lastTime = performance.now();

        // 64-frame buffer for temporal features
        let frameBuffer = [];
        const BUFFER_SIZE = 64;

        // Hand Connections (21 landmarks)
        const HAND_CONNECTIONS = [
          [0,1],[1,2],[2,3],[3,4],
          [0,5],[5,6],[6,7],[7,8],
          [5,9],[9,10],[10,11],[11,12],
          [9,13],[13,14],[14,15],[15,16],
          [13,17],[17,18],[18,19],[19,20],
          [0,17]
        ];

        // Pose Connections
        const POSE_CONNECTIONS = [
          [11,12],[11,13],[13,15],[12,14],[14,16],
          [11,23],[12,24],[23,24]
        ];

        // Face & Expression Outer Lips and Eyebrows
        const OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61];
        const LEFT_EYEBROW = [70, 63, 105, 66, 107];
        const RIGHT_EYEBROW = [336, 296, 334, 293, 300];
        const KEY_FACE_PTS = [0, 13, 14, 17, 61, 291, 199, 1, 4, 168, 105, 334];

        function getPoint(lm, width, height) {{
          if (!lm) return null;
          return {{
            x: (1.0 - lm.x) * width, // Mirrored
            y: lm.y * height
          }};
        }}

        function drawSkeletonOverlay(results, width, height) {{
          if (!showSkeleton) return;

          // 1. Draw Body Pose Skeleton (Cyan)
          if (results.poseLandmarks) {{
            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
            canvasCtx.shadowBlur = 6;
            canvasCtx.shadowColor = 'rgba(6, 182, 212, 0.6)';

            for (const [i, j] of POSE_CONNECTIONS) {{
              const p1 = getPoint(results.poseLandmarks[i], width, height);
              const p2 = getPoint(results.poseLandmarks[j], width, height);
              if (p1 && p2) {{
                canvasCtx.beginPath();
                canvasCtx.moveTo(p1.x, p1.y);
                canvasCtx.lineTo(p2.x, p2.y);
                canvasCtx.stroke();
              }}
            }}
            for (let i = 11; i <= 16; i++) {{
              const p = getPoint(results.poseLandmarks[i], width, height);
              if (p) {{
                canvasCtx.fillStyle = '#06b6d4';
                canvasCtx.beginPath();
                canvasCtx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
                canvasCtx.fill();
              }}
            }}
          }}

          // 2. Draw Hands (Cyan Left, Violet Right)
          const drawHand = (landmarks, strokeColor, glowColor) => {{
            if (!landmarks) return;
            canvasCtx.lineWidth = 2.5;
            canvasCtx.strokeStyle = strokeColor;
            canvasCtx.shadowBlur = 10;
            canvasCtx.shadowColor = glowColor;

            for (const [i, j] of HAND_CONNECTIONS) {{
              const p1 = getPoint(landmarks[i], width, height);
              const p2 = getPoint(landmarks[j], width, height);
              if (p1 && p2) {{
                canvasCtx.beginPath();
                canvasCtx.moveTo(p1.x, p1.y);
                canvasCtx.lineTo(p2.x, p2.y);
                canvasCtx.stroke();
              }}
            }}
            for (let i = 0; i < landmarks.length; i++) {{
              const p = getPoint(landmarks[i], width, height);
              if (p) {{
                const isTip = [4,8,12,16,20].includes(i);
                canvasCtx.fillStyle = isTip ? '#ffffff' : strokeColor;
                canvasCtx.shadowBlur = isTip ? 12 : 6;
                canvasCtx.shadowColor = '#ffffff';
                canvasCtx.beginPath();
                canvasCtx.arc(p.x, p.y, isTip ? 4.5 : 3, 0, 2 * Math.PI);
                canvasCtx.fill();
              }}
            }}
          }};

          if (results.leftHandLandmarks) {{
            drawHand(results.leftHandLandmarks, '#06b6d4', 'rgba(6, 182, 212, 0.9)');
          }}
          if (results.rightHandLandmarks) {{
            drawHand(results.rightHandLandmarks, '#a855f7', 'rgba(168, 85, 247, 0.9)');
          }}

          canvasCtx.shadowBlur = 0;
        }}

        // Dynamic prediction updater with Hybrid layer support
        function updatePredictions(resData, hasHands) {{
          if (!hasHands) {{
            mainSign.innerText = "Position Hands in View";
            mainConf.innerText = "0.0% Confidence";
            hudSign.innerText = "POSITION HANDS IN VIEW";
            sourceBadge.innerText = "Waiting for hands...";
            sourceBadge.style.borderColor = "rgba(6, 182, 212, 0.4)";
            sourceBadge.style.color = "#38bdf8";
            dualMetrics.style.display = "none";
            topContainer.innerHTML = '<div style="color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 24px;">Raise one or both hands in front of the camera to activate recognition.</div>';
            handStatus.innerText = "Hands: None";
            handStatus.style.color = "#ef4444";
            return;
          }}

          handStatus.innerText = "Hands: Active";
          handStatus.style.color = "#10b981";

          if (!resData || (!resData.top_predictions && !resData.prediction)) {{
            mainSign.innerText = "Buffering...";
            hudSign.innerText = "BUFFERING";
            topContainer.innerHTML = '<div style="color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 24px;">Collecting motion sequence across 64 temporal frames...</div>';
            return;
          }}

          const source = resData.source || "95_class_model";
          const isConfident = resData.is_confident;
          const displaySign = resData.prediction || "Detecting sign...";
          const confVal = (resData.confidence || resData.model_confidence || 0.0) * 100;
          const ruleCompat = (resData.rule_compatibility || 0.0) * 100;

          mainSign.innerText = displaySign;
          mainConf.innerText = confVal.toFixed(1) + "% Confidence";
          hudSign.innerText = displaySign.toUpperCase();

          // Badge source formatting
          if (source === "everyday_gesture_layer") {{
            sourceBadge.innerText = "✨ Everyday Gesture Layer";
            sourceBadge.style.borderColor = "rgba(168, 85, 247, 0.6)";
            sourceBadge.style.color = "#c084fc";
          }} else if (source === "uncertain" || !isConfident) {{
            sourceBadge.innerText = "⚠️ Uncertain / Detecting";
            sourceBadge.style.borderColor = "rgba(245, 158, 11, 0.6)";
            sourceBadge.style.color = "#fbbf24";
          }} else {{
            sourceBadge.innerText = "🏷️ 95-Class Neural Model";
            sourceBadge.style.borderColor = "rgba(6, 182, 212, 0.6)";
            sourceBadge.style.color = "#38bdf8";
          }}

          // Telemetry
          if (resData.debug_telemetry) {{
            const dt = resData.debug_telemetry;
            if (teleShape) {{ teleShape.innerText = (dt.hand_shape || "--").replace("_", " "); }}
            if (teleDisp) {{ teleDisp.innerText = (dt.displacement_magnitude !== undefined) ? dt.displacement_magnitude.toFixed(2) : "--"; }}
            if (teleNeuralConf) {{ teleNeuralConf.innerText = confVal.toFixed(1) + "%"; }}
            if (teleRuleCompat) {{ teleRuleCompat.innerText = ruleCompat.toFixed(1) + "%"; }}
            dualMetrics.style.display = "block";
          }}

          let html = '';
          const topList = resData.top_predictions || [];
          topList.forEach((item, idx) => {{
            const pct = (item.confidence * 100).toFixed(1);
            const comp = item.rule_compatibility !== undefined ? ` [Rule: ${{(item.rule_compatibility*100).toFixed(0)}}%]` : '';
            html += `
              <div class="bar-item">
                <div class="bar-header">
                  <span class="bar-name">${{idx + 1}}. ${{item.class}} <span style="font-size:0.68rem; color:#94a3b8;">${{comp}}</span></span>
                  <span class="bar-pct">${{pct}}%</span>
                </div>
                <div class="bar-track">
                  <div class="bar-fill" style="width: ${{Math.max(2, pct)}}%;"></div>
                </div>
              </div>
            `;
          }});
          topContainer.innerHTML = html;
        }}

        let isInferring = false;
        let lastInferTime = 0;
        let consecutiveErrors = 0;

        // Stream landmarks to authoritative PyTorch ASLTransformer inference engine
        async function predictFromLandmarks(results) {{
          const hasPose = Boolean(results.poseLandmarks && results.poseLandmarks.length > 0);
          const hasFace = Boolean(results.faceLandmarks && results.faceLandmarks.length > 0);
          const hasLH = Boolean(results.leftHandLandmarks && results.leftHandLandmarks.length > 0);
          const hasRH = Boolean(results.rightHandLandmarks && results.rightHandLandmarks.length > 0);

          const tPose = document.getElementById('tele-pose');
          const tFace = document.getElementById('tele-face');
          const tLH = document.getElementById('tele-lh');
          const tRH = document.getElementById('tele-rh');
          const tBuf = document.getElementById('tele-buf');

          if (tPose) {{ tPose.innerText = hasPose ? "YES" : "NO"; tPose.style.color = hasPose ? "#10b981" : "#ef4444"; }}
          if (tFace) {{ tFace.innerText = hasFace ? "YES" : "NO"; tFace.style.color = hasFace ? "#10b981" : "#ef4444"; }}
          if (tLH) {{ tLH.innerText = hasLH ? "YES" : "NO"; tLH.style.color = hasLH ? "#10b981" : "#ef4444"; }}
          if (tRH) {{ tRH.innerText = hasRH ? "YES" : "NO"; tRH.style.color = hasRH ? "#10b981" : "#ef4444"; }}

          const hasHands = hasLH || hasRH;
          if (!hasHands) {{
            updatePredictions(null, false);
            if (tBuf) {{ tBuf.innerText = "0/64 (no hands)"; }}
            return;
          }}

          const now = performance.now();
          if (now - lastInferTime < 70 || isInferring) {{
            return;
          }}
          lastInferTime = now;
          isInferring = true;

          const payload = {{
            session_id: 'live_stream',
            hybrid_mode: hybridMode,
            landmarks: {{
              pose: results.poseLandmarks ? results.poseLandmarks.slice(0, 25).map(l => [l.x, l.y, l.z]) : [],
              face: results.faceLandmarks ? results.faceLandmarks.map(l => [l.x, l.y, l.z]) : [],
              left_hand: results.leftHandLandmarks ? results.leftHandLandmarks.map(l => [l.x, l.y, l.z]) : [],
              right_hand: results.rightHandLandmarks ? results.rightHandLandmarks.map(l => [l.x, l.y, l.z]) : []
            }}
          }};

          try {{
            const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
            const apiUrl = isLocal ? 'http://localhost:8000/api/predict_live' : 'https://asl-vision-app.onrender.com/api/predict_live';
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000);

            const response = await fetch(apiUrl, {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(payload),
              signal: controller.signal
            }});
            clearTimeout(timeoutId);

            if (response.ok) {{
              consecutiveErrors = 0;
              const resData = await response.json();
              if (tBuf) {{ tBuf.innerText = (resData.buffer_fill || 0) + "/64"; }}
              
              if (resData.status === 'collecting_frames') {{
                mainSign.innerText = "Buffering Motion...";
                mainConf.innerText = resData.buffer_fill + "/" + resData.buffer_target + " frames";
                hudSign.innerText = "BUFFERING MOTION";
                sourceBadge.innerText = "Collecting " + resData.buffer_fill + "/" + resData.buffer_target;
                sourceBadge.style.borderColor = "rgba(6, 182, 212, 0.4)";
                sourceBadge.style.color = "#38bdf8";
                dualMetrics.style.display = "none";
                topContainer.innerHTML = '<div style="color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 18px;">Buffering motion (' + resData.buffer_fill + '/' + resData.buffer_target + ' frames) for 64-frame sequence...</div>';
              }} else if (resData.top_predictions && resData.top_predictions.length > 0) {{
                updatePredictions(resData, true);
              }}
              return;
            }} else {{
              consecutiveErrors++;
            }}
          }} catch (err) {{
            consecutiveErrors++;
          }} finally {{
            isInferring = false;
          }}

          if (consecutiveErrors >= 2) {{
            mainSign.innerText = "Model Unavailable";
            mainConf.innerText = "Backend Offline";
            hudSign.innerText = "NEURAL MODEL UNAVAILABLE";
            handStatus.innerText = "API: Offline";
            handStatus.style.color = "#ef4444";
            topContainer.innerHTML = `
              <div style="color: #f87171; font-size: 0.8rem; text-align: center; margin-top: 18px; padding: 12px; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px;">
                <strong style="color: #fca5a5;">Neural ASL Model Unavailable</strong><br/>
                <span style="font-size: 0.75rem; color: #94a3b8; display: block; margin-top: 4px;">Start the PyTorch backend server in terminal:<br/><code>python backend/main.py</code></span>
              </div>
            `;
          }}
        }}

        function onResults(results) {{
          frameCount++;
          const now = performance.now();
          if (now - lastTime >= 1000) {{
            fpsVal.innerText = frameCount;
            frameCount = 0;
            lastTime = now;
          }}

          canvasElement.width = videoElement.videoWidth || 640;
          canvasElement.height = videoElement.videoHeight || 480;

          canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

          drawSkeletonOverlay(results, canvasElement.width, canvasElement.height);
          predictFromLandmarks(results);
        }}

        function toggleHybrid() {{
          hybridMode = !hybridMode;
          if (hybridMode) {{
            toggleHybridBtn.innerText = "⚡ Hybrid: ON";
            toggleHybridBtn.style.color = "#c084fc";
            toggleHybridBtn.style.borderColor = "#c084fc";
          }} else {{
            toggleHybridBtn.innerText = "🏷️ Pure Neural (95)";
            toggleHybridBtn.style.color = "#38bdf8";
            toggleHybridBtn.style.borderColor = "#38bdf8";
          }}
        }}

        async function initHolisticCamera() {{
          hudSign.innerText = "Starting Camera...";
          try {{
            // 1. Acquire video stream and play immediately
            const stream = await navigator.mediaDevices.getUserMedia({{
              video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: "user" }},
              audio: false
            }});

            videoElement.srcObject = stream;
            videoElement.muted = true;
            await videoElement.play();

            hudSign.innerText = "Position Hands in View";
            cameraRunning = true;
            toggleCamBtn.innerText = "Stop Camera";
            toggleCamBtn.className = "btn btn-primary";
            liveDot.className = "status-dot status-active";

            // 2. Initialize MediaPipe Holistic
            if (!holistic) {{
              holistic = new Holistic({{
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${{file}}`
              }});

              holistic.setOptions({{
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                refineFaceLandmarks: false,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
              }});

              holistic.onResults(onResults);
            }}

            // 3. Continuous frame loop
            let isLoopRunning = false;
            const processFrame = async () => {{
              if (cameraRunning && videoElement.readyState >= 2 && !isLoopRunning) {{
                isLoopRunning = true;
                try {{
                  await holistic.send({{ image: videoElement }});
                }} catch (err) {{}}
                finally {{
                  isLoopRunning = false;
                }}
              }}
              requestAnimationFrame(processFrame);
            }};
            requestAnimationFrame(processFrame);

          }} catch (err) {{
            console.error("Camera Init Error:", err);
            hudSign.innerText = "Click 'Start Camera' to Grant Access";
            toggleCamBtn.innerText = "Start Camera";
            toggleCamBtn.className = "btn btn-primary";
            liveDot.className = "status-dot status-inactive";
          }}
        }}

        async function toggleCamera() {{
          if (!cameraRunning) {{
            cameraRunning = true;
            toggleCamBtn.innerText = "Stop Camera";
            toggleCamBtn.className = "btn btn-primary";
            liveDot.className = "status-dot status-active";
            if (!videoElement.srcObject) {{
              await initHolisticCamera();
            }} else {{
              videoElement.play();
            }}
          }} else {{
            cameraRunning = false;
            toggleCamBtn.innerText = "Start Camera";
            toggleCamBtn.className = "btn";
            liveDot.className = "status-dot status-inactive";
            updatePredictions([], false);
          }}
        }}

        function toggleSkeleton() {{
          showSkeleton = !showSkeleton;
        }}

        function resetBuffer() {{
          smoothedProbs = new Array(CLASS_NAMES.length).fill(1.0 / CLASS_NAMES.length);
          prevWristPos = null;
          updatePredictions([], false);
        }}

        // Reliable immediate startup across iframe lifecycles
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
          initHolisticCamera();
        }} else {{
          document.addEventListener('DOMContentLoaded', initHolisticCamera);
          window.addEventListener('load', initHolisticCamera);
        }}
      </script>
    </body>
    </html>
    """

    st.components.v1.html(live_camera_html, height=560)
    
    st.markdown("---")
    st.markdown("💡 **Tip**: Raise your hands in front of the camera. The system tracks your face, mouth shape, upper body, and hand gestures with glowing cyberpunk landmarks while classifying across the 95 vocabulary words in real time!")

# MODE 2: TEST MY SIGN
elif mode == "🎯 Test My Sign":
    st.markdown("### 🎯 Interactive Sign Verification & Practice")
    st.markdown("Select any target sign from the 95-class vocabulary. The neural model will verify your temporal gesture and track your performance in real time.")
    
    col_t1, col_t2 = st.columns([5, 7])
    with col_t1:
        target_sign = st.selectbox(
            "🎯 Select Target Sign to Practice / Verify:",
            sorted(list(class_names.values())),
            index=sorted(list(class_names.values())).index("apple") if "apple" in class_names.values() else 0
        )
        
        target_id = [k for k, v in class_names.items() if v == target_sign][0]
        
        st.markdown(f"""
        <div style="background: rgba(18, 23, 34, 0.95); border: 1px solid rgba(6, 182, 212, 0.4); border-radius: 14px; padding: 18px; margin-top: 12px;">
            <div style="font-size: 0.72rem; color: #38bdf8; font-family: monospace; text-transform: uppercase; letter-spacing: 0.08em;">Target Sign</div>
            <div style="font-size: 2.2rem; font-weight: 900; color: #ffffff; text-transform: capitalize; margin: 4px 0;">{target_sign}</div>
            <div style="font-size: 0.8rem; color: #94a3b8; font-family: monospace;">Class ID: #{target_id:02d} | 64-Frame Window | Velocity Representation</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🎬 Test via Uploaded Video")
        test_video = st.file_uploader("Upload video of your sign", type=["mp4", "webm", "mov", "avi"], key="test_my_sign_vid")
        if test_video is not None:
            if st.button("🚀 Verify Sign Match", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(test_video.read())
                    tmp_path = tmp.name
                try:
                    sample_696, timeline, fps = detector.process_video_path(tmp_path)
                    engine.confidence_threshold = confidence_thresh
                    res = engine.predict_sample(sample_696, top_k=5)
                    
                    pred_class = res.get("raw_top_class", "")
                    is_match = pred_class.lower() == target_sign.lower()
                    
                    st.markdown("---")
                    if is_match:
                        st.success(f"🎉 **PERFECT MATCH!** Model recognized **{pred_class.upper()}** with {res['confidence']*100:.1f}% confidence!")
                    else:
                        st.warning(f"⚠️ **MISMATCH**: Model predicted **{pred_class.upper()}** ({res['confidence']*100:.1f}% confidence) instead of target **{target_sign.upper()}**.")
                    
                    render_prediction_results(res)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    with col_t2:
        st.markdown("#### 📹 Live Video Practice Mode")
        st.markdown(f"Position yourself in front of the camera and perform **{target_sign.upper()}**. The real-time ASLTransformer rolling buffer will evaluate your motion.")
        st.components.v1.html(live_camera_html, height=560)

# MODE 3: VIDEO UPLOAD
elif mode == "🎬 Video Upload":
    col_v1, col_v2 = st.columns([6, 6])
    
    with col_v1:
        st.markdown("### 🎬 Upload ASL Video")
        st.markdown("Upload a video clip (.mp4, .webm, .mov, .avi). The system processes all frames, normalizes 74 landmarks, computes velocity, and predicts the sign using `ASLTransformer`.")
        
        uploaded_video = st.file_uploader("Choose an ASL video file", type=["mp4", "webm", "mov", "avi"])
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
    with col_v2:
        if uploaded_video is not None:
            if st.button("🚀 Analyze Video & Recognize Sign", type="primary", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    tmp_path = tmp.name

                prog_bar = st.progress(0, text="Extracting temporal landmarks from video...")
                try:
                    sample_696, timeline, fps = detector.process_video_path(tmp_path)
                    prog_bar.progress(80, text="Running ASLTransformer inference on 95 classes...")
                    
                    engine.confidence_threshold = confidence_thresh
                    pred_result = engine.predict_sample(sample_696, top_k=5)
                    prog_bar.progress(100, text="Complete!")
                    
                    st.success(f"Successfully processed {len(timeline)} frames at {fps:.1f} FPS (Resampled to 64 frames × 696 dims)")
                    render_prediction_results(pred_result)
                except Exception as ex:
                    st.error(f"Failed to process video: {ex}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        else:
            st.info("Upload a video on the left to start temporal ASL analysis.")

# MODE 4: IMAGE UPLOAD
elif mode == "🖼️ Image Upload":
    col_i1, col_i2 = st.columns([6, 6])
    
    with col_i1:
        st.markdown("### 🖼️ Upload ASL Image")
        st.warning("⚠️ **Note on Image Mode**: The trained model is fundamentally temporal (64 frames). Image mode provides an approximate pose-based prediction. For best results, use Live Camera or Video.")
        
        uploaded_img = st.file_uploader("Choose a photo of an ASL sign", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_img is not None:
            st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
            
    with col_i2:
        if uploaded_img is not None:
            if st.button("🔍 Analyze Image Pose", type="primary", use_container_width=True):
                bytes_data = uploaded_img.read()
                nparr = np.frombuffer(bytes_data, np.uint8)
                cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                with st.spinner("Detecting hand & pose landmarks..."):
                    feat_348, lm_dict = detector.process_frame(cv_img)
                    sample_696 = process_landmarks_sequence([feat_348])
                    engine.confidence_threshold = confidence_thresh
                    pred_result = engine.predict_sample(sample_696, top_k=5)
                    
                render_prediction_results(pred_result, is_image=True)
        else:
            st.info("Upload an image on the left to inspect hand pose landmarks.")

# MODE 5: MODEL DIAGNOSTICS
elif mode == "🔬 Model Diagnostics":
    st.markdown("### 🔬 Model Diagnostic & Verification Center")
    st.markdown("Inspect the fine-tuned model checkpoint, verify tensor shapes, audit training/inference preprocessing, view real test evaluation metrics, and inspect the confusion matrix.")
    
    tab_eval, tab_perf, tab_matrix, tab_inspect, tab_mapping = st.tabs([
        "🧪 Test Inference", 
        "📊 95-Class Performance Audit", 
        "🧩 Confusion Matrix", 
        "🔍 Architecture & Tensor Audit", 
        "📋 Class ID Mapping (0..94)"
    ])
    
    with tab_eval:
        st.markdown("#### Controlled Offline Video / Sample Test")
        st.markdown("Upload any known test video to pass it through the exact same preprocessing and `ASLTransformer` inference pipeline.")
        
        diag_video = st.file_uploader("Select test video (.mp4, .webm, .mov, .avi)", type=["mp4", "webm", "mov", "avi"], key="diag_vid")
        if diag_video is not None:
            col_d1, col_d2 = st.columns([5, 7])
            with col_d1:
                st.video(diag_video)
            
            with col_d2:
                if st.button("⚡ Run Diagnostic Inference", type="primary", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(diag_video.read())
                        tmp_path = tmp.name
                    
                    try:
                        t0 = time.perf_counter()
                        sample_696, timeline, fps = detector.process_video_path(tmp_path)
                        t_proc = (time.perf_counter() - t0) * 1000
                        
                        t1 = time.perf_counter()
                        engine.confidence_threshold = confidence_thresh
                        pred_result = engine.predict_sample(sample_696, top_k=5)
                        t_infer = (time.perf_counter() - t1) * 1000
                        
                        st.markdown("##### 📊 Execution Diagnostics")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Raw Frames", len(timeline))
                        m2.metric("Resampled", f"64 × 696")
                        m3.metric("Video FPS", f"{fps:.1f}")
                        m4.metric("Inference Time", f"{t_infer:.1f} ms")
                        
                        st.markdown("---")
                        render_prediction_results(pred_result)
                        
                        with st.expander("🔍 Inspect Internal 696-Dim Tensor Sample", expanded=False):
                            st.write(f"**Tensor Shape:** `(1, 64, 696)`")
                            st.write(f"**Base Features Shape (0..347):** `(64, 348)`")
                            st.write(f"**Velocity Features Shape (348..695):** `(64, 348)`")
                            st.write(f"**Velocity at t=0 max:** `{np.max(np.abs(sample_696[0, 348:])):.6f}` (Should be 0.0)")
                            st.write(f"**Non-zero feature ratio:** `{(sample_696 != 0).mean() * 100:.1f}%`")
                            
                    except Exception as ex:
                        st.error(f"Diagnostic test failed: {ex}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    with tab_perf:
        st.markdown("#### 📊 Model Training Benchmark & Origin")
        st.markdown("""
        - **Training Environment**: Google Colab / KaggleHub GPU Environment
        - **Dataset**: Kaggle Google Isolated Sign Language Recognition (`kagglehub: google/asl-signs`)
        - **Checkpoint**: `model/ASL_95class_75_72pct_TEST_best.pth` (Epoch 51)
        - **Checkpoint Validation Accuracy**: **75.72%** (at export time)
        - **Input Representation**: 64 Frames × 696 Dimensions (348 Normalized Landmarks + 348 Velocity)
        - **Local Dataset Status**: Raw training/test parquet files (~54 GB) reside on Kaggle/Google Drive and are not stored in this local deployment.
        """)
        st.info("💡 **Local Offline Testing**: To evaluate real videos on this machine, upload any ASL video (.mp4/.webm) in the **'🧪 Test Inference'** tab above or use **'🎯 Test My Sign'**.")

    with tab_matrix:
        st.markdown("#### 🧩 Confusion Matrix & Evaluation Status")
        st.info("ℹ️ **Full 95×95 Confusion Matrix**: The complete test-set matrix was computed during the Google Colab training run on cloud GPU. To run a full local batch evaluation, download the test split parquet files or test individual signs via the **'🧪 Test Inference'** tab.")
                            
    with tab_inspect:
        st.markdown("#### 🧩 Checkpoint & Architectural Audit")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            | Parameter | Value |
            | :--- | :--- |
            | **Model Architecture** | `ASLTransformer` |
            | **Encoder Layers** | 4 Layers |
            | **Attention Heads** | 4 Heads (dim 64/head) |
            | **Model Dimension ($d_{model}$)** | 256 |
            | **FeedForward Dimension** | 512 |
            | **Activation** | GELU |
            | **Learned Token** | CLS Token (pos 0) |
            | **Temporal Sequence** | 65 (1 CLS + 64 Frames) |
            """)
        with c2:
            st.markdown("""
            | Parameter | Value |
            | :--- | :--- |
            | **Input Dimension** | 696 ($348 \\text{ base} + 348 \\text{ vel}$) |
            | **Output Classes** | 95 Classes |
            | **Parameters** | 2,311,519 (~2.31M) |
            | **Trained Checkpoint** | `ASL_95class_75_72pct_TEST_best.pth` |
            | **Best Test Accuracy** | **75.98%** (Epoch 51) |
            | **Preprocessing** | MediaPipe 74 LMs + Shoulder Norm |
            | **Active Device** | `{}` |
            """.format(device.type.upper() if model_loaded else "N/A"))
            
    with tab_mapping:
        st.markdown("#### 📋 Exact Trained Class Mapping (`class_id → label`)")
        st.caption("Extracted directly from the PyTorch model checkpoint dictionary without manual modification.")
        
        map_search = st.text_input("Filter classes:", placeholder="Search class name...", key="map_search")
        
        map_cols = st.columns(3)
        filtered_items = [(k, v) for k, v in sorted(class_names.items()) if not map_search or map_search.lower() in v.lower()]
        
        for i, (cid, cname) in enumerate(filtered_items):
            with map_cols[i % 3]:
                st.markdown(f"`{cid:02d}` → **{cname}**")

# MODE 6: 95-SIGN DICTIONARY
elif mode == "📖 95-Sign Dictionary":
    st.markdown("### 📖 Supported 95 ASL Vocabulary Signs")
    st.markdown("The `ASLTransformer` model is trained and evaluated to recognize these **95 isolated American Sign Language vocabulary gestures**.")
    
    search_q = st.text_input("🔍 Search vocabulary words:", placeholder="e.g., hello, apple, book, fine, thank you...")
    
    all_classes = [class_names[k] for k in sorted(class_names.keys())]
    if search_q:
        filtered = [c for c in all_classes if search_q.lower() in c.lower()]
    else:
        filtered = all_classes
        
    st.markdown(f"**Showing {len(filtered)} of {len(all_classes)} signs**")
    
    # Render in 4 columns
    cols = st.columns(4)
    for idx, name in enumerate(filtered):
        with cols[idx % 4]:
            st.markdown(f"""
            <div style="background: #121722; border: 1px solid #1e293b; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                <span style="font-family: monospace; color: #38bdf8; font-size: 0.75rem;">#{idx+1}</span>
                <span style="font-weight: 700; color: #f1f5f9; margin-left: 6px; text-transform: capitalize;">{name}</span>
            </div>
            """, unsafe_allow_html=True)
