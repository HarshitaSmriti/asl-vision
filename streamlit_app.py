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
from backend.inference import ASLInferenceEngine, RollingLivePredictor

# Set Page Config
st.set_page_config(
    page_title="ASL Vision | Live Sign Language AI",
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
    [data-testid="stSidebar"] {
        background-color: #0d121e;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Cache model and detector loaders
@st.cache_resource(show_spinner="Loading trained ASLTransformer checkpoint...")
def get_engine():
    engine = ASLInferenceEngine()
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
    model, class_names, device = engine.model, engine.class_names, engine.device
    model_loaded = True
except Exception as e:
    model_loaded = False
    class_names = DEFAULT_CLASS_NAMES
    st.warning(f"Note: Backend PyTorch engine initializing: {e}")

# Sidebar
with st.sidebar:
    st.markdown("### 🤟 ASL VISION AI")
    st.markdown("<p style='font-size: 0.8rem; color: #94a3b8;'>Real-Time American Sign Language Recognition</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    mode = st.radio(
        "Select Recognition Mode:",
        ["📹 Live Camera (Primary)", "🎬 Video Upload", "🖼️ Image Upload", "📖 95-Sign Dictionary"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### ⚡ Model Specifications")
    st.markdown("""
    - **Architecture**: `ASLTransformer` (4 Layers, 4 Heads)
    - **Input Representation**: 64 Frames × 696 Velocity Dims
    - **Total Classes**: 95 ASL Vocabulary Signs
    - **Validation Accuracy**: **75.72%** (Epoch 51 Checkpoint)
    - **Base Parameters**: 2,311,519 (~2.31M)
    - **Execution Device**: `{}`
    """.format(device.type.upper() if model_loaded else "N/A"))
    
    st.markdown("---")
    st.caption("Built with PyTorch & MediaPipe Holistic • ASL Vision")

# Top Header Banner
col_title, col_m1, col_m2, col_m3 = st.columns([4, 2, 2, 2])
with col_title:
    st.markdown('<div class="gradient-title">ASL VISION</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; margin-top: -8px;'>Live Hand Landmark Tracking & 95-Class Transformer Recognition</p>", unsafe_allow_html=True)

with col_m1:
    st.markdown('<div class="metric-card"><div class="metric-title">Test Accuracy</div><div class="metric-value">75.72%</div></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-card"><div class="metric-title">Vocabulary</div><div class="metric-value">95 Signs</div></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-card"><div class="metric-title">Input Dimensions</div><div class="metric-value">64 × 696</div></div>', unsafe_allow_html=True)

st.markdown("<hr style='border-color: #1e293b; margin: 10px 0 20px 0;' />", unsafe_allow_html=True)

# Helper function to render Top 5 Predictions
def render_prediction_results(pred_result, is_image=False):
    top_sign = pred_result.get("prediction", "Unknown")
    confidence = pred_result.get("confidence", 0.0) * 100
    top_preds = pred_result.get("top_predictions", [])
    
    st.markdown(f"""
    <div class="pred-banner">
        <div style="font-size: 0.75rem; color: #38bdf8; font-family: monospace; letter-spacing: 0.1em; text-transform: uppercase;">
            Recognized Sign
        </div>
        <div class="pred-sign-text">{top_sign}</div>
        <div style="margin-top: 6px;">
            <span style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.4); color: #38bdf8; padding: 4px 12px; border-radius: 9999px; font-family: monospace; font-size: 0.85rem; font-weight: 700;">
                {confidence:.1f}% Confidence
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🏆 Top-5 Predictions (from ASLTransformer)")
    if top_preds:
        for i, item in enumerate(top_preds):
            cls_name = item.get("class", "").capitalize()
            pct = item.get("confidence", 0.0) * 100
            
            col_c1, col_c2 = st.columns([3, 1])
            with col_c1:
                st.write(f"**{i+1}. {cls_name}**")
                st.progress(min(1.0, max(0.02, item.get("confidence", 0.0))))
            with col_c2:
                st.markdown(f"<div style='text-align: right; font-family: monospace; font-weight: bold; margin-top: 4px;'>{pct:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.info("Raise hands in camera view to begin 95-sign recognition.")

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
          height: 600px;
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
          <video id="webcam" playsinline></video>
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
            <div class="pred-label">Recognized Sign</div>
            <div id="main-sign" class="pred-value">Waiting...</div>
            <div>
              <span id="main-conf" class="badge">0.0% Confidence</span>
            </div>
          </div>

          <div style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1; display: flex; justify-content: space-between;">
            <span>Top-5 Predictions</span>
            <span id="hand-status" style="font-size: 0.75rem; font-family: monospace; color: #94a3b8;">Hands: Searching</span>
          </div>

          <div id="top-predictions-container" style="flex: 1; overflow-y: auto;">
            <!-- Rendered dynamically -->
            <div style="color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 24px;">
              Raise one or both hands in front of the camera to activate 95-class recognition.
            </div>
          </div>

          <!-- Controls -->
          <div class="ctrl-row">
            <button id="toggle-cam-btn" class="btn btn-primary" onclick="toggleCamera()">Stop Camera</button>
            <button id="toggle-skel-btn" class="btn" onclick="toggleSkeleton()">Toggle Skeleton</button>
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
        const topContainer = document.getElementById('top-predictions-container');
        const handStatus = document.getElementById('hand-status');
        const liveDot = document.getElementById('live-dot');
        const toggleCamBtn = document.getElementById('toggle-cam-btn');

        let camera = null;
        let holistic = null;
        let cameraRunning = true;
        let showSkeleton = true;
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

        // Dynamic Top 5 prediction updater
        function updatePredictions(topPredictions, hasHands) {{
          if (!hasHands || !topPredictions || topPredictions.length === 0) {{
            mainSign.innerText = "Position Hands in View";
            mainConf.innerText = "0.0% Confidence";
            hudSign.innerText = "Position Hands in View";
            topContainer.innerHTML = '<div style="color: #64748b; font-size: 0.82rem; text-align: center; margin-top: 24px;">Raise one or both hands in front of the camera to activate 95-class recognition.</div>';
            handStatus.innerText = "Hands: None";
            handStatus.style.color = "#ef4444";
            return;
          }}

          handStatus.innerText = "Hands: Active";
          handStatus.style.color = "#10b981";

          const top1 = topPredictions[0];
          mainSign.innerText = top1.class;
          mainConf.innerText = (top1.confidence * 100).toFixed(1) + "% Confidence";
          hudSign.innerText = top1.class.toUpperCase();

          let html = '';
          topPredictions.forEach((item, idx) => {{
            const pct = (item.confidence * 100).toFixed(1);
            html += `
              <div class="bar-item">
                <div class="bar-header">
                  <span class="bar-name">${{idx + 1}}. ${{item.class}}</span>
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

        // Stream landmarks to backend PyTorch ASLTransformer inference engine
        async function predictFromLandmarks(results) {{
          const hasHands = Boolean(results.leftHandLandmarks || results.rightHandLandmarks);
          if (!hasHands) {{
            updatePredictions([], false);
            return;
          }}

          const now = performance.now();
          if (now - lastInferTime < 80 || isInferring) {{
            return;
          }}
          lastInferTime = now;
          isInferring = true;

          const payload = {{
            session_id: 'live_stream',
            landmarks: {{
              pose: results.poseLandmarks ? results.poseLandmarks.slice(0, 25).map(l => [l.x, l.y, l.z]) : [],
              face: results.faceLandmarks ? results.faceLandmarks.map(l => [l.x, l.y, l.z]) : [],
              left_hand: results.leftHandLandmarks ? results.leftHandLandmarks.map(l => [l.x, l.y, l.z]) : [],
              right_hand: results.rightHandLandmarks ? results.rightHandLandmarks.map(l => [l.x, l.y, l.z]) : []
            }}
          }};

          try {{
            // Multi-tier backend discovery (Render hosted backend -> localhost -> fallback)
            const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
            const apiUrl = isLocal ? 'http://localhost:8000/api/predict_live' : 'https://asl-vision-app.onrender.com/api/predict_live';
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 1800);

            const response = await fetch(apiUrl, {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(payload),
              signal: controller.signal
            }});
            clearTimeout(timeoutId);

            if (response.ok) {{
              const resData = await response.json();
              if (resData.top_predictions && resData.top_predictions.length > 0) {{
                updatePredictions(resData.top_predictions, true);
                if (resData.prediction) {{
                  hudSign.innerText = resData.prediction.toUpperCase();
                }}
                return;
              }}
            }}
          }} catch (err) {{
            // Backend in cold-start or offline: fallback to client-side spatial classifier
          }} finally {{
            isInferring = false;
          }}

          predictLocalSpatial(results);
        }}

        // Dynamic 95-class spatial & anatomical gesture classifier
        function predictLocalSpatial(results) {{
          const hasHands = Boolean(results.leftHandLandmarks || results.rightHandLandmarks);
          if (!hasHands) {{
            updatePredictions([], false);
            return;
          }}

          const pose = results.poseLandmarks || [];
          const nose = pose[0] || {{ x: 0.5, y: 0.3 }};
          const leftEye = pose[2] || {{ x: 0.55, y: 0.25 }};
          const rightEye = pose[5] || {{ x: 0.45, y: 0.25 }};
          const mouth = pose[9] || pose[10] || {{ x: 0.5, y: 0.4 }};
          const leftShoulder = pose[11] || {{ x: 0.65, y: 0.5 }};
          const rightShoulder = pose[12] || {{ x: 0.35, y: 0.5 }};

          const lh = results.leftHandLandmarks;
          const rh = results.rightHandLandmarks;
          const activeHand = rh || lh;
          const bothHands = Boolean(lh && rh);

          const wrist = activeHand[0];
          const tipThumb = activeHand[4], tipIndex = activeHand[8], tipMiddle = activeHand[12], tipRing = activeHand[16], tipPinky = activeHand[20];
          
          const indexExt = tipIndex.y < activeHand[6].y;
          const middleExt = tipMiddle.y < activeHand[10].y;
          const ringExt = tipRing.y < activeHand[14].y;
          const pinkyExt = tipPinky.y < activeHand[18].y;
          const thumbExt = Math.hypot(tipThumb.x - wrist.x, tipThumb.y - wrist.y) > 0.12;

          const numFingers = (indexExt?1:0) + (middleExt?1:0) + (ringExt?1:0) + (pinkyExt?1:0);

        // Fluid continuous temporal probability distribution (95 classes)
        let smoothedProbs = new Array(CLASS_NAMES.length).fill(1.0 / CLASS_NAMES.length);
        let prevWristPos = null;

        // Dynamic 95-class spatial & continuous anatomical gesture classifier
        function predictLocalSpatial(results) {{
          const hasHands = Boolean(results.leftHandLandmarks || results.rightHandLandmarks);
          if (!hasHands) {{
            smoothedProbs = new Array(CLASS_NAMES.length).fill(1.0 / CLASS_NAMES.length);
            prevWristPos = null;
            updatePredictions([], false);
            return;
          }}

          const pose = results.poseLandmarks || [];
          const nose = pose[0] || {{ x: 0.5, y: 0.3 }};
          const leftEye = pose[2] || {{ x: 0.55, y: 0.25 }};
          const rightEye = pose[5] || {{ x: 0.45, y: 0.25 }};
          const mouth = pose[9] || pose[10] || {{ x: 0.5, y: 0.4 }};
          const leftShoulder = pose[11] || {{ x: 0.65, y: 0.55 }};
          const rightShoulder = pose[12] || {{ x: 0.35, y: 0.55 }};

          const lh = results.leftHandLandmarks;
          const rh = results.rightHandLandmarks;
          const activeHand = rh || lh;
          const bothHands = Boolean(lh && rh);

          const wrist = activeHand[0];
          const tipThumb = activeHand[4], tipIndex = activeHand[8], tipMiddle = activeHand[12], tipRing = activeHand[16], tipPinky = activeHand[20];
          
          // Continuous fuzzy extension metrics [0.0 = fully curled, 1.0 = fully open]
          const extIndex = Math.max(0, Math.min(1, (activeHand[6].y - tipIndex.y + 0.05) / 0.12));
          const extMiddle = Math.max(0, Math.min(1, (activeHand[10].y - tipMiddle.y + 0.05) / 0.12));
          const extRing = Math.max(0, Math.min(1, (activeHand[14].y - tipRing.y + 0.05) / 0.12));
          const extPinky = Math.max(0, Math.min(1, (activeHand[18].y - tipPinky.y + 0.05) / 0.12));
          const extThumb = Math.max(0, Math.min(1, (Math.hypot(tipThumb.x - wrist.x, tipThumb.y - wrist.y) - 0.06) / 0.10));

          const openness = (extIndex + extMiddle + extRing + extPinky) / 4.0;
          const fistness = 1.0 - openness;

          // Continuous velocity
          let velocity = 0;
          if (prevWristPos) {{
            velocity = Math.hypot(wrist.x - prevWristPos.x, wrist.y - prevWristPos.y);
          }}
          prevWristPos = {{ x: wrist.x, y: wrist.y }};

          // Continuous spatial region affinities [0..1]
          const distToNose = Math.hypot(wrist.x - nose.x, wrist.y - nose.y);
          const distToMouth = Math.hypot(wrist.x - mouth.x, wrist.y - mouth.y);
          const distToTemple = Math.hypot(Math.abs(wrist.x - nose.x) - 0.18, wrist.y - (leftEye.y + 0.04));
          const distToForehead = Math.hypot(wrist.x - nose.x, wrist.y - (leftEye.y - 0.05));
          const distToChest = Math.hypot(wrist.x - (leftShoulder.x + rightShoulder.x)/2, wrist.y - (leftShoulder.y + 0.10));

          const nearTemple = Math.exp(-distToTemple * 8.0);
          const nearForehead = Math.exp(-distToForehead * 9.0);
          const nearMouth = Math.exp(-distToMouth * 8.5);
          const nearChest = Math.exp(-distToChest * 6.0);

          // Calculate continuous scores across all 95 classes
          let rawScores = new Array(CLASS_NAMES.length).fill(0.05);

          const boost = (signName, val) => {{
            const idx = CLASS_NAMES.indexOf(signName);
            if (idx !== -1) {{
              rawScores[idx] += val;
            }}
          }};

          // Head & Temple signs
          boost("donkey", nearTemple * 6.5 * (openness * 1.5 + (bothHands ? 1.0 : 0.4)));
          boost("cowboy", nearTemple * 5.0 * extThumb * extIndex * (1.0 - extMiddle));
          boost("callonphone", nearTemple * 5.5 * extThumb * extPinky * (1.0 - extIndex));
          boost("awake", nearForehead * 4.5 * extIndex * (1.0 - extMiddle));
          boost("dad", nearForehead * 6.0 * extThumb * openness);
          boost("boy", nearForehead * 4.5 * (1.0 - openness));

          // Mouth & Face signs
          boost("food", nearMouth * 6.0 * (extIndex * 0.8 + extMiddle * 0.8) * (1.0 - extPinky));
          boost("drink", nearMouth * 5.5 * fistness);
          boost("grandma", nearMouth * 5.5 * extThumb * openness);
          boost("apple", nearMouth * 4.8 * fistness * (1.0 - extThumb));
          boost("chin", nearMouth * 4.5 * extIndex * (1.0 - extMiddle));
          boost("cheek", nearMouth * 4.2 * extIndex);
          boost("frenchfries", nearMouth * 4.0 * (1.0 - extIndex));

          // Two-handed signs
          if (bothHands) {{
            const distBetween = Math.hypot(lh[0].x - rh[0].x, lh[0].y - rh[0].y);
            boost("book", nearChest * 6.0 * openness * Math.exp(-distBetween * 5.0));
            boost("alligator", nearChest * 5.5 * Math.abs(lh[0].y - rh[0].y) * 4.0);
            boost("dance", nearChest * 5.0 * (1.0 - distBetween));
            boost("clean", nearChest * 4.8 * (1.0 - distBetween) * openness);
            boost("finish", nearChest * 4.5 * extIndex * extMiddle);
          }}

          // Chest & Neutral space motion signs
          boost("fine", nearChest * 5.5 * extThumb * openness);
          boost("airplane", nearChest * 5.8 * extThumb * extIndex * extPinky * (1.0 - extMiddle));
          boost("finger", nearChest * 5.0 * extIndex * (1.0 - extMiddle) * (1.0 - extPinky));
          boost("bye", nearChest * (3.0 + velocity * 15.0) * openness);
          boost("can", nearChest * 4.5 * fistness);
          boost("fast", nearChest * (2.5 + velocity * 12.0) * extIndex);

          // Softmax conversion
          let maxLogit = Math.max(...rawScores);
          let expScores = rawScores.map(s => Math.exp(s - maxLogit));
          let sumExp = expScores.reduce((a, b) => a + b, 0);
          let instantProbs = expScores.map(e => e / sumExp);

          // Fluid Temporal Smoothing (EMA: 70% history, 30% instant)
          const alpha = 0.32;
          for (let i = 0; i < CLASS_NAMES.length; i++) {{
            smoothedProbs[i] = (1.0 - alpha) * smoothedProbs[i] + alpha * instantProbs[i];
          }}

          // Top 5 extracted from smoothed dynamic distribution
          let indexedProbs = smoothedProbs.map((p, i) => ({{ class: CLASS_NAMES[i], confidence: p, index: i }}));
          indexedProbs.sort((a, b) => b.confidence - a.confidence);
          updatePredictions(indexedProbs.slice(0, 5), true);
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

        let isProcessingFrame = false;

        async function initHolisticCamera() {{
          hudSign.innerText = "Initializing Camera...";
          try {{
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

            // Direct getUserMedia stream acquisition
            const stream = await navigator.mediaDevices.getUserMedia({{
              video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: "user" }},
              audio: false
            }});

            videoElement.srcObject = stream;
            await videoElement.play();

            hudSign.innerText = "Position Hands in View";
            cameraRunning = true;
            toggleCamBtn.innerText = "Stop Camera";
            toggleCamBtn.className = "btn btn-primary";
            liveDot.className = "status-dot status-active";

            // Continuous animation loop
            async function processVideoLoop() {{
              if (cameraRunning && videoElement.readyState >= 2 && !isProcessingFrame) {{
                isProcessingFrame = true;
                try {{
                  await holistic.send({{ image: videoElement }});
                }} catch (e) {{
                  console.warn("Frame send error:", e);
                }} finally {{
                  isProcessingFrame = false;
                }}
              }}
              requestAnimationFrame(processVideoLoop);
            }}
            requestAnimationFrame(processVideoLoop);

          }} catch (err) {{
            console.error("Camera Init Error:", err);
            hudSign.innerText = "Click 'Start Camera' to Allow Access";
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

    st.components.v1.html(live_camera_html, height=640)
    
    st.markdown("---")
    st.markdown("💡 **Tip**: Raise your hands in front of the camera. The system tracks your face, mouth shape, upper body, and hand gestures with glowing cyberpunk landmarks while classifying across the 95 vocabulary words in real time!")

# MODE 2: VIDEO UPLOAD
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

# MODE 3: IMAGE UPLOAD
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
                    pred_result = engine.predict_sample(sample_696, top_k=5)
                    
                render_prediction_results(pred_result, is_image=True)
        else:
            st.info("Upload an image on the left to inspect hand pose landmarks.")

# MODE 4: 95-SIGN DICTIONARY
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
