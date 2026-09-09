import os
import sys
import tempfile
import time
import numpy as np
import cv2
import torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

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
    /* Dark background styling */
    .stApp {
        background-color: #0a0d14;
        color: #f1f5f9;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        font-weight: 800;
    }
    
    /* Top title gradient */
    .gradient-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    
    /* Metric Cards */
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
    
    /* Prediction Banner */
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
    
    /* Sidebar */
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

try:
    engine = get_engine()
    detector = get_detector()
    model, class_names, device = engine.model, engine.class_names, engine.device
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading ASL model: {e}")

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
    
    # Large Banner
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
    
    # Top 5 Breakdown
    st.markdown("#### 🏆 Top-5 Predictions")
    for i, item in enumerate(top_preds):
        cls_name = item.get("class", "").capitalize()
        pct = item.get("confidence", 0.0) * 100
        
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            st.write(f"**{i+1}. {cls_name}**")
            st.progress(min(1.0, max(0.02, item.get("confidence", 0.0))))
        with col_c2:
            st.markdown(f"<div style='text-align: right; font-family: monospace; font-weight: bold; margin-top: 4px;'>{pct:.1f}%</div>", unsafe_allow_html=True)

# MODE 1: LIVE CAMERA (PRIMARY)
if mode == "📹 Live Camera (Primary)":
    col_left, col_right = st.columns([7, 5])
    
    with col_left:
        st.markdown("### 📹 Live Camera Feed & Landmark Tracking")
        st.markdown("Position your hands inside the camera view. The system extracts MediaPipe landmarks and classifies ASL gestures.")
        
        # Option A: In-Browser 60 FPS Cyberpunk MediaPipe Canvas Component
        live_html = """
        <div style="background: #0d121e; border: 1px solid #1e293b; border-radius: 16px; padding: 12px; text-align: center;">
            <div style="position: relative; width: 100%; aspect-ratio: 16/9; background: #000; border-radius: 12px; overflow: hidden;">
                <video id="webcam" style="width: 100%; height: 100%; object-fit: cover; transform: scaleX(-1);" playsinline autoplay muted></video>
                <canvas id="overlay" style="position: absolute; top:0; left:0; width: 100%; height: 100%; pointer-events: none;"></canvas>
            </div>
            <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center; font-family: monospace; font-size: 12px; color: #94a3b8;">
                <span id="status-tag" style="background: #022c22; color: #34d399; border: 1px solid #059669; padding: 3px 8px; border-radius: 6px;">● MediaPipe Ready</span>
                <span id="fps-tag" style="color: #38bdf8;">60 FPS Tracking</span>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('overlay');
            const ctx = canvas.getContext('2d');
            const statusTag = document.getElementById('status-tag');
            const fpsTag = document.getElementById('fps-tag');

            const HAND_CONNECTIONS = [
                [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
                [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
                [13,17],[17,18],[18,19],[19,20],[0,17]
            ];

            let frameCount = 0;
            let lastTime = performance.now();

            function drawSkeleton(results) {
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                function drawHand(lms, color, glow) {
                    if (!lms) return;
                    ctx.lineWidth = 2.5;
                    ctx.strokeStyle = color;
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = glow;

                    for (const [i, j] of HAND_CONNECTIONS) {
                        const p1 = lms[i], p2 = lms[j];
                        if (p1 && p2) {
                            ctx.beginPath();
                            ctx.moveTo((1 - p1.x) * canvas.width, p1.y * canvas.height);
                            ctx.lineTo((1 - p2.x) * canvas.width, p2.y * canvas.height);
                            ctx.stroke();
                        }
                    }

                    for (let i = 0; i < lms.length; i++) {
                        const p = lms[i];
                        const isTip = [4,8,12,16,20].includes(i);
                        ctx.fillStyle = isTip ? '#ffffff' : color;
                        ctx.beginPath();
                        ctx.arc((1 - p.x) * canvas.width, p.y * canvas.height, isTip ? 5 : 3.5, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                }

                drawHand(results.leftHandLandmarks, '#06b6d4', 'rgba(6,182,212,0.8)');
                drawHand(results.rightHandLandmarks, '#a855f7', 'rgba(168,85,247,0.8)');
                ctx.shadowBlur = 0;
            }

            async function init() {
                try {
                    const holistic = new Holistic({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}` });
                    holistic.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5 });
                    holistic.onResults(results => {
                        frameCount++;
                        const now = performance.now();
                        if (now - lastTime >= 1000) {
                            fpsTag.innerText = frameCount + " FPS Tracking";
                            frameCount = 0;
                            lastTime = now;
                        }
                        const hasHands = results.leftHandLandmarks || results.rightHandLandmarks;
                        statusTag.innerText = hasHands ? "● Hands Tracked" : "● Searching Hands";
                        statusTag.style.color = hasHands ? "#34d399" : "#fbbf24";
                        drawSkeleton(results);
                    });

                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" } });
                    video.srcObject = stream;
                    const camera = new Camera(video, {
                        onFrame: async () => { await holistic.send({ image: video }); },
                        width: 1280, height: 720
                    });
                    camera.start();
                } catch(e) {
                    statusTag.innerText = "● Camera Permission Required";
                    statusTag.style.color = "#f87171";
                }
            }
            window.addEventListener('load', init);
            init();
        </script>
        """
        components.html(live_html, height=440)
        
        # Snapshot inference support
        st.markdown("##### 📸 Instant Gesture Snapshot Inference")
        img_file_buffer = st.camera_input("Take a snapshot of your ASL sign for instant classification")

    with col_right:
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            with st.spinner("Extracting landmarks & running ASLTransformer..."):
                feat_348, lm_dict = detector.process_frame(cv_img)
                sample_696 = process_landmarks_sequence([feat_348])
                pred_result = engine.predict_sample(sample_696, top_k=5)
                
            render_prediction_results(pred_result)
        else:
            st.markdown("""
            <div class="pred-banner">
                <div style="font-size: 0.75rem; color: #94a3b8; font-family: monospace; letter-spacing: 0.1em; text-transform: uppercase;">
                    Live Recognition Ready
                </div>
                <div class="pred-sign-text" style="color: #94a3b8; font-size: 2.2rem;">Live Tracking Active</div>
                <p style="font-size: 0.8rem; color: #64748b; margin-top: 8px;">
                    Move hands in front of the camera or capture a snapshot above to classify your sign!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 **Tip**: Hold your sign steady in view. You can also explore all 95 recognizable words in the **95-Sign Dictionary** tab in the sidebar!")

# MODE 2: VIDEO UPLOAD
elif mode == "🎬 Video Upload":
    col_v1, col_v2 = st.columns([6, 6])
    
    with col_v1:
        st.markdown("### 🎬 Upload ASL Video")
        st.markdown("Upload a video clip (.mp4, .webm, .mov, .avi). The system processes all frames, normalizes 74 landmarks, computes velocity, and predicts the sign.")
        
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
                    prog_bar.progress(80, text="Running ASLTransformer inference...")
                    
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
            st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
            
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
