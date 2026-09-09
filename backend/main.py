import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import tempfile
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, List

from backend.model import load_asl_model
from backend.preprocessing import (
    process_client_landmarks_dict,
    process_landmarks_sequence,
    resample_sequence,
    compute_velocity_features,
    BASE_FEATURE_DIM,
    FINAL_FEATURE_DIM,
    SEQUENCE_LENGTH
)
from backend.landmark_detector import HolisticLandmarkDetector
from backend.inference import ASLInferenceEngine, RollingLivePredictor

from contextlib import asynccontextmanager

# Global instances loaded once on startup
inference_engine: ASLInferenceEngine = None
video_detector: HolisticLandmarkDetector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_engine, video_detector
    print("Starting ASL Vision backend...")
    inference_engine = ASLInferenceEngine()
    video_detector = HolisticLandmarkDetector()
    print("ASL Vision backend ready.")
    yield
    if video_detector:
        video_detector.close()

app = FastAPI(
    title="ASL Vision AI API",
    description="Real-time American Sign Language recognition powered by ASLTransformer (95 classes, 75.72% accuracy).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.head("/health")
def health_check():
    """Health probe endpoint for Render."""
    return {"status": "ok", "service": "asl-vision-backend"}

@app.get("/api/info")
def get_info():
    """Returns model metadata, class vocabulary, and device information."""
    model, class_names, device = load_asl_model()
    return {
        "model_name": "ASLTransformer",
        "num_classes": len(class_names),
        "test_accuracy": "75.72%",
        "best_epoch": 51,
        "input_dim": 696,
        "base_features": 348,
        "sequence_length": 64,
        "device": device.type.upper(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "classes": [class_names[i] for i in sorted(class_names.keys())]
    }

# Session dictionary for HTTP-based live streaming
live_http_predictors: Dict[str, RollingLivePredictor] = {}

@app.post("/api/predict_live")
async def predict_live_step(data: Dict[str, Any]):
    """
    Accepts client landmarks dictionary from browser/Streamlit,
    processes 348-dim features through rolling temporal predictor,
    and returns real-time PyTorch ASLTransformer predictions.
    """
    session_id = data.get("session_id", "default")
    if session_id not in live_http_predictors:
        live_http_predictors[session_id] = RollingLivePredictor(buffer_size=64, step_size=2, min_frames=16)

    predictor = live_http_predictors[session_id]
    
    if data.get("action") == "reset":
        predictor.reset()
        return {"status": "reset"}

    raw_landmarks = data.get("landmarks", {})
    lh_present = len(raw_landmarks.get("left_hand", [])) > 0
    rh_present = len(raw_landmarks.get("right_hand", [])) > 0

    if not (lh_present or rh_present):
        return {
            "success": True,
            "prediction": "Position hands in view",
            "is_confident": False,
            "confidence": 0.0,
            "top_predictions": [],
            "hand_detected": False
        }

    frame_features = process_client_landmarks_dict(raw_landmarks)
    pred_result = predictor.add_frame(frame_features)
    
    if pred_result is None:
        pred_result = {
            "success": True,
            "prediction": "Buffering motion...",
            "confidence": 0.0,
            "top_predictions": [],
            "hand_detected": True
        }
    else:
        pred_result["hand_detected"] = True

    return pred_result

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Accepts video file (.mp4, .webm, .mov, .avi), extracts landmarks using MediaPipe,
    resamples to 64 frames, computes velocity features (696 dims), and predicts ASL sign.
    """
    if not file.filename.lower().endswith(('.mp4', '.webm', '.mov', '.avi', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format. Please upload MP4, WebM, MOV, or AVI.")

    # Save to temp file for OpenCV decoding
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        sample_696, landmarks_timeline, fps = video_detector.process_video_path(tmp_path)
        result = inference_engine.predict_sample(sample_696, top_k=5)
        result["frames_processed"] = len(landmarks_timeline)
        result["fps"] = fps
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts single image (.jpg, .jpeg, .png, .webp).
    Extracts MediaPipe landmarks and creates an honest stationary 64-frame sequence.
    Returns approximate pose-based prediction with clear disclaimer.
    """
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        raise HTTPException(status_code=400, detail="Unsupported image format. Please upload JPG, PNG, or WEBP.")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    try:
        feat_348, landmarks_dict = video_detector.process_frame(image)
        
        # Check if hands were detected
        has_hands = landmarks_dict.get("left_hand_present", False) or landmarks_dict.get("right_hand_present", False)
        
        # Construct stationary 64-frame sequence
        frames_list = [feat_348]
        sample_696 = process_landmarks_sequence(frames_list)
        
        result = inference_engine.predict_sample(sample_696, top_k=5)
        result["has_hands"] = has_hands
        result["landmarks"] = landmarks_dict
        result["disclaimer"] = (
            "Single Image mode provides an approximate pose-based prediction. "
            "Since ASL gestures involve temporal motion across 64 frames, for full 75.72% test accuracy, use Live Camera or Video."
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    High-performance bidirectional WebSocket endpoint for live camera recognition.
    Accepts landmark payloads from the browser, maintains rolling 64-frame buffer,
    and returns predictions in real-time.
    """
    await websocket.accept()
    predictor = RollingLivePredictor(buffer_size=64, step_size=2, min_frames=16)

    try:
        while True:
            text_data = await websocket.receive_text()
            data = json.loads(text_data)

            msg_type = data.get("type", "landmarks")

            if msg_type == "reset":
                predictor.reset()
                await websocket.send_text(json.dumps({"status": "reset_complete"}))
                continue

            if msg_type == "landmarks":
                # Extract client landmarks and compute 348-dim feature
                frame_features = process_client_landmarks_dict(data.get("landmarks", {}))
                
                # Check hand presence
                lh_present = len(data.get("landmarks", {}).get("left_hand", [])) > 0
                rh_present = len(data.get("landmarks", {}).get("right_hand", [])) > 0

                pred_result = predictor.add_frame(frame_features)
                if pred_result:
                    pred_result["hand_detected"] = lh_present or rh_present
                    pred_result["left_hand_detected"] = lh_present
                    pred_result["right_hand_detected"] = rh_present
                    await websocket.send_text(json.dumps(pred_result))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

# Mount frontend build if available
frontend_dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist_dir):
    assets_dir = os.path.join(frontend_dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(frontend_dist_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
