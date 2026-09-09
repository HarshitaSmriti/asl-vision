# ASL Vision — Real-Time Sign Language Recognition

A real-time American Sign Language (ASL) computer-vision application powered by the trained **ASLTransformer** model checkpoint (`ASL_95class_75_72pct_TEST_best.pth`).

---

## Key Highlights

- **Primary Mode: Live Webcam Recognition**: Instant real-time video tracking with MediaPipe hand & pose landmark tracking rendered at 60 FPS on HTML5 Canvas.
- **Cyberpunk AI Skeleton Overlay**: Glowing keypoints and skeletal lines moving seamlessly with the user's hands.
- **Trained ASLTransformer**: Exact 4-layer Transformer with 4 attention heads, learned CLS token, GELU activation, and 95 output classes (75.72% test accuracy).
- **Temporal & Velocity Pipeline**: 74 MediaPipe landmarks (25 pose, 7 face, 21 left hand, 21 right hand) normalized for body scale & wrist local frames (348 base features) concatenated with frame-to-frame velocity for a 696-dimensional representation across a 64-frame rolling temporal window.
- **Multi-Modal Recognition**: Supports **Live Camera**, **Video Upload** (.mp4, .webm, .mov, .avi), and **Image Upload** (with transparent pose-based disclaimer).
- **95 ASL Vocabulary Explorer**: Integrated searchable directory of all 95 recognizable ASL signs.

---

## Project Structure

```
d:/asl_model/
├── model/
│   └── ASL_95class_75_72pct_TEST_best.pth   # Provided PyTorch checkpoint
│
├── backend/
│   ├── model.py                            # Exact ASLTransformer & PositionalEncoding
│   ├── preprocessing.py                    # Landmark normalization, 64-frame resampling, velocity
│   ├── landmark_detector.py                # MediaPipe Holistic detector
│   ├── inference.py                        # Temporal inference engine & rolling live predictor
│   ├── main.py                             # FastAPI server + WebSocket /ws/live + Static SPA
│   ├── requirements.txt                    # Python dependencies
│   ├── test_pipeline.py                    # Architecture & preprocessing unit tests
│   └── test_e2e.py                         # End-to-end WebSocket & REST tests
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx                  # Mode switcher & dictionary button
│   │   │   ├── LiveCamera.jsx              # Webcam stream + 60fps canvas skeleton overlay
│   │   │   ├── PredictionPanel.jsx         # Sign name, confidence gauge, Top-5 chart
│   │   │   ├── VideoUpload.jsx             # Video file uploader
│   │   │   ├── ImageUpload.jsx             # Single image uploader + disclaimer
│   │   │   └── SignDictionary.jsx          # 95 ASL vocabulary browser modal
│   │   ├── App.jsx                         # Main app container
│   │   ├── index.css                       # Tailwind styles & glow animations
│   │   └── main.jsx                        # React entrypoint
│   ├── dist/                               # Production build
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
└── README.md
```

---

## Getting Started

### 1. Run the Backend & Application

The FastAPI backend automatically serves both the API endpoints and the built frontend application at `http://localhost:8000`:

```bash
python backend/main.py
```

Then open your browser and navigate to:
**`http://localhost:8000`**

### 2. (Optional) Run Frontend in Vite Hot-Reload Dev Mode

If you are developing or modifying UI components:

```bash
cd frontend
npm run dev
```

Navigate to **`http://localhost:5173`** (API requests & WebSockets are proxied to `:8000`).

---

## Running Verification Tests

```bash
# Unit test for model architecture, forward pass, and velocity features
python backend/test_pipeline.py

# End-to-end test for WebSocket live streaming, video, and image endpoints
python backend/test_e2e.py
```

---

## Preprocessing & Velocity Pipeline

1. **Landmark Extraction**: 74 landmarks via MediaPipe Holistic (25 Pose, 7 Face, 21 Left Hand, 21 Right Hand).
2. **Body Normalization**: Landmarks normalized relative to shoulder midpoint and shoulder width distance ($74 \times 3 = 222$ dims).
3. **Hand-Local Normalization**: Left and right hand coordinates normalized relative to each wrist and hand bounding scale ($21 \times 3 + 21 \times 3 = 126$ dims). Total base features: $222 + 126 = 348$.
4. **Temporal Resampling**: Resampled to a 64-frame sequence ($64 \times 348$).
5. **Velocity Concatenation**:
   $$v_0 = \mathbf{0}_{348}, \quad v_t = x_t - x_{t-1}$$
   $$\text{Final Input} = [x_t, v_t] \in \mathbb{R}^{64 \times 696}$$
6. **Transformer Forward Pass**: Output shape $(1, 95)$ followed by Softmax Top-5 ranking.
