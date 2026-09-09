import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, CameraOff, Play, Pause, RefreshCw, Eye, EyeOff, ShieldAlert, Zap, Cpu } from 'lucide-react';

// Hand connection pairs (MediaPipe 21 landmarks)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                               // Palm base
];

// Upper body pose connections (MediaPipe Pose landmarks 0..24)
const POSE_CONNECTIONS = [
  [11, 12], // Shoulders
  [11, 13], [13, 15], // Left arm
  [12, 14], [14, 16], // Right arm
  [11, 23], [12, 24], [23, 24], // Torso
  [0, 1], [1, 2], [2, 3], [3, 7], // Left eye/ear
  [0, 4], [4, 5], [5, 6], [6, 8], // Right eye/ear
  [9, 10] // Mouth
];

export default function LiveCamera({ onPredictionUpdate, onStatusChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  const [cameraActive, setCameraActive] = useState(false);
  const [recognitionPaused, setRecognitionPaused] = useState(false);
  const [isMirrored, setIsMirrored] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [handDetected, setHandDetected] = useState(false);
  const [fps, setFps] = useState(0);
  const [cameraError, setCameraError] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);

  const holisticRef = useRef(null);
  const cameraRef = useRef(null);
  const wsRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());
  const recognitionPausedRef = useRef(recognitionPaused);

  useEffect(() => {
    recognitionPausedRef.current = recognitionPaused;
  }, [recognitionPaused]);

  // Connect WebSocket to backend
  const initWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/live`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (onPredictionUpdate) {
            onPredictionUpdate(data);
          }
          if (data.hand_detected !== undefined) {
            setHandDetected(data.hand_detected);
          }
        } catch (err) {
          console.error('Error parsing prediction from WebSocket:', err);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        // Attempt reconnect after 2 seconds if camera is active
        setTimeout(() => {
          if (cameraActive) initWebSocket();
        }, 2000);
      };

      ws.onerror = (err) => {
        console.warn('WebSocket connection error:', err);
      };
    } catch (e) {
      console.error('WebSocket initialization error:', e);
    }
  }, [cameraActive, onPredictionUpdate]);

  useEffect(() => {
    initWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [initWebSocket]);

  // Draw cyber-aesthetic skeleton on canvas
  const drawLandmarksOverlay = (results) => {
    const canvas = canvasRef.current;
    if (!canvas || !videoRef.current) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (!showSkeleton) return;

    // Helper: calculate transformed coordinates (with mirror support)
    const getPoint = (lm) => {
      if (!lm) return null;
      const x = isMirrored ? (1.0 - lm.x) * width : lm.x * width;
      const y = lm.y * height;
      return { x, y };
    };

    // Draw Pose Skeleton (Subtle blue-cyan)
    if (results.poseLandmarks) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
      ctx.shadowBlur = 6;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.6)';

      for (const [i, j] of POSE_CONNECTIONS) {
        const p1 = getPoint(results.poseLandmarks[i]);
        const p2 = getPoint(results.poseLandmarks[j]);
        if (p1 && p2 && (results.poseLandmarks[i].visibility === undefined || results.poseLandmarks[i].visibility > 0.4)) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // Draw Pose joints
      for (let i = 0; i <= 24; i++) {
        const p = getPoint(results.poseLandmarks[i]);
        if (p && (results.poseLandmarks[i].visibility === undefined || results.poseLandmarks[i].visibility > 0.4)) {
          ctx.fillStyle = '#06b6d4';
          ctx.beginPath();
          ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // Helper to draw a single hand with cyberpunk aesthetic
    const drawHand = (handLandmarks, primaryColor, glowColor) => {
      if (!handLandmarks) return;

      // Draw bones
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = primaryColor;
      ctx.shadowBlur = 10;
      ctx.shadowColor = glowColor;

      for (const [i, j] of HAND_CONNECTIONS) {
        const p1 = getPoint(handLandmarks[i]);
        const p2 = getPoint(handLandmarks[j]);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // Draw glowing joints
      for (let i = 0; i < handLandmarks.length; i++) {
        const p = getPoint(handLandmarks[i]);
        if (p) {
          const isTip = [4, 8, 12, 16, 20].includes(i);
          const isWrist = i === 0;
          
          ctx.fillStyle = isTip ? '#ffffff' : primaryColor;
          ctx.shadowBlur = isTip ? 14 : 8;
          ctx.shadowColor = '#ffffff';
          
          ctx.beginPath();
          ctx.arc(p.x, p.y, isTip ? 5 : (isWrist ? 6 : 3.5), 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    };

    // Left Hand (Cyan glow)
    if (results.leftHandLandmarks) {
      drawHand(results.leftHandLandmarks, '#06b6d4', 'rgba(6, 182, 212, 0.9)');
    }

    // Right Hand (Violet glow)
    if (results.rightHandLandmarks) {
      drawHand(results.rightHandLandmarks, '#a855f7', 'rgba(168, 85, 247, 0.9)');
    }

    // Reset shadow
    ctx.shadowBlur = 0;
  };

  // MediaPipe Results Callback
  const onResults = (results) => {
    // Calculate FPS
    frameCountRef.current += 1;
    const now = performance.now();
    if (now - lastFpsTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;
    }

    const hasHands = Boolean(results.leftHandLandmarks || results.rightHandLandmarks);
    setHandDetected(hasHands);

    // Draw visual landmark overlay
    drawLandmarksOverlay(results);

    // Stream landmarks to backend WebSocket if recognition is active
    if (!recognitionPausedRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const payload = {
        type: 'landmarks',
        landmarks: {
          pose: results.poseLandmarks ? results.poseLandmarks.slice(0, 25).map(l => [l.x, l.y, l.z]) : [],
          face: results.faceLandmarks ? results.faceLandmarks.map(l => [l.x, l.y, l.z]) : [],
          left_hand: results.leftHandLandmarks ? results.leftHandLandmarks.map(l => [l.x, l.y, l.z]) : [],
          right_hand: results.rightHandLandmarks ? results.rightHandLandmarks.map(l => [l.x, l.y, l.z]) : []
        }
      };
      wsRef.current.send(JSON.stringify(payload));
    }
  };

  // Start Webcam & MediaPipe
  const startCamera = async () => {
    setCameraError(null);
    try {
      if (!window.Holistic) {
        throw new Error("MediaPipe Holistic library is still initializing. Please wait a moment and try again.");
      }

      const holistic = new window.Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        refineFaceLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      holistic.onResults(onResults);
      holisticRef.current = holistic;

      const videoElement = videoRef.current;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });

      videoElement.srcObject = stream;
      await videoElement.play();

      if (window.Camera) {
        const camera = new window.Camera(videoElement, {
          onFrame: async () => {
            if (videoRef.current && videoRef.current.videoWidth) {
              const canvas = canvasRef.current;
              if (canvas && (canvas.width !== videoRef.current.videoWidth || canvas.height !== videoRef.current.videoHeight)) {
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
              }
              await holistic.send({ image: videoRef.current });
            }
          },
          width: 1280,
          height: 720
        });
        camera.start();
        cameraRef.current = camera;
      }

      setCameraActive(true);
      if (onStatusChange) onStatusChange('active');
    } catch (err) {
      console.error('Failed to start camera:', err);
      setCameraError(err.message || "Failed to access webcam. Please ensure camera permissions are allowed.");
      setCameraActive(false);
      if (onStatusChange) onStatusChange('error');
    }
  };

  // Stop Webcam
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (cameraRef.current) {
      try { cameraRef.current.stop(); } catch (e) {}
      cameraRef.current = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setCameraActive(false);
    setHandDetected(false);
    setFps(0);
    if (onStatusChange) onStatusChange('stopped');
  };

  // Auto-start camera when component mounts
  useEffect(() => {
    const timer = setTimeout(() => {
      startCamera();
    }, 500);

    return () => {
      clearTimeout(timer);
      stopCamera();
    };
  }, []);

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Video / Canvas Viewport */}
      <div className="relative w-full aspect-video bg-[#0a0d14] rounded-2xl overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center group">
        {/* Actual Webcam Video */}
        <video
          ref={videoRef}
          className={`w-full h-full object-cover ${isMirrored ? 'mirror-mode' : ''} ${!cameraActive ? 'hidden' : 'block'}`}
          playsInline
          muted
        />

        {/* 60 FPS HTML5 Canvas Overlay for Landmarks */}
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full pointer-events-none ${!cameraActive ? 'hidden' : 'block'}`}
        />

        {/* Camera Off / Loading State */}
        {!cameraActive && (
          <div className="text-center p-6 max-w-md">
            {cameraError ? (
              <div className="flex flex-col items-center gap-3 text-rose-400">
                <ShieldAlert className="w-12 h-12 stroke-[1.5]" />
                <h3 className="font-semibold text-base">Camera Access Error</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{cameraError}</p>
                <button
                  onClick={startCamera}
                  className="mt-2 px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 rounded-xl text-xs font-semibold transition-all"
                >
                  Retry Permission
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 text-slate-400">
                <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center animate-pulse">
                  <Camera className="w-6 h-6 text-cyan-400" />
                </div>
                <h3 className="font-semibold text-slate-200">Initializing Live ASL Stream</h3>
                <p className="text-xs text-slate-500">Requesting camera access & loading MediaPipe AI engine...</p>
                <button
                  onClick={startCamera}
                  className="mt-2 px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-xl text-xs font-bold shadow-lg shadow-cyan-500/20 hover:opacity-90 transition-all"
                >
                  Start Camera Feed
                </button>
              </div>
            )}
          </div>
        )}

        {/* Live HUD Badges & Indicators */}
        {cameraActive && (
          <>
            {/* Top Left: Status Indicators */}
            <div className="absolute top-3 left-3 flex flex-wrap items-center gap-2 pointer-events-none">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-slate-950/80 backdrop-blur-md border border-slate-800 text-slate-200 shadow-md">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                LIVE
              </span>

              <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium backdrop-blur-md border shadow-md transition-all ${
                handDetected
                  ? 'bg-emerald-950/80 border-emerald-500/40 text-emerald-300'
                  : 'bg-amber-950/80 border-amber-500/40 text-amber-300'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${handDetected ? 'bg-emerald-400' : 'bg-amber-400 animate-ping'}`} />
                {handDetected ? 'Hands Detected' : 'No Hand in View'}
              </span>

              {recognitionPaused && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-amber-500/20 border border-amber-500/40 text-amber-300 backdrop-blur-md">
                  <Pause className="w-3 h-3" /> PAUSED
                </span>
              )}
            </div>

            {/* Top Right: FPS & Engine Stats */}
            <div className="absolute top-3 right-3 flex items-center gap-2 pointer-events-none">
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-slate-950/80 backdrop-blur-md border border-slate-800 text-cyan-400 shadow-md">
                {fps} FPS
              </span>
              <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-cyan-400' : 'bg-rose-500'}`} title={wsConnected ? "WebSocket Connected" : "WebSocket Disconnected"} />
            </div>
          </>
        )}
      </div>

      {/* Camera Action Toolbar */}
      <div className="bg-[#121722]/90 border border-slate-800/80 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 shadow-lg">
        {/* Primary Controls */}
        <div className="flex items-center gap-2">
          {cameraActive ? (
            <button
              onClick={stopCamera}
              className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 border border-rose-500/30 transition-all shadow-sm"
            >
              <CameraOff className="w-4 h-4" />
              Stop Camera
            </button>
          ) : (
            <button
              onClick={startCamera}
              className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20 hover:opacity-90 transition-all"
            >
              <Camera className="w-4 h-4" />
              Start Camera
            </button>
          )}

          {cameraActive && (
            <button
              onClick={() => setRecognitionPaused(!recognitionPaused)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all ${
                recognitionPaused
                  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/25'
                  : 'bg-amber-500/15 text-amber-300 border-amber-500/30 hover:bg-amber-500/25'
              }`}
            >
              {recognitionPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
              {recognitionPaused ? 'Resume Recognition' : 'Pause Recognition'}
            </button>
          )}
        </div>

        {/* View Options */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsMirrored(!isMirrored)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              isMirrored
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                : 'bg-slate-800/80 border-slate-700 text-slate-400'
            }`}
            title="Toggle video mirror effect"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Mirror {isMirrored ? 'On' : 'Off'}
          </button>

          <button
            onClick={() => setShowSkeleton(!showSkeleton)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              showSkeleton
                ? 'bg-violet-500/10 border-violet-500/30 text-violet-300'
                : 'bg-slate-800/80 border-slate-700 text-slate-400'
            }`}
            title="Toggle hand & body skeleton overlay"
          >
            {showSkeleton ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            Landmarks
          </button>
        </div>
      </div>
    </div>
  );
}
