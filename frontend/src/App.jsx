import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import LiveCamera from './components/LiveCamera';
import VideoUpload from './components/VideoUpload';
import ImageUpload from './components/ImageUpload';
import PredictionPanel from './components/PredictionPanel';
import SignDictionary from './components/SignDictionary';
import { Cpu, Zap, Activity, Layers, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function App() {
  const [currentMode, setMode] = useState('live'); // 'live' | 'video' | 'image'
  const [modelInfo, setModelInfo] = useState(null);
  const [dictionaryOpen, setDictionaryOpen] = useState(false);

  // Live prediction state
  const [livePrediction, setLivePrediction] = useState(null);
  const [videoPrediction, setVideoPrediction] = useState(null);
  const [imagePrediction, setImagePrediction] = useState(null);
  const [cameraStatus, setCameraStatus] = useState('ready');

  // Fetch model metadata on mount
  useEffect(() => {
    fetch('/api/info')
      .then((res) => res.json())
      .then((data) => {
        setModelInfo(data);
      })
      .catch((err) => {
        console.warn('Could not fetch /api/info:', err);
      });
  }, []);

  // Determine current active prediction to display in the side panel
  const activePrediction =
    currentMode === 'live'
      ? livePrediction
      : currentMode === 'video'
      ? videoPrediction
      : imagePrediction;

  return (
    <div className="min-h-screen bg-[#0a0d14] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white">
      {/* Top Navigation */}
      <Header
        currentMode={currentMode}
        setMode={setMode}
        onOpenDictionary={() => setDictionaryOpen(true)}
        modelInfo={modelInfo}
      />

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-6">
        {/* Dual Column Layout: Video/Camera Viewport + Live Prediction Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Main Visualizer Area (7 cols on desktop) */}
          <div className="lg:col-span-7 flex flex-col justify-between">
            {currentMode === 'live' && (
              <LiveCamera
                onPredictionUpdate={(pred) => setLivePrediction(pred)}
                onStatusChange={(status) => setCameraStatus(status)}
              />
            )}

            {currentMode === 'video' && (
              <VideoUpload
                onPredictionComplete={(pred) => setVideoPrediction(pred)}
              />
            )}

            {currentMode === 'image' && (
              <ImageUpload
                onPredictionComplete={(pred) => setImagePrediction(pred)}
              />
            )}
          </div>

          {/* Real-Time Prediction & Top-5 Rankings Panel (5 cols on desktop) */}
          <div className="lg:col-span-5">
            <PredictionPanel
              predictionResult={activePrediction}
              bufferFill={livePrediction?.buffer_fill ?? 0}
              bufferTarget={livePrediction?.buffer_target ?? 64}
              handDetected={livePrediction?.hand_detected ?? false}
              isLive={currentMode === 'live'}
              status={cameraStatus}
            />
          </div>
        </div>

        {/* Technical Architecture & Specs Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {/* Card 1: Model Architecture */}
          <div className="bg-[#121722]/80 border border-slate-800/80 rounded-xl p-4 flex items-start gap-3.5 shadow-lg">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 shrink-0">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-xs font-mono font-bold tracking-wide uppercase text-slate-300">ASLTransformer</h3>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                4-layer TransformerEncoder with GELU activation, learned CLS token, and 2.31M parameters.
              </p>
              <div className="mt-2 flex items-center gap-2 text-[11px] font-mono text-cyan-400 font-semibold">
                <span>D_MODEL: 256</span> &bull; <span>4 Heads</span> &bull; <span>FFN: 512</span>
              </div>
            </div>
          </div>

          {/* Card 2: Landmark & Velocity Pipeline */}
          <div className="bg-[#121722]/80 border border-slate-800/80 rounded-xl p-4 flex items-start gap-3.5 shadow-lg">
            <div className="p-2.5 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400 shrink-0">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-xs font-mono font-bold tracking-wide uppercase text-slate-300">Temporal & Velocity</h3>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                MediaPipe Holistic extracts 74 normalized landmarks (348 dims) + frame-to-frame velocity for 696-dim input.
              </p>
              <div className="mt-2 flex items-center gap-2 text-[11px] font-mono text-violet-400 font-semibold">
                <span>64 Frames</span> &bull; <span>696 Features</span> &bull; <span>Body Normalized</span>
              </div>
            </div>
          </div>

          {/* Card 3: Evaluation & Accuracy */}
          <div className="bg-[#121722]/80 border border-slate-800/80 rounded-xl p-4 flex items-start gap-3.5 shadow-lg">
            <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-xs font-mono font-bold tracking-wide uppercase text-slate-300">Validation Performance</h3>
              <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                Verified 75.72% test accuracy evaluated across 95 isolated American Sign Language vocabulary gestures.
              </p>
              <div className="mt-2 flex items-center gap-2 text-[11px] font-mono text-emerald-400 font-semibold">
                <span>95 Classes</span> &bull; <span>Epoch 51 Best</span> &bull; <span>75.72% Test Acc</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* 95 ASL Vocabulary Directory Modal */}
      <SignDictionary
        isOpen={dictionaryOpen}
        onClose={() => setDictionaryOpen(false)}
        classes={modelInfo?.classes || []}
      />
    </div>
  );
}
