import React from 'react';
import { Award, Zap, Activity, Info, CheckCircle2, AlertCircle } from 'lucide-react';

export default function PredictionPanel({
  predictionResult,
  bufferFill = 0,
  bufferTarget = 64,
  handDetected = false,
  isLive = true,
  status = 'ready'
}) {
  const hasHands = handDetected || (predictionResult?.hand_detected ?? false);
  const currentSign = hasHands ? (predictionResult?.prediction || 'Detecting sign...') : 'Position hands in view';
  const confidence = hasHands && predictionResult?.confidence ? (predictionResult.confidence * 100).toFixed(1) : '0.0';
  const isConfident = hasHands && (predictionResult?.is_confident ?? false);
  const topPredictions = hasHands ? (predictionResult?.top_predictions || []) : [];

  // Color mapping based on confidence
  const confNum = parseFloat(confidence);
  let badgeColor = 'bg-slate-800 text-slate-400 border-slate-700';
  let progressColor = 'from-slate-600 to-slate-500';
  if (confNum >= 70) {
    badgeColor = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
    progressColor = 'from-emerald-500 to-cyan-500';
  } else if (confNum >= 35) {
    badgeColor = 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30';
    progressColor = 'from-cyan-500 to-blue-500';
  } else if (confNum > 0) {
    badgeColor = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    progressColor = 'from-amber-500 to-orange-500';
  }

  return (
    <div className="bg-[#121722]/90 border border-slate-800/90 rounded-2xl p-5 shadow-xl flex flex-col justify-between h-full backdrop-blur-sm">
      {/* Top Banner: Primary Detected Sign */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-mono font-semibold tracking-wider text-slate-400 uppercase flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            Current Sign Recognition
          </span>
          {isLive && (
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${handDetected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
              <span className="text-[11px] font-mono text-slate-400">
                {handDetected ? 'Hands Tracked' : 'Searching Hands'}
              </span>
            </div>
          )}
        </div>

        {/* Large Prominent Prediction Display */}
        <div className="bg-[#0a0d14]/80 border border-slate-800/80 rounded-xl p-5 text-center relative overflow-hidden group">
          {/* Subtle background glow */}
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-violet-500/5 to-transparent pointer-events-none" />
          
          <div className="relative z-10">
            <div className="text-3xl sm:text-4xl lg:text-5xl font-black tracking-tight text-white capitalize drop-shadow-md transition-all duration-200">
              {currentSign}
            </div>

            <div className="mt-3 flex items-center justify-center gap-2">
              <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-mono font-bold border ${badgeColor}`}>
                <Activity className="w-3 h-3" />
                {confidence}% Confidence
              </span>
            </div>
          </div>
        </div>

        {/* Temporal Rolling Buffer Progress (for live mode) */}
        {isLive && (
          <div className="mt-3 bg-slate-900/60 border border-slate-800/60 rounded-lg p-2.5">
            <div className="flex justify-between items-center text-[11px] font-mono text-slate-400 mb-1">
              <span>Rolling Temporal Window</span>
              <span className="text-cyan-400 font-semibold">{bufferFill} / {bufferTarget} frames</span>
            </div>
            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-gradient-to-r from-cyan-500 to-violet-500 h-full transition-all duration-150 rounded-full"
                style={{ width: `${Math.min(100, (bufferFill / bufferTarget) * 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Top 5 Predictions Section */}
      <div className="mt-5">
        <div className="flex items-center justify-between mb-2.5">
          <span className="text-xs font-mono font-semibold tracking-wider text-slate-400 uppercase flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5 text-violet-400" />
            Top-5 Predictions
          </span>
          <span className="text-[10px] font-mono text-slate-500">Softmax Probabilities</span>
        </div>

        <div className="space-y-2">
          {topPredictions.length > 0 ? (
            topPredictions.map((item, index) => {
              const itemPct = (item.confidence * 100).toFixed(1);
              const isFirst = index === 0;
              return (
                <div
                  key={index}
                  className={`p-2 rounded-lg border transition-all duration-200 ${
                    isFirst
                      ? 'bg-slate-800/90 border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                      : 'bg-slate-900/50 border-slate-800/60 hover:bg-slate-800/40'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-1">
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono font-bold ${
                          isFirst
                            ? 'bg-gradient-to-tr from-cyan-500 to-blue-600 text-white'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {index + 1}
                      </span>
                      <span className={`font-semibold capitalize ${isFirst ? 'text-cyan-300' : 'text-slate-300'}`}>
                        {item.class}
                      </span>
                    </div>
                    <span className="font-mono font-bold text-slate-300">
                      {itemPct}%
                    </span>
                  </div>
                  
                  {/* Confidence bar */}
                  <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isFirst
                          ? `bg-gradient-to-r ${progressColor}`
                          : 'bg-slate-700'
                      }`}
                      style={{ width: `${Math.max(2, itemPct)}%` }}
                    />
                  </div>
                </div>
              );
            })
          ) : (
            <div className="text-center py-6 text-slate-500 text-xs font-mono">
              Waiting for stream inference data...
            </div>
          )}
        </div>
      </div>

      {/* Model Spec Footer Tag */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span>ASLTransformer (2.31M)</span>
        </div>
        <span>696-Dim Velocity</span>
      </div>
    </div>
  );
}
