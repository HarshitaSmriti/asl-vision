import React from 'react';
import { Video, Sparkles, BookOpen, Activity, Cpu } from 'lucide-react';

export default function Header({ currentMode, setMode, onOpenDictionary, modelInfo }) {
  return (
    <header className="border-b border-slate-800/80 bg-[#0d121e]/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo & Title */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-violet-600 p-[1px] shadow-lg shadow-cyan-500/20">
            <div className="w-full h-full bg-[#0d121e] rounded-[11px] flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-cyan-400 animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-cyan-300">
                ASL VISION
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono font-bold tracking-wide rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
                75.72% ACC
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Real-Time ASL Recognition &bull; 95 Classes &bull; 64-Frame Temporal Transformer
            </p>
          </div>
        </div>

        {/* Mode Selector Buttons */}
        <div className="flex items-center bg-slate-900/90 border border-slate-800 rounded-xl p-1 shadow-inner">
          <button
            onClick={() => setMode('live')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              currentMode === 'live'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${currentMode === 'live' ? 'bg-white animate-ping' : 'bg-slate-500'}`} />
            LIVE CAMERA
          </button>
          
          <button
            onClick={() => setMode('video')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              currentMode === 'video'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            VIDEO
          </button>
          
          <button
            onClick={() => setMode('image')}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              currentMode === 'image'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/25'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            IMAGE
          </button>
        </div>

        {/* Dictionary / Vocabulary Explorer */}
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenDictionary}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/60 transition-colors shadow-sm"
            title="Browse all 95 recognizable ASL vocabulary words"
          >
            <BookOpen className="w-4 h-4 text-violet-400" />
            <span className="hidden md:inline">95 Signs</span>
          </button>
        </div>
      </div>
    </header>
  );
}
