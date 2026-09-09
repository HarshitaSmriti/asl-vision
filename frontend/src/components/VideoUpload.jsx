import React, { useState, useRef } from 'react';
import { Upload, Video, Play, CheckCircle2, AlertCircle, FileText, Loader2, Sparkles } from 'lucide-react';

export default function VideoUpload({ onPredictionComplete }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      setResult(null);
    }
  };

  const handleUploadAndPredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/predict/video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to process video');
      }

      const data = await response.json();
      setResult(data);
      if (onPredictionComplete) {
        onPredictionComplete(data);
      }
    } catch (err) {
      console.error('Video prediction error:', err);
      setError(err.message || 'An error occurred during video analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Upload Box / Video Preview */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="relative w-full aspect-video bg-[#0a0d14] rounded-2xl overflow-hidden border-2 border-dashed border-slate-800 hover:border-cyan-500/50 transition-all flex flex-col items-center justify-center p-4 text-center group"
      >
        {previewUrl ? (
          <div className="relative w-full h-full flex items-center justify-center">
            <video
              src={previewUrl}
              controls
              className="max-h-full max-w-full rounded-lg object-contain"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-slate-400">
            <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center group-hover:scale-105 transition-transform">
              <Upload className="w-7 h-7 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-200 text-sm">Drag & drop your ASL video here</h3>
              <p className="text-xs text-slate-500 mt-1">Supports MP4, WebM, MOV, AVI (up to 60s)</p>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 rounded-xl text-xs font-semibold transition-all mt-1"
            >
              Browse Local Files
            </button>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Control Bar & Status */}
      <div className="bg-[#121722]/90 border border-slate-800/80 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-lg">
        <div className="flex items-center gap-2">
          {selectedFile && (
            <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
              <Video className="w-4 h-4 text-cyan-400" />
              <span className="truncate max-w-[200px]">{selectedFile.name}</span>
              <span className="text-slate-500">({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {selectedFile && (
            <button
              onClick={() => {
                setSelectedFile(null);
                setPreviewUrl(null);
                setResult(null);
                setError(null);
              }}
              className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
            >
              Clear
            </button>
          )}

          <button
            onClick={handleUploadAndPredict}
            disabled={!selectedFile || loading}
            className={`flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-bold transition-all shadow-md ${
              !selectedFile || loading
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-cyan-500/20 hover:opacity-90'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-white" />
                Extracting Landmarks & Inferring...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Run ASL Recognition
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-center gap-2.5 text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Result stats summary */}
      {result && (
        <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl flex items-center justify-between text-emerald-300 text-xs font-mono">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            <span>Processed {result.frames_processed || 'all'} frames at {result.fps ? Math.round(result.fps) : 30} FPS</span>
          </div>
          <span>Temporal Resampling: 64 Frames &bull; 696 Velocity Dims</span>
        </div>
      )}
    </div>
  );
}
