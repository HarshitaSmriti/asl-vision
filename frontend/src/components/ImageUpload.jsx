import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, Sparkles, Loader2, AlertCircle, Info, ShieldAlert } from 'lucide-react';

export default function ImageUpload({ onPredictionComplete }) {
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
    if (file && file.type.startsWith('image/')) {
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
      const response = await fetch('/predict/image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to process image');
      }

      const data = await response.json();
      setResult(data);
      if (onPredictionComplete) {
        onPredictionComplete(data);
      }
    } catch (err) {
      console.error('Image prediction error:', err);
      setError(err.message || 'An error occurred during image analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Upload Box / Image Preview */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="relative w-full aspect-video bg-[#0a0d14] rounded-2xl overflow-hidden border-2 border-dashed border-slate-800 hover:border-violet-500/50 transition-all flex flex-col items-center justify-center p-4 text-center group"
      >
        {previewUrl ? (
          <div className="relative w-full h-full flex items-center justify-center">
            <img
              src={previewUrl}
              alt="Uploaded ASL"
              className="max-h-full max-w-full rounded-lg object-contain"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-slate-400">
            <div className="w-14 h-14 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center group-hover:scale-105 transition-transform">
              <ImageIcon className="w-7 h-7 text-violet-400" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-200 text-sm">Drag & drop your ASL image here</h3>
              <p className="text-xs text-slate-500 mt-1">Supports JPG, PNG, WEBP</p>
            </div>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-violet-400 border border-slate-700 rounded-xl text-xs font-semibold transition-all mt-1"
            >
              Browse Local Photos
            </button>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>

      {/* Explicit Single-Image Mode Disclaimer Note */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 flex items-start gap-2.5 text-amber-300 text-xs">
        <Info className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
        <p className="leading-relaxed">
          <strong className="font-semibold text-amber-200">Single Image Mode:</strong> The trained model is fundamentally temporal and expects a 64-frame motion sequence. Image mode extracts pose/hand landmarks and produces an approximate pose-based estimate. For full 75.72% test accuracy, use <strong>Live Camera</strong> or <strong>Video Upload</strong>.
        </p>
      </div>

      {/* Control Bar & Actions */}
      <div className="bg-[#121722]/90 border border-slate-800/80 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-3 shadow-lg">
        <div className="flex items-center gap-2">
          {selectedFile && (
            <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
              <ImageIcon className="w-4 h-4 text-violet-400" />
              <span className="truncate max-w-[200px]">{selectedFile.name}</span>
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
                : 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-violet-500/20 hover:opacity-90'
            }`}
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-white" />
                Detecting Landmarks...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Analyze Single Image
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-center gap-2.5 text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
