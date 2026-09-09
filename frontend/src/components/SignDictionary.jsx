import React, { useState } from 'react';
import { X, Search, BookOpen, Sparkles, Check } from 'lucide-react';

export default function SignDictionary({ isOpen, onClose, classes = [] }) {
  const [searchQuery, setSearchQuery] = useState('');

  if (!isOpen) return null;

  const filteredClasses = classes.filter((cls) =>
    cls.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="bg-[#121722] border border-slate-800 rounded-2xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-violet-500/10 border border-violet-500/20 text-violet-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                Supported ASL Signs Directory
                <span className="text-xs font-mono font-normal px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
                  {classes.length} Vocabulary Words
                </span>
              </h2>
              <p className="text-xs text-slate-400">These are the 95 gestures trained and recognized by the ASLTransformer model.</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search Filter */}
        <div className="p-4 border-b border-slate-800/80 bg-slate-900/40">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search sign vocabulary (e.g., hello, apple, book, fine)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 transition-colors font-mono"
            />
          </div>
        </div>

        {/* Grid of Signs */}
        <div className="p-5 overflow-y-auto grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2.5">
          {filteredClasses.length > 0 ? (
            filteredClasses.map((cls, idx) => (
              <div
                key={cls}
                className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-cyan-500/40 hover:bg-slate-800/40 transition-all flex items-center justify-between group"
              >
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-slate-500 w-5">
                    {idx + 1}
                  </span>
                  <span className="text-xs font-semibold capitalize text-slate-200 group-hover:text-cyan-300 transition-colors">
                    {cls}
                  </span>
                </div>
                <Sparkles className="w-3 h-3 text-slate-600 group-hover:text-cyan-400 transition-colors" />
              </div>
            ))
          ) : (
            <div className="col-span-full text-center py-10 text-slate-500 text-xs font-mono">
              No signs found matching "{searchQuery}".
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>75.72% Test Accuracy on 95 Isolated Classes</span>
          <button
            onClick={onClose}
            className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
