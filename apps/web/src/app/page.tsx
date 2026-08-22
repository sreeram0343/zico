'use client';

import React, { useState } from 'react';
import { Timeline } from '@/components/Timeline';
import { Compass, Sparkles, Shield, Activity, RefreshCw } from 'lucide-react';

export default function Home() {
  const [tripId, setTripId] = useState<string>('trip_demo_global_01');

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-600/30">
              <Compass className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">ZICO</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-blue-950 text-blue-400 border border-blue-800">
                  Phase 2 Operations
                </span>
              </div>
              <p className="text-xs text-slate-400">Intelligent Multi-Agent Travel Orchestration</p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              <span>HITL Security Active</span>
            </div>

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-slate-300">
              <Activity className="w-3.5 h-3.5 text-blue-400" />
              <span className="font-mono">Trip: {tripId}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main App Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        <Timeline tripId={tripId} />
      </div>
    </main>
  );
}
