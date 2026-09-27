'use client';

import React from 'react';
import { Globe } from 'lucide-react';

export function TravelAssistantCard() {
  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] p-4 shadow-soft flex items-center gap-3.5 mb-4">
      <div className="w-11 h-11 rounded-full bg-[#FFF6D8] border border-[#FDE266] flex items-center justify-center shrink-0 text-[#101828]">
        <Globe className="w-6 h-6 text-[#D99E10]" />
      </div>
      <div>
        <h3 className="text-sm font-bold text-[#101828]">Your Travel Assistant</h3>
        <p className="text-xs text-[#667085] leading-snug mt-0.5">
          Get flight info, plan trips, explore destinations, and more.
        </p>
      </div>
    </div>
  );
}
