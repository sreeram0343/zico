'use client';

import React from 'react';
import { RotateCcw, Scale, Luggage, Bed } from 'lucide-react';

interface FollowUpAction {
  id: string;
  label: string;
  query: string;
  icon: React.ComponentType<{ className?: string }>;
}

export interface QuickAction {
  id: string;
  label: string;
  query: string;
  icon?: string;
}

export interface QuickFollowUpActionsProps {
  onSelectAction: (query: string) => void;
  disabled?: boolean;
  actions?: QuickAction[];
}

export function QuickFollowUpActions({
  onSelectAction,
  disabled,
  actions,
}: QuickFollowUpActionsProps) {
  if (!actions || actions.length === 0) {
    return null;
  }
  return (
    <div
      className="mt-3 flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar"
      role="toolbar"
      aria-label="Quick follow-up options"
    >
      {actions.map((item) => {
        let Icon = RotateCcw;
        if (item.id === 'compare') Icon = Scale;
        else if (item.id === 'baggage') Icon = Luggage;
        else if (item.id === 'hotels' || item.id.includes('hotel')) Icon = Bed;

        return (
          <button
            key={item.id}
            onClick={() => onSelectAction(item.query)}
            disabled={disabled}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white hover:bg-[#FFF6D8] border border-[#E5E7EB] hover:border-[#F7C948] text-xs font-semibold text-[#344054] hover:text-[#101828] shadow-sm transition-all duration-150 whitespace-nowrap active:scale-95 disabled:opacity-50"
          >
            <Icon className="w-3.5 h-3.5 text-[#D99E10]" />
            <span>{item.label}</span>
          </button>
        );
      })}
    </div>
  );
}
