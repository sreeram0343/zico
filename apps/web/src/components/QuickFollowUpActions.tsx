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

const DEFAULT_FOLLOW_UPS: QuickAction[] = [
  {
    id: 'return',
    label: 'Show return flights',
    query: 'Show return flights from Dubai to Kochi next week',
  },
  {
    id: 'compare',
    label: 'Compare airlines',
    query: 'Compare Emirates and IndiGo flights on this route',
  },
  {
    id: 'baggage',
    label: 'Check baggage policy',
    query: 'Check the baggage policy for these flights',
  },
  {
    id: 'hotels',
    label: 'Find hotels in Dubai',
    query: 'Hotels in Dubai under 15k',
  },
];

interface QuickFollowUpActionsProps {
  onSelectAction: (query: string) => void;
  disabled?: boolean;
  actions?: QuickAction[];
}

export function QuickFollowUpActions({
  onSelectAction,
  disabled,
  actions,
}: QuickFollowUpActionsProps) {
  const displayActions = actions && actions.length > 0 ? actions : DEFAULT_FOLLOW_UPS;
  return (
    <div
      className="mt-3 flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar"
      role="toolbar"
      aria-label="Quick follow-up options"
    >
      {displayActions.map((item) => {
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
