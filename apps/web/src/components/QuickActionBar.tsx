'use client';

import React from 'react';
import { Plane, Calendar, MapPin, Search } from 'lucide-react';

interface QuickAction {
  id: string;
  label: string;
  query: string;
  icon: React.ComponentType<{ className?: string }>;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    id: 'flights',
    label: 'Find Flights',
    query: 'Find me a flight from Kochi to Dubai next Friday',
    icon: Plane,
  },
  {
    id: 'itinerary',
    label: 'Plan Itinerary',
    query: 'Help me plan a 3-day trip itinerary to Bangalore',
    icon: Calendar,
  },
  {
    id: 'destinations',
    label: 'Explore Destinations',
    query: 'What are the best places to visit in Munnar?',
    icon: MapPin,
  },
  {
    id: 'research',
    label: 'Travel Research',
    query: 'What are the baggage policies for Emirates flights?',
    icon: Search,
  },
];

interface QuickActionBarProps {
  onSelectAction: (query: string) => void;
  disabled?: boolean;
}

export function QuickActionBar({ onSelectAction, disabled }: QuickActionBarProps) {
  return (
    <div
      className="flex items-center gap-2.5 overflow-x-auto pb-1 mb-4 no-scrollbar"
      role="toolbar"
      aria-label="Quick travel actions"
    >
      {QUICK_ACTIONS.map((action) => {
        const Icon = action.icon;
        return (
          <button
            key={action.id}
            onClick={() => onSelectAction(action.query)}
            disabled={disabled}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-[#E5E7EB] hover:border-[#F7C948] hover:bg-[#FFFDF7] text-xs sm:text-sm font-semibold text-[#101828] shadow-sm transition-all duration-150 whitespace-nowrap active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed group"
          >
            <span className="w-5 h-5 rounded-full bg-[#FFF6D8] flex items-center justify-center text-[#D99E10] group-hover:bg-[#F7C948] transition-colors">
              <Icon className="w-3 h-3 text-[#101828]" />
            </span>
            <span>{action.label}</span>
          </button>
        );
      })}
    </div>
  );
}
