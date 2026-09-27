'use client';

import React from 'react';
import {
  Lightbulb,
  Plane,
  Calendar,
  MapPin,
  Building,
  FileText,
  CloudSun,
  Briefcase,
  Compass,
} from 'lucide-react';

interface PopularQuestion {
  id: string;
  label: string;
  query: string;
  icon: React.ComponentType<{ className?: string }>;
}

const POPULAR_QUESTIONS: PopularQuestion[] = [
  {
    id: 'flights',
    label: 'Find flights',
    query: 'Find me flights from Pune to Dubai next Friday',
    icon: Plane,
  },
  {
    id: 'plan_trip',
    label: 'Plan a trip',
    query: 'Plan a 3-day trip itinerary for Bangalore',
    icon: Calendar,
  },
  {
    id: 'best_places',
    label: 'Best places to visit',
    query: 'What are the best places to visit in Pune?',
    icon: MapPin,
  },
  {
    id: 'hotels',
    label: 'Hotel recommendations',
    query: 'Hotels in Pune under 10k',
    icon: Building,
  },
  {
    id: 'visa',
    label: 'Visa and travel requirements',
    query: 'What are the visa and travel requirements for Dubai?',
    icon: FileText,
  },
  {
    id: 'weather',
    label: 'Weather at destination',
    query: 'What is the current weather in Munnar?',
    icon: CloudSun,
  },
  {
    id: 'baggage',
    label: 'Baggage policy',
    query: 'What is the baggage policy for Emirates flights?',
    icon: Briefcase,
  },
  {
    id: 'tips',
    label: 'Travel tips',
    query: 'What are essential travel tips for international flying?',
    icon: Compass,
  },
];

interface PopularQuestionsCardProps {
  onSelectQuestion: (query: string) => void;
  disabled?: boolean;
}

export function PopularQuestionsCard({
  onSelectQuestion,
  disabled,
}: PopularQuestionsCardProps) {
  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] p-4 shadow-soft mb-4">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="w-4 h-4 text-[#F5BE22] fill-[#F7C948]" />
        <h3 className="text-sm font-bold text-[#101828]">Popular Things to Ask</h3>
      </div>

      <div className="space-y-1.5" role="list">
        {POPULAR_QUESTIONS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onSelectQuestion(item.query)}
              disabled={disabled}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl bg-[#FFFDF7] hover:bg-[#FFF6D8] border border-[#F2F4F7] hover:border-[#FDE266] text-xs font-semibold text-[#101828] transition-all duration-150 text-left active:scale-[0.99] disabled:opacity-50"
              role="listitem"
            >
              <Icon className="w-3.5 h-3.5 text-[#B37D08] shrink-0" />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
