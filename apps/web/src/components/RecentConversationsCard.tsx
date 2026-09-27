'use client';

import React from 'react';
import { Clock, ChevronRight } from 'lucide-react';

interface RecentItem {
  id: string;
  title: string;
  timestamp: string;
  query: string;
}

const RECENT_CONVERSATIONS: RecentItem[] = [
  {
    id: 'c1',
    title: 'Flights to Dubai',
    timestamp: '10:24 AM',
    query: 'Find flights from Pune to Dubai tomorrow',
  },
  {
    id: 'c2',
    title: '3-day Bangalore trip',
    timestamp: 'Yesterday',
    query: 'Help me plan a 3-day itinerary in Bangalore',
  },
  {
    id: 'c3',
    title: 'Baggage policy for Emirates',
    timestamp: 'May 18',
    query: 'What is the Emirates baggage policy?',
  },
  {
    id: 'c4',
    title: 'Places to visit in Munnar',
    timestamp: 'May 16',
    query: 'What are the best places to visit in Munnar?',
  },
  {
    id: 'c5',
    title: 'Flight status EK 524',
    timestamp: 'May 15',
    query: 'Check flight status for EK 524',
  },
];

interface RecentConversationsCardProps {
  onSelectConversation: (query: string) => void;
  disabled?: boolean;
}

export function RecentConversationsCard({
  onSelectConversation,
  disabled,
}: RecentConversationsCardProps) {
  return (
    <div className="bg-white rounded-2xl border border-[#E5E7EB] p-4 shadow-soft">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="w-4 h-4 text-[#475467]" />
        <h3 className="text-sm font-bold text-[#101828]">Recent Conversations</h3>
      </div>

      <div className="divide-y divide-[#F2F4F7]" role="list">
        {RECENT_CONVERSATIONS.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelectConversation(item.query)}
            disabled={disabled}
            className="w-full flex items-center justify-between py-2.5 px-1 hover:bg-[#FFFDF7] rounded-lg transition-colors text-left group"
            role="listitem"
          >
            <span className="text-xs font-semibold text-[#344054] group-hover:text-[#101828] truncate max-w-[170px]">
              {item.title}
            </span>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[11px] text-[#98A2B3]">{item.timestamp}</span>
              <ChevronRight className="w-3.5 h-3.5 text-[#D0D5DD] group-hover:text-[#667085] transition-colors" />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
