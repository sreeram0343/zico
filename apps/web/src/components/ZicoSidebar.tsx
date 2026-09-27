'use client';

import React from 'react';
import Image from 'next/image';
import {
  MessageSquare,
  Calendar,
  Plane,
  MapPin,
  Search,
  Briefcase,
  Mic,
  Settings,
  HelpCircle,
  X,
} from 'lucide-react';

export type NavItemKey =
  | 'chat'
  | 'plan'
  | 'flights'
  | 'destinations'
  | 'research'
  | 'trips'
  | 'voice'
  | 'settings'
  | 'help';

interface NavItem {
  key: NavItemKey;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const PRIMARY_NAV_ITEMS: NavItem[] = [
  { key: 'chat', label: 'Chat', icon: MessageSquare },
  { key: 'plan', label: 'Plan a Trip', icon: Calendar },
  { key: 'flights', label: 'Flights', icon: Plane },
  { key: 'destinations', label: 'Destinations', icon: MapPin },
  { key: 'research', label: 'Travel Research', icon: Search },
  { key: 'trips', label: 'My Trips', icon: Briefcase },
  { key: 'voice', label: 'Voice Input', icon: Mic },
];

const SECONDARY_NAV_ITEMS: NavItem[] = [
  { key: 'settings', label: 'Settings', icon: Settings },
  { key: 'help', label: 'Help & Support', icon: HelpCircle },
];

interface ZicoSidebarProps {
  activeTab: NavItemKey;
  onSelectTab: (tab: NavItemKey) => void;
  onCloseMobile?: () => void;
}

export function ZicoSidebar({ activeTab, onSelectTab, onCloseMobile }: ZicoSidebarProps) {
  return (
    <aside
      className="w-[250px] shrink-0 bg-white border-r border-[#E5E7EB] flex flex-col justify-between h-full select-none"
      aria-label="Sidebar navigation"
    >
      <div className="p-5 flex flex-col">
        {/* Top Logo */}
        <div className="flex items-center justify-between mb-7">
          <div className="flex items-start gap-2.5">
            <div className="relative mt-0.5">
              {/* Yellow aviation icon */}
              <div className="w-8 h-8 rounded-full bg-[#FFF6D8] flex items-center justify-center text-[#D99E10]">
                <Plane className="w-5 h-5 fill-[#F7C948] stroke-[#B37D08] transform -rotate-45" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-extrabold text-2xl tracking-tight text-[#101828]">ZICO</span>
              </div>
              <p className="text-[11px] font-medium text-[#667085] leading-tight">
                Intelligent Travel
                <br />
                Operations Assistant
              </p>
            </div>
          </div>

          {onCloseMobile && (
            <button
              onClick={onCloseMobile}
              className="lg:hidden p-1.5 rounded-lg text-[#667085] hover:bg-slate-100 transition-colors"
              aria-label="Close sidebar"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Primary Navigation */}
        <nav className="space-y-1" aria-label="Main Navigation">
          {PRIMARY_NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.key;
            return (
              <button
                key={item.key}
                onClick={() => {
                  onSelectTab(item.key);
                  onCloseMobile?.();
                }}
                className={`w-full flex items-center gap-3.5 px-4 py-2.5 rounded-2xl text-[14px] font-semibold transition-all duration-150 text-left ${
                  isActive
                    ? 'bg-[#F7C948] text-[#101828] shadow-sm font-bold'
                    : 'text-[#475467] hover:bg-[#FAFAF7] hover:text-[#101828]'
                }`}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon
                  className={`w-4 h-4 transition-colors ${
                    isActive ? 'text-[#101828]' : 'text-[#667085]'
                  }`}
                />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Divider */}
        <div className="my-5 border-t border-[#F2F4F7]" />

        {/* Secondary Navigation */}
        <nav className="space-y-1" aria-label="Secondary Navigation">
          {SECONDARY_NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.key;
            return (
              <button
                key={item.key}
                onClick={() => {
                  onSelectTab(item.key);
                  onCloseMobile?.();
                }}
                className={`w-full flex items-center gap-3.5 px-4 py-2 rounded-2xl text-[13px] font-medium transition-all duration-150 text-left ${
                  isActive
                    ? 'bg-[#FFF6D8] text-[#101828] font-semibold'
                    : 'text-[#667085] hover:bg-[#FAFAF7] hover:text-[#101828]'
                }`}
              >
                <Icon className="w-4 h-4 text-[#667085]" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Travel Illustration Banner at bottom */}
      <div className="p-4 pt-0">
        <div className="bg-[#FFFDF7] rounded-2xl p-3 border border-[#F3F4F6] text-center flex flex-col items-center">
          <div className="relative w-36 h-28 mb-1 overflow-hidden flex items-center justify-center">
            <Image
              src="/sidebar-travel.jpg"
              alt="Travel illustration"
              width={144}
              height={112}
              className="object-contain"
              priority
            />
          </div>
          <p className="text-[13px] font-bold text-[#101828] leading-tight">
            Explore the World
            <br />
            <span className="relative inline-block font-extrabold">
              Smarter
              <span className="absolute bottom-0 left-0 w-full h-[3px] bg-[#F7C948] rounded-full" />
            </span>
            <br />
            With ZICO
          </p>
        </div>
      </div>
    </aside>
  );
}
