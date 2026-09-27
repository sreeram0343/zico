'use client';

import React, { useState } from 'react';
import { ZicoSidebar, NavItemKey } from '@/components/ZicoSidebar';
import { HeroSection } from '@/components/HeroSection';
import { QuickActionBar } from '@/components/QuickActionBar';
import { TravelAssistantCard } from '@/components/TravelAssistantCard';
import { PopularQuestionsCard } from '@/components/PopularQuestionsCard';
import { RecentConversationsCard } from '@/components/RecentConversationsCard';
import { ThemeToggle } from '@/components/ThemeToggle';
import { ProfileMenu } from '@/components/ProfileMenu';
import { Timeline } from '@/components/Timeline';
import { Menu, PanelRight, X } from 'lucide-react';

export default function Home() {
  const [tripId] = useState<string>('trip_demo_global_01');
  const [activeTab, setActiveTab] = useState<NavItemKey>('chat');
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const [isMobileRightPanelOpen, setIsMobileRightPanelOpen] = useState(false);

  // External prompt triggered by clicking quick actions, popular questions, or recent conversations
  const [externalPrompt, setExternalPrompt] = useState<{
    text: string;
    timestamp: number;
  } | null>(null);

  const handleSelectPrompt = (promptText: string) => {
    setExternalPrompt({ text: promptText, timestamp: Date.now() });
    setIsMobileRightPanelOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#FAFAF7] text-[#101828] flex flex-col antialiased">
      {/* Top Header Bar for Mobile/Tablet controls & Top Right Desktop controls */}
      <header className="bg-white border-b border-[#E5E7EB] sticky top-0 z-40 px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Mobile Left Sidebar Toggle */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsMobileNavOpen(true)}
            className="lg:hidden p-2 rounded-xl border border-[#E5E7EB] text-[#475467] hover:bg-[#FFFDF7] hover:text-[#101828] transition-colors"
            aria-label="Open sidebar menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          {/* Mobile Logo display */}
          <div className="flex items-center gap-2 lg:hidden">
            <span className="font-extrabold text-xl tracking-tight text-[#101828]">ZICO</span>
          </div>
        </div>

        {/* Top Right Controls: Theme Toggle & Profile Menu */}
        <div className="flex items-center gap-3 sm:gap-4 ml-auto">
          {/* Mobile Right Sidebar Toggle */}
          <button
            onClick={() => setIsMobileRightPanelOpen(true)}
            className="xl:hidden p-2 rounded-full border border-[#E5E7EB] text-[#475467] hover:bg-[#FFFDF7] transition-colors"
            aria-label="Open assistant panel"
            title="Assistant panel"
          >
            <PanelRight className="w-4 h-4" />
          </button>

          <ThemeToggle />
          <ProfileMenu name="John Doe" initials="JD" />
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <div className="flex-1 flex max-w-[1600px] w-full mx-auto relative overflow-hidden">
        {/* 1. Left Sidebar (Desktop) */}
        <div className="hidden lg:block shrink-0 h-[calc(100vh-4rem)] sticky top-16">
          <ZicoSidebar
            activeTab={activeTab}
            onSelectTab={(tab) => {
              setActiveTab(tab);
              if (tab === 'plan') {
                handleSelectPrompt('Help me plan a 3-day trip itinerary');
              } else if (tab === 'flights') {
                handleSelectPrompt('Find flights from Pune to Dubai tomorrow');
              } else if (tab === 'destinations') {
                handleSelectPrompt('What are the best places to visit in Munnar?');
              } else if (tab === 'research') {
                handleSelectPrompt('Check baggage policies for international flights');
              }
            }}
          />
        </div>

        {/* Left Sidebar (Mobile Drawer) */}
        {isMobileNavOpen && (
          <div className="fixed inset-0 z-50 flex lg:hidden">
            <div
              className="fixed inset-0 bg-black/30 backdrop-blur-sm"
              onClick={() => setIsMobileNavOpen(false)}
            />
            <div className="relative z-10 w-[260px] h-full shadow-2xl bg-white animate-in slide-in-from-left duration-200">
              <ZicoSidebar
                activeTab={activeTab}
                onSelectTab={(tab) => {
                  setActiveTab(tab);
                  setIsMobileNavOpen(false);
                }}
                onCloseMobile={() => setIsMobileNavOpen(false)}
              />
            </div>
          </div>
        )}

        {/* 2. Main Content Area */}
        <main className="flex-1 flex flex-col p-3 sm:p-5 lg:p-6 overflow-y-auto max-w-full">
          {/* Hero Section */}
          <HeroSection />

          {/* Quick Actions Bar */}
          <QuickActionBar onSelectAction={handleSelectPrompt} />

          {/* Chat Container Card */}
          <div className="flex-1 bg-white rounded-3xl border border-[#E5E7EB] shadow-soft p-4 sm:p-6 min-h-[520px] flex flex-col">
            <Timeline tripId={tripId} externalPrompt={externalPrompt} />
          </div>
        </main>

        {/* 3. Right Sidebar (Desktop) */}
        <aside
          className="hidden xl:block w-[340px] shrink-0 p-5 pl-0 overflow-y-auto h-[calc(100vh-4rem)] sticky top-16 select-none"
          aria-label="Assistant tools and history"
        >
          <TravelAssistantCard />
          <PopularQuestionsCard onSelectQuestion={handleSelectPrompt} />
          <RecentConversationsCard onSelectConversation={handleSelectPrompt} />
        </aside>

        {/* Right Sidebar (Mobile / Tablet Drawer) */}
        {isMobileRightPanelOpen && (
          <div className="fixed inset-0 z-50 flex justify-end xl:hidden">
            <div
              className="fixed inset-0 bg-black/30 backdrop-blur-sm"
              onClick={() => setIsMobileRightPanelOpen(false)}
            />
            <div className="relative z-10 w-[320px] sm:w-[350px] h-full shadow-2xl bg-[#FAFAF7] p-5 overflow-y-auto animate-in slide-in-from-right duration-200 border-l border-[#E5E7EB]">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-base text-[#101828]">Travel Assistant</h3>
                <button
                  onClick={() => setIsMobileRightPanelOpen(false)}
                  className="p-1.5 rounded-lg text-[#667085] hover:bg-white transition-colors"
                  aria-label="Close assistant drawer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <TravelAssistantCard />
              <PopularQuestionsCard onSelectQuestion={handleSelectPrompt} />
              <RecentConversationsCard onSelectConversation={handleSelectPrompt} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
