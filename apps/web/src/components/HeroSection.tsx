'use client';

import React from 'react';
import Image from 'next/image';

export function HeroSection() {
  return (
    <div className="relative w-full rounded-3xl overflow-hidden bg-white border border-[#E5E7EB] shadow-soft mb-4">
      {/* Background travel banner image with Santorini & airplane */}
      <div className="absolute inset-0 select-none pointer-events-none">
        <Image
          src="/hero-banner.jpg"
          alt="Travel scenery with airplane"
          fill
          className="object-cover object-right"
          priority
        />
        {/* Soft gradient fade from white on the left across to 60% */}
        <div className="absolute inset-0 bg-gradient-to-r from-white via-white/95 to-transparent sm:via-white/90 md:to-transparent" />
      </div>

      {/* Content */}
      <div className="relative z-10 px-6 py-8 sm:px-10 sm:py-10 max-w-2xl">
        <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-[#101828] tracking-tight leading-[1.15]">
          Hi, I&apos;m <span className="text-[#F5BE22]">ZICO</span>
        </h1>
        <p className="text-lg sm:text-xl font-bold text-[#101828] mt-1.5 mb-2.5 tracking-tight">
          Your Intelligent Travel Operations Assistant
        </p>
        <p className="text-sm sm:text-base text-[#475467] leading-relaxed max-w-xl font-normal">
          Ask me anything about flights, destinations, itineraries, travel policies and more.
          I&apos;m here to make your travel easier.
        </p>
      </div>
    </div>
  );
}
