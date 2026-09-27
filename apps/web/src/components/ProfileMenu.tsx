'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, User, Heart, Settings, LogOut } from 'lucide-react';

interface ProfileMenuProps {
  name?: string;
  initials?: string;
}

export function ProfileMenu({ name = 'John Doe', initials = 'JD' }: ProfileMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex items-center gap-2.5 p-1 pr-2.5 rounded-full hover:bg-slate-100/70 transition-all focus:outline-none focus:ring-2 focus:ring-[#F7C948]"
        aria-expanded={isOpen}
        aria-haspopup="menu"
        aria-label="User account menu"
      >
        <div className="w-9 h-9 rounded-full bg-[#F7C948] flex items-center justify-center font-bold text-sm text-[#101828] shadow-sm select-none">
          {initials}
        </div>
        <span className="text-sm font-semibold text-[#101828] hidden sm:inline-block">
          {name}
        </span>
        <ChevronDown
          className={`w-3.5 h-3.5 text-[#667085] transition-transform ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {isOpen && (
        <div
          className="absolute right-0 mt-2 w-52 bg-white rounded-2xl shadow-card border border-[#E5E7EB] py-1.5 z-50 text-sm animate-in fade-in slide-in-from-top-2 duration-150"
          role="menu"
        >
          <div className="px-4 py-2 border-b border-[#F2F4F7]">
            <p className="font-bold text-[#101828]">{name}</p>
            <p className="text-xs text-[#667085]">john.doe@example.com</p>
          </div>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 flex items-center gap-2.5 text-[#344054] hover:bg-[#FFFDF7] hover:text-[#101828] transition-colors text-left"
            role="menuitem"
          >
            <User className="w-4 h-4 text-[#667085]" />
            <span>Profile & Account</span>
          </button>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 flex items-center gap-2.5 text-[#344054] hover:bg-[#FFFDF7] hover:text-[#101828] transition-colors text-left"
            role="menuitem"
          >
            <Heart className="w-4 h-4 text-[#667085]" />
            <span>Saved Itineraries</span>
          </button>

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 flex items-center gap-2.5 text-[#344054] hover:bg-[#FFFDF7] hover:text-[#101828] transition-colors text-left"
            role="menuitem"
          >
            <Settings className="w-4 h-4 text-[#667085]" />
            <span>Travel Preferences</span>
          </button>

          <div className="my-1 border-t border-[#F2F4F7]" />

          <button
            onClick={() => setIsOpen(false)}
            className="w-full px-4 py-2 flex items-center gap-2.5 text-[#B42318] hover:bg-[#FEF3F2] transition-colors text-left"
            role="menuitem"
          >
            <LogOut className="w-4 h-4" />
            <span>Sign out</span>
          </button>
        </div>
      )}
    </div>
  );
}
