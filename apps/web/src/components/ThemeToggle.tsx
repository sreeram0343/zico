'use client';

import React, { useState } from 'react';
import { Sun, Moon } from 'lucide-react';

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  return (
    <button
      onClick={() => setIsDark((prev) => !prev)}
      className="w-10 h-10 rounded-full bg-white border border-[#E5E7EB] shadow-sm flex items-center justify-center text-[#475467] hover:text-[#101828] hover:border-[#D0D5DD] transition-all"
      aria-label="Toggle visual theme"
      title="Toggle theme"
    >
      {isDark ? <Sun className="w-4 h-4 text-[#F5BE22]" /> : <Moon className="w-4 h-4" />}
    </button>
  );
}
