'use client';

import React, { useRef } from 'react';
import { Paperclip, Mic, MicOff, Send, Loader2 } from 'lucide-react';

interface ChatComposerProps {
  value: string;
  onChange: (val: string) => void;
  onSend: () => void;
  isProcessing: boolean;
  isRecording: boolean;
  recordingSeconds: number;
  onToggleVoice: () => void;
  disabled?: boolean;
}

export function ChatComposer({
  value,
  onChange,
  onSend,
  isProcessing,
  isRecording,
  recordingSeconds,
  onToggleVoice,
  disabled,
}: ChatComposerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (value.trim() && !isProcessing && !disabled) {
        onSend();
      }
    }
  };

  const handleFileAttach = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onChange(value ? `${value} [Attached: ${file.name}]` : `Review attached travel document: ${file.name}`);
    }
  };

  return (
    <div className="w-full bg-[#FFFDF7] rounded-full border border-[#E5E7EB] shadow-soft p-1.5 sm:p-2 flex items-center gap-2 focus-within:border-[#F7C948] focus-within:ring-2 focus-within:ring-[#F7C948]/20 transition-all">
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileAttach}
        className="hidden"
        accept=".pdf,.png,.jpg,.jpeg,.txt,.csv"
        aria-label="Upload travel documents"
      />

      {/* Attachment Button */}
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={isProcessing || disabled}
        className="w-9 h-9 sm:w-10 sm:h-10 rounded-full flex items-center justify-center text-[#667085] hover:text-[#101828] hover:bg-white transition-colors shrink-0 disabled:opacity-40"
        title="Attach travel itinerary or document"
        aria-label="Attach file"
      >
        <Paperclip className="w-4 h-4 sm:w-5 sm:h-5 -rotate-45" />
      </button>

      {/* Input or Voice Recording Indicator */}
      {isRecording ? (
        <div className="flex-1 flex items-center gap-2.5 px-3">
          <div className="w-3 h-3 rounded-full bg-red-500 animate-ping shrink-0" />
          <span className="text-sm font-semibold text-red-600">
            Listening... ({recordingSeconds}s)
          </span>
          <span className="text-xs text-[#667085]">Speak your travel query now</span>
        </div>
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your travel question here..."
          disabled={isProcessing || disabled}
          className="flex-1 bg-transparent border-none text-sm sm:text-base text-[#101828] placeholder-[#98A2B3] focus:outline-none px-2"
          aria-label="Travel query input"
        />
      )}

      {/* Voice Microphone Button */}
      <button
        type="button"
        onClick={onToggleVoice}
        disabled={isProcessing || disabled}
        className={`w-9 h-9 sm:w-10 sm:h-10 rounded-full flex items-center justify-center transition-all shrink-0 ${
          isRecording
            ? 'bg-red-500 text-white animate-pulse'
            : 'bg-[#FFF6D8] text-[#B37D08] hover:bg-[#F7C948] hover:text-[#101828]'
        } disabled:opacity-40`}
        title={isRecording ? 'Stop recording' : 'Voice input'}
        aria-label={isRecording ? 'Stop recording voice input' : 'Start voice input'}
      >
        {isRecording ? (
          <MicOff className="w-4 h-4 sm:w-5 sm:h-5" />
        ) : (
          <Mic className="w-4 h-4 sm:w-5 sm:h-5 text-[#101828]" />
        )}
      </button>

      {/* Send Button */}
      <button
        type="button"
        onClick={onSend}
        disabled={!value.trim() || isProcessing || disabled}
        className="w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-[#F7C948] hover:bg-[#F5BE22] text-[#101828] flex items-center justify-center shadow-sm transition-all shrink-0 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-[#F7C948]"
        title="Send travel question"
        aria-label="Send message"
      >
        {isProcessing ? (
          <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin text-[#101828]" />
        ) : (
          <Send className="w-4 h-4 sm:w-5 sm:h-5 transform translate-x-[-1px] text-[#101828]" />
        )}
      </button>
    </div>
  );
}
