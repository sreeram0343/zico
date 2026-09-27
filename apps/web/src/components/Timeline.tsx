'use client';

import React, { useState, useEffect, useRef } from 'react';
import { StreamEvent, InterruptEvent, TripSegment, ChatMessage, FlightOption } from '@/types';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { FlightResultsCard } from '@/components/FlightResultsCard';
import { QuickFollowUpActions } from '@/components/QuickFollowUpActions';
import { ChatComposer } from '@/components/ChatComposer';
import {
  Plane,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ExternalLink,
  RefreshCw,
  Loader2,
  User,
} from 'lucide-react';

function getFriendlyWorkerName(node: string): string {
  const lower = node.toLowerCase();
  if (lower.includes('flight')) return 'Searching flights...';
  if (lower.includes('hotel') || lower.includes('research')) return 'Researching travel options...';
  if (lower.includes('policy') || lower.includes('rag')) return 'Checking travel policies...';
  if (lower.includes('disruption')) return 'Analyzing travel disruptions...';
  if (lower.includes('validator')) return 'Verifying itinerary details...';
  if (lower.includes('booking')) return 'Processing confirmation...';
  return 'Processing travel request...';
}

function hasFlightContent(text: string): boolean {
  const lower = text.toLowerCase();
  return (
    lower.includes('available flights') ||
    lower.includes('flight options') ||
    (lower.includes('flight') && (lower.includes('dubai') || lower.includes('pune') || lower.includes('kochi')))
  );
}

interface TimelineProps {
  tripId: string;
  userId?: string;
  wsBaseUrl?: string;
  apiBaseUrl?: string;
  externalPrompt?: { text: string; timestamp: number } | null;
}

export function Timeline({
  tripId,
  userId = 'traveler_01',
  wsBaseUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  externalPrompt,
}: TimelineProps) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  // Initial messages mirror the reference design demo state
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'user',
      text: 'Find me a flight from Kochi to Dubai next Friday',
      time: '10:24 AM',
    },
    {
      role: 'assistant',
      text: 'Here are the available flights from Kochi (COK) to Dubai (DXB) for Friday, 23 May 2025. These results are from live data via AviationStack.',
      time: '10:24 AM',
    },
  ]);

  const [activeInterrupt, setActiveInterrupt] = useState<InterruptEvent | null>(null);
  const [itinerary, setItinerary] = useState<TripSegment[]>([]);
  const [promptInput, setPromptInput] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeWorkerNode, setActiveWorkerNode] = useState<string | null>(null);

  // Streaming token state
  const [streamingTokenMessage, setStreamingTokenMessage] = useState<string>('');
  const streamingTokenRef = useRef<string>('');
  const [activeToolCall, setActiveToolCall] = useState<{ tool: string; input?: any } | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);

  // Voice recording state
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingSeconds, setRecordingSeconds] = useState<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const wsUrl = `${wsBaseUrl.replace(/^http/, 'ws')}/ws/stream/${tripId}`;
    let ws: WebSocket;

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        setStreamError(null);
      };

      ws.onmessage = (event) => {
        let streamEvent: StreamEvent;
        try {
          streamEvent = JSON.parse(event.data);
        } catch (parseErr) {
          console.error('[CLIENT] Failed to parse WebSocket message JSON:', parseErr);
          setStreamError(`Malformed data from server: ${String(parseErr)}`);
          return;
        }

        // 1. Token chunk stream
        if (streamEvent.type === 'token') {
          const tokenChunk = streamEvent.content || '';
          streamingTokenRef.current += tokenChunk;
          setStreamingTokenMessage((prev) => prev + tokenChunk);
          setIsProcessing(false);
        }

        // 2. Tool call invocation
        else if (streamEvent.type === 'tool_call') {
          setActiveToolCall({
            tool: streamEvent.tool || 'Tool',
            input: streamEvent.input,
          });
          setActiveWorkerNode(streamEvent.tool || null);
        }

        // 3. State update (Itinerary changes)
        else if (streamEvent.type === 'state_update') {
          if (streamEvent.itinerary && Array.isArray(streamEvent.itinerary)) {
            setItinerary(streamEvent.itinerary);
          }
        }

        // 4. Status update
        else if (streamEvent.type === 'status') {
          setActiveWorkerNode(streamEvent.node || 'supervisor');
          setIsProcessing(true);
        }

        // 5. Node update
        else if (streamEvent.type === 'node_update') {
          setActiveWorkerNode(streamEvent.node || null);

          if (streamEvent.output?.itinerary && Array.isArray(streamEvent.output.itinerary)) {
            setItinerary(streamEvent.output.itinerary);
          }

          const msgText =
            streamEvent.message ||
            streamEvent.content ||
            (streamEvent.output?.messages && streamEvent.output.messages[0]?.content) ||
            '';

          const isInternalNotice =
            !msgText ||
            msgText === 'Node completed' ||
            msgText.startsWith('Orchestrating request:') ||
            ['input_node', 'supervisor_node', 'validator_node', 'booking_approval_node'].includes(
              streamEvent.node || ''
            );

          if (msgText && !isInternalNotice && !streamingTokenRef.current) {
            setMessages((prev) => [
              ...prev,
              {
                role: 'assistant',
                text: typeof msgText === 'string' ? msgText : JSON.stringify(msgText),
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                sources: streamEvent.sources,
              },
            ]);
          }
        }

        // 6. Human-in-the-Loop Interrupt
        else if (streamEvent.type === 'interrupt' && streamEvent.interrupt_value) {
          setActiveInterrupt(streamEvent.interrupt_value);
          setIsProcessing(false);
          setActiveToolCall(null);
        }

        // 7. Audio playback
        else if (streamEvent.type === 'voice_chunk' && streamEvent.audio_base64) {
          try {
            const audio = new Audio(`data:audio/wav;base64,${streamEvent.audio_base64}`);
            audio.play().catch(() => {});
          } catch (e) {
            console.debug('Audio playback error:', e);
          }
        }

        // 8. Turn completion
        else if (streamEvent.type === 'turn_complete') {
          if (streamingTokenRef.current) {
            const finalTokenContent = streamingTokenRef.current;
            setMessages((prev) => [
              ...prev,
              {
                role: 'assistant',
                text: finalTokenContent,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              },
            ]);
            streamingTokenRef.current = '';
            setStreamingTokenMessage('');
          }
          setIsProcessing(false);
          setActiveWorkerNode(null);
          setActiveToolCall(null);
        }

        // 9. Error frame
        else if (streamEvent.type === 'error') {
          const errText = streamEvent.message || streamEvent.content || 'An error occurred';
          setStreamError(errText);
          setIsProcessing(false);
          setActiveToolCall(null);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setActiveWorkerNode(null);
        setActiveToolCall(null);
      };

      ws.onerror = () => {
        setIsConnected(false);
      };

      setSocket(ws);
    } catch {
      setIsConnected(false);
    }

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [tripId, wsBaseUrl]);

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingTokenMessage, activeInterrupt, activeToolCall]);

  // Voice recording timer
  useEffect(() => {
    if (isRecording) {
      setRecordingSeconds(0);
      timerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecordingSeconds(0);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  // Voice recording toggle
  const handleToggleVoiceRecording = async () => {
    if (isRecording) {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          stream.getTracks().forEach((track) => track.stop());
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });

          setIsProcessing(true);
          try {
            const formData = new FormData();
            formData.append('file', audioBlob, 'voice_query.wav');

            const resp = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/v1/voice/transcribe`, {
              method: 'POST',
              body: formData,
            });

            if (resp.ok) {
              const data = await resp.json();
              if (data.transcript) {
                setPromptInput(data.transcript);
              }
            } else {
              setStreamError('Audio transcription failed on server.');
            }
          } catch (err) {
            setStreamError(`Voice service error: ${String(err)}`);
          } finally {
            setIsProcessing(false);
          }
        };

        mediaRecorder.start();
        setIsRecording(true);
      } catch (err) {
        setStreamError('Microphone access denied or unavailable.');
        console.error('Mic error:', err);
      }
    }
  };

  // Send message
  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || promptInput).trim();
    if (!text || isProcessing) return;

    setStreamError(null);
    streamingTokenRef.current = '';
    setStreamingTokenMessage('');
    setActiveToolCall(null);

    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        text,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    setPromptInput('');
    setIsProcessing(true);

    if (socket && isConnected && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: 'prompt',
          message: text,
          content: text,
          trip_id: tripId,
          user_id: userId,
          enable_tts: true,
        })
      );
    } else {
      try {
        const resp = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            session_id: tripId,
            trip_id: tripId,
            user_id: userId,
          }),
        });

        if (resp.ok) {
          const data = await resp.json();
          const replyText = data.response || data.reply || 'Your travel inquiry has been processed.';

          if (data.itinerary && Array.isArray(data.itinerary) && data.itinerary.length > 0) {
            setItinerary(data.itinerary);
          }

          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              text: replyText,
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              sources: data.sources || [],
            },
          ]);
        } else {
          setStreamError(`Server request failed (Status: ${resp.status})`);
        }
      } catch (err) {
        setStreamError(`Network connection error: ${String(err)}`);
      } finally {
        setIsProcessing(false);
      }
    }
  };

  // Listen for external prompts from QuickActionBar, PopularQuestions, or RecentConversations
  useEffect(() => {
    if (externalPrompt && externalPrompt.text) {
      handleSendMessage(externalPrompt.text);
    }
  }, [externalPrompt]);

  // Respond to Human-in-the-Loop Interrupt
  const handleInterruptResponse = (decision: 'approve' | 'reject') => {
    if (!activeInterrupt) return;

    if (socket && isConnected && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: 'interrupt_response',
          action_id: activeInterrupt.action_id,
          decision,
          trip_id: tripId,
        })
      );
    } else {
      fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/v1/approval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action_id: activeInterrupt.action_id,
          decision,
          trip_id: tripId,
        }),
      }).catch(console.error);
    }

    setMessages((prev) => [
      ...prev,
      {
        role: 'system',
        text: `HITL Decision submitted: ${decision.toUpperCase()}`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    setActiveInterrupt(null);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Container */}
      <div className="flex-1 space-y-6 overflow-y-auto pr-1 pb-4">
        {messages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          const isSystem = msg.role === 'system';
          const showFlightCard = !isUser && !isSystem && hasFlightContent(msg.text);

          if (isSystem) {
            return (
              <div key={`msg-${idx}`} className="text-center my-2">
                <span className="inline-block px-3 py-1 rounded-full bg-[#FFFDF7] border border-[#E5E7EB] text-[11px] font-medium text-[#667085]">
                  {msg.text}
                </span>
              </div>
            );
          }

          return (
            <div
              key={`msg-${idx}`}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} w-full`}
            >
              <div className={`flex items-start gap-2.5 max-w-[92%] sm:max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                {isUser ? (
                  <div className="w-8 h-8 rounded-full bg-[#E5E7EB] flex items-center justify-center text-[#475467] shrink-0 mt-0.5 shadow-sm">
                    <User className="w-4 h-4" />
                  </div>
                ) : (
                  <div className="w-8 h-8 rounded-full bg-[#F7C948] flex items-center justify-center text-[#101828] shrink-0 mt-0.5 shadow-sm font-bold">
                    <Plane className="w-4 h-4 fill-[#101828] stroke-none transform -rotate-45" />
                  </div>
                )}

                {/* Message Bubble */}
                <div
                  className={`rounded-2xl px-4 py-3 text-sm leading-relaxed transition-all ${
                    isUser
                      ? 'bg-[#FFF6D8] border border-[#FDE266] text-[#101828] rounded-tr-sm shadow-sm'
                      : 'bg-[#FFFDF7] border border-[#E5E7EB] text-[#101828] rounded-tl-sm shadow-sm w-full'
                  }`}
                >
                  <MarkdownRenderer content={msg.text} />

                  {/* Flight Results Card if response is a flight inquiry */}
                  {showFlightCard && <FlightResultsCard />}

                  {/* Message Timestamp */}
                  <div
                    className={`mt-1.5 flex items-center gap-1.5 text-[10px] text-[#98A2B3] font-medium ${
                      isUser ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <span>{msg.time}</span>
                  </div>
                </div>
              </div>

              {/* Quick Follow-up Actions under the latest assistant response */}
              {!isUser && idx === messages.length - 1 && (
                <div className="ml-10 sm:ml-11 mt-1 max-w-full">
                  <QuickFollowUpActions
                    onSelectAction={(query) => handleSendMessage(query)}
                    disabled={isProcessing}
                  />
                </div>
              )}
            </div>
          );
        })}

        {/* Live Streaming Token Preview */}
        {streamingTokenMessage && (
          <div className="flex items-start gap-2.5 max-w-[90%] sm:max-w-[85%]">
            <div className="w-8 h-8 rounded-full bg-[#F7C948] flex items-center justify-center text-[#101828] shrink-0 mt-0.5 shadow-sm">
              <Plane className="w-4 h-4 fill-[#101828] stroke-none transform -rotate-45" />
            </div>
            <div className="rounded-2xl rounded-tl-sm bg-[#FFFDF7] border border-[#E5E7EB] px-4 py-3 text-sm text-[#101828] shadow-sm w-full">
              <MarkdownRenderer content={streamingTokenMessage} />
            </div>
          </div>
        )}

        {/* Thinking / Agent Processing State */}
        {isProcessing && !streamingTokenMessage && (
          <div className="flex items-center gap-3 ml-2 text-xs font-semibold text-[#667085] animate-pulse">
            <div className="w-7 h-7 rounded-full bg-[#FFF6D8] flex items-center justify-center text-[#D99E10]">
              <Loader2 className="w-4 h-4 animate-spin text-[#D99E10]" />
            </div>
            <span>
              {activeWorkerNode
                ? getFriendlyWorkerName(activeWorkerNode)
                : 'ZICO is thinking...'}
            </span>
          </div>
        )}

        {/* Stream / Connection Error UI */}
        {streamError && (
          <div className="p-3 bg-[#FEF3F2] border border-[#FECDCA] rounded-2xl flex items-center justify-between text-xs text-[#B42318]">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{streamError}</span>
            </div>
            <button
              onClick={() => setStreamError(null)}
              className="px-2 py-1 bg-white rounded-lg border border-[#FDA29B] text-[#B42318] font-bold text-[11px]"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Human-In-The-Loop Interrupt UI */}
        {activeInterrupt && (
          <div className="p-4 bg-[#FFFDF7] border-2 border-[#F7C948] rounded-2xl shadow-card my-3">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-full bg-[#F7C948] flex items-center justify-center text-[#101828] shrink-0 font-bold">
                <AlertTriangle className="w-5 h-5 text-[#101828]" />
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-extrabold text-[#101828]">
                  Approval Required: {activeInterrupt.action_type}
                </h4>
                <p className="text-xs text-[#475467] mt-1 leading-relaxed">
                  {activeInterrupt.description || activeInterrupt.prompt}
                </p>

                <div className="mt-3 flex items-center gap-3">
                  <button
                    onClick={() => handleInterruptResponse('approve')}
                    className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-[#F7C948] hover:bg-[#F5BE22] text-[#101828] font-bold text-xs shadow-sm transition-colors"
                  >
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Approve & Continue</span>
                  </button>
                  <button
                    onClick={() => handleInterruptResponse('reject')}
                    className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-[#FEF3F2] hover:bg-[#FEE4E2] text-[#B42318] font-bold text-xs border border-[#FECDCA] transition-colors"
                  >
                    <XCircle className="w-4 h-4" />
                    <span>Reject</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Chat Composer */}
      <div className="pt-3 border-t border-[#F2F4F7]">
        <ChatComposer
          value={promptInput}
          onChange={setPromptInput}
          onSend={() => handleSendMessage()}
          isProcessing={isProcessing}
          isRecording={isRecording}
          recordingSeconds={recordingSeconds}
          onToggleVoice={handleToggleVoiceRecording}
        />
      </div>
    </div>
  );
}
