'use client';

import React, { useState, useEffect, useRef } from 'react';
import { StreamEvent, InterruptEvent, TripSegment, ChatMessage, SourceCitation } from '@/types';
import {
  Plane,
  Hotel,
  Calendar,
  AlertTriangle,
  AlertCircle,
  Wrench,
  CheckCircle2,
  XCircle,
  Send,
  Radio,
  Clock,
  MapPin,
  DollarSign,
  X,
  Mic,
  MicOff,
  ExternalLink,
  RefreshCw,
  Sparkles,
} from 'lucide-react';

interface TimelineProps {
  tripId: string;
  userId?: string;
  wsBaseUrl?: string;
  apiBaseUrl?: string;
}

export function Timeline({
  tripId,
  userId = 'traveler_01',
  wsBaseUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
  apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
}: TimelineProps) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
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
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            text: `Connected to ZICO Real-time Operations Engine (Trip: ${tripId})`,
            time: new Date().toLocaleTimeString(),
          },
        ]);
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

          if (msgText && !streamingTokenRef.current) {
            setMessages((prev) => [
              ...prev,
              {
                role: 'assistant',
                text: typeof msgText === 'string' ? msgText : JSON.stringify(msgText),
                node: streamEvent.node,
                time: new Date().toLocaleTimeString(),
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
            console.debug('Audio note:', e);
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
                time: new Date().toLocaleTimeString(),
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
          setMessages((prev) => [
            ...prev,
            {
              role: 'system',
              text: `Error: ${errText}`,
              time: new Date().toLocaleTimeString(),
            },
          ]);
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

  // Auto-scroll
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

  // Toggle voice recording
  const handleToggleVoiceRecording = async () => {
    if (isRecording) {
      // Stop recording
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      setIsRecording(false);
    } else {
      // Start recording
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

          // Send to voice transcribe API
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

  // Send message (WebSocket with HTTP Fallback)
  const handleSendPrompt = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!promptInput.trim() || isProcessing) return;

    const userText = promptInput.trim();
    setStreamError(null);
    streamingTokenRef.current = '';
    setStreamingTokenMessage('');
    setActiveToolCall(null);

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: userText, time: new Date().toLocaleTimeString() },
    ]);
    setPromptInput('');
    setIsProcessing(true);

    // If WebSocket is open and connected, send via WebSocket
    if (socket && isConnected && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: 'prompt',
          message: userText,
          content: userText,
          trip_id: tripId,
          user_id: userId,
          enable_tts: true,
        })
      );
    } else {
      // HTTP API Fallback to POST /api/v1/chat
      try {
        const resp = await fetch(`${apiBaseUrl.replace(/\/$/, '')}/api/v1/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: userText,
            session_id: tripId,
            trip_id: tripId,
            user_id: userId,
          }),
        });

        if (resp.ok) {
          const data = await resp.json();
          const replyText = data.response || data.reply || 'Your travel inquiry has been processed.';
          const sources: SourceCitation[] = data.sources || [];

          if (data.itinerary && Array.isArray(data.itinerary) && data.itinerary.length > 0) {
            setItinerary(data.itinerary);
          }

          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              text: replyText,
              time: new Date().toLocaleTimeString(),
              sources: sources,
            },
          ]);
        } else {
          const errData = await resp.json().catch(() => ({}));
          const errMsg = errData.error?.message || `Server returned HTTP ${resp.status}`;
          setStreamError(errMsg);
        }
      } catch (err) {
        setStreamError(`Network failure communicating with ZICO backend: ${String(err)}`);
      } finally {
        setIsProcessing(false);
      }
    }
  };

  // Human-in-the-Loop decision submission
  const handleDecision = (approved: boolean) => {
    if (!activeInterrupt) return;

    if (socket && isConnected) {
      const payload = {
        type: 'decision',
        approved,
        action_id: activeInterrupt.action_id,
        actor: userId,
        trip_id: tripId,
      };
      socket.send(JSON.stringify(payload));
    }

    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        text: approved
          ? `Approved proposal: ${activeInterrupt.description}`
          : `Rejected proposal: ${activeInterrupt.description}`,
        time: new Date().toLocaleTimeString(),
      },
    ]);

    setActiveInterrupt(null);
    setIsProcessing(true);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-6rem)]">
      {/* Left Column: Itinerary Timeline */}
      <div className="lg:col-span-5 flex flex-col bg-slate-900/60 backdrop-blur border border-slate-800 rounded-2xl p-5 overflow-hidden shadow-xl">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-400" />
            <h2 className="text-lg font-semibold text-slate-100">Live Trip Timeline</h2>
          </div>
          <span className="text-xs px-2.5 py-1 rounded-full bg-blue-950 text-blue-300 border border-blue-800 font-mono">
            {itinerary.length} Segments
          </span>
        </div>

        <div className="flex-1 overflow-y-auto pt-4 space-y-4 pr-1">
          {itinerary.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-center text-slate-400">
              <Calendar className="w-10 h-10 mb-2 text-slate-600" />
              <p className="text-sm font-medium">No scheduled segments yet.</p>
              <p className="text-xs text-slate-500 mt-1 max-w-xs">
                Ask ZICO to search live flights or build a custom multi-day travel itinerary.
              </p>
            </div>
          ) : (
            itinerary.map((seg, idx) => (
              <div
                key={seg.id || idx}
                className={`p-4 rounded-xl border transition-all ${
                  seg.is_confirmed
                    ? 'bg-slate-800/40 border-emerald-800/60 shadow-sm'
                    : 'bg-slate-800/20 border-slate-700/60'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-lg bg-blue-950/80 text-blue-400 border border-blue-800/50">
                      {seg.type === 'FLIGHT' ? (
                        <Plane className="w-4 h-4" />
                      ) : seg.type === 'HOTEL' ? (
                        <Hotel className="w-4 h-4" />
                      ) : (
                        <MapPin className="w-4 h-4" />
                      )}
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-slate-100">{seg.title}</h3>
                      <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
                        <MapPin className="w-3 h-3 text-slate-500" />
                        <span>{seg.location.name}</span>
                        {seg.location.iata_code && (
                          <span className="font-mono text-blue-400 font-medium">({seg.location.iata_code})</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                      seg.is_confirmed
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                        : 'bg-amber-950 text-amber-300 border border-amber-800'
                    }`}
                  >
                    {seg.is_confirmed ? 'Confirmed' : 'Pending'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-slate-800/60 text-xs text-slate-300">
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-slate-500" />
                    <span>
                      {new Date(seg.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <div className="flex items-center justify-end gap-1.5 font-medium text-slate-200">
                    <DollarSign className="w-3.5 h-3.5 text-emerald-400" />
                    <span>{seg.cost > 0 ? `${seg.cost.toFixed(2)} ${seg.currency}` : 'Included'}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Right Column: Conversational Stream & HITL Approvals */}
      <div className="lg:col-span-7 flex flex-col bg-slate-900/60 backdrop-blur border border-slate-800 rounded-2xl p-5 overflow-hidden shadow-xl">
        {/* Header with Connection Status & Active Execution Indicator */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div
              className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}
            />
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Operations Assistant</h2>
              <p className="text-xs text-slate-400 font-mono">
                {isConnected ? 'LIVE WEBSOCKET STREAMING' : 'HTTP REST FALLBACK ACTIVE'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {activeToolCall && (
              <div className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-amber-950/80 border border-amber-800/70 text-xs text-amber-300 shadow-sm animate-pulse">
                <Wrench className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                <span className="font-mono font-medium">Tool: {activeToolCall.tool}</span>
              </div>
            )}

            {activeWorkerNode && !activeToolCall && (
              <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-blue-950/80 border border-blue-800/60 text-xs text-blue-300 shadow-sm">
                <Radio className="w-3.5 h-3.5 animate-spin text-blue-400" />
                <span>Executing: {activeWorkerNode}</span>
              </div>
            )}
          </div>
        </div>

        {/* Error Boundary UI Card */}
        {streamError && (
          <div className="my-3 p-3.5 rounded-xl bg-rose-950/60 border border-rose-800 text-rose-200 flex items-start justify-between gap-3 shadow-md animate-in fade-in">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-semibold text-rose-100">Operational Notice</h4>
                <p className="text-xs text-rose-300/90 mt-0.5 font-mono">{streamError}</p>
              </div>
            </div>
            <button
              onClick={() => setStreamError(null)}
              className="p-1 rounded-md hover:bg-rose-900/60 text-rose-400 hover:text-rose-200 transition-colors"
              title="Dismiss error"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Stream Messages Container */}
        <div className="flex-1 overflow-y-auto py-4 space-y-4 pr-1">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-48 text-center text-slate-400">
              <Sparkles className="w-8 h-8 mb-2 text-blue-400/60" />
              <p className="text-sm font-medium text-slate-300">Welcome to ZICO</p>
              <p className="text-xs text-slate-500 mt-1 max-w-sm">
                Try asking: &quot;Find flights from BOM to DXB next Friday&quot; or &quot;What are the EU261 flight cancellation rules?&quot;
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex flex-col ${
                msg.role === 'user'
                  ? 'items-end'
                  : msg.role === 'system'
                  ? 'items-center'
                  : 'items-start'
              }`}
            >
              {msg.role === 'system' ? (
                <div className="text-xs text-slate-500 bg-slate-800/50 px-3 py-1 rounded-full border border-slate-800">
                  {msg.text}
                </div>
              ) : (
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-none shadow-md'
                      : 'bg-slate-800/80 text-slate-100 border border-slate-700/60 rounded-bl-none'
                  }`}
                >
                  {msg.node && (
                    <div className="text-[10px] font-mono uppercase tracking-wider text-blue-400 mb-1">
                      {msg.node}
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">{msg.text}</p>

                  {/* Clickable Research & Live Sources Badges */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-slate-700/60">
                      <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider block mb-1.5">
                        Verified Sources
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {msg.sources.map((src, sIdx) => (
                          <a
                            key={sIdx}
                            href={src.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-900/80 hover:bg-slate-900 text-blue-300 hover:text-blue-200 border border-slate-700/80 text-[11px] transition-colors"
                          >
                            <ExternalLink className="w-2.5 h-2.5" />
                            <span className="truncate max-w-[180px]">{src.title}</span>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  <span className="text-[10px] opacity-60 block text-right mt-1.5">{msg.time}</span>
                </div>
              )}
            </div>
          ))}

          {/* Active Streaming Token Bubble */}
          {streamingTokenMessage && (
            <div className="flex flex-col items-start animate-in fade-in">
              <div className="max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-slate-800/90 text-slate-100 border border-blue-500/50 rounded-bl-none shadow-lg shadow-blue-900/20">
                <div className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-blue-400 mb-1">
                  <Radio className="w-3 h-3 animate-spin" />
                  <span>Streaming response...</span>
                </div>
                <p className="whitespace-pre-wrap">{streamingTokenMessage}</p>
              </div>
            </div>
          )}

          {/* Dynamic Human-in-the-Loop Interrupt Approval Card */}
          {activeInterrupt && (
            <div className="p-5 rounded-2xl bg-amber-950/40 border border-amber-800/80 shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-amber-200">
                    Human Authorization Required: {activeInterrupt.action_type}
                  </h4>
                  <p className="text-xs text-amber-300/80 mt-1">{activeInterrupt.description}</p>
                  <div className="flex items-center gap-3 mt-4">
                    <button
                      onClick={() => handleDecision(true)}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-md transition-all"
                    >
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Approve & Execute</span>
                    </button>
                    <button
                      onClick={() => handleDecision(false)}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-all"
                    >
                      <XCircle className="w-4 h-4" />
                      <span>Reject Action</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {isProcessing && !activeInterrupt && !streamingTokenMessage && (
            <div className="flex items-center gap-2 text-xs text-slate-400 animate-pulse pl-1">
              <Radio className="w-3.5 h-3.5 animate-spin text-blue-400" />
              <span>ZICO is reasoning across domain agents...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Controls with Voice Transcription Support */}
        <form onSubmit={handleSendPrompt} className="pt-3 border-t border-slate-800 flex items-center gap-2">
          {/* Microphone Recording Button */}
          <button
            type="button"
            onClick={handleToggleVoiceRecording}
            className={`p-2.5 rounded-xl border transition-all ${
              isRecording
                ? 'bg-rose-600 text-white border-rose-500 animate-pulse'
                : 'bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 border-slate-700/80'
            }`}
            title={isRecording ? 'Stop Recording' : 'Voice Input (Whisper)'}
          >
            {isRecording ? <MicOff className="w-4 h-4 text-white" /> : <Mic className="w-4 h-4" />}
          </button>

          {isRecording ? (
            <div className="flex-1 bg-slate-800/80 border border-rose-500/60 rounded-xl px-4 py-2.5 text-sm text-rose-300 flex items-center justify-between animate-pulse">
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-ping" />
                Listening... Speak your travel query
              </span>
              <span className="font-mono text-xs text-rose-400">{recordingSeconds}s</span>
            </div>
          ) : (
            <input
              type="text"
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              placeholder="Ask ZICO (e.g. 'Find flights from Mumbai to Dubai')..."
              className="flex-1 bg-slate-800/80 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              disabled={isProcessing}
            />
          )}

          <button
            type="submit"
            disabled={!promptInput.trim() || isProcessing}
            className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white shadow-md transition-all"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
