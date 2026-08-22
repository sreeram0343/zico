'use client';

import React, { useState, useEffect, useRef } from 'react';
import { StreamEvent, InterruptEvent, TripSegment } from '@/types';
import { Plane, Hotel, Calendar, AlertTriangle, ShieldCheck, CheckCircle2, XCircle, Send, Mic, Radio, Clock, MapPin, DollarSign } from 'lucide-react';

interface TimelineProps {
  tripId: string;
  userId?: string;
  wsBaseUrl?: string;
}

export function Timeline({
  tripId,
  userId = 'traveler_01',
  wsBaseUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000',
}: TimelineProps) {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant' | 'system'; text: string; node?: string; time: string }>>([]);
  const [activeInterrupt, setActiveInterrupt] = useState<InterruptEvent | null>(null);
  const [itinerary, setItinerary] = useState<TripSegment[]>([]);
  const [promptInput, setPromptInput] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [activeWorkerNode, setActiveWorkerNode] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const wsUrl = `${wsBaseUrl.replace(/^http/, 'ws')}/ws/stream/${tripId}`;
    console.log('[CLIENT] Connecting to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[CLIENT] WebSocket connected successfully to:', wsUrl);
      setIsConnected(true);
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: `Connected to ZICO Real-time Orchestration Engine (Trip: ${tripId})`,
          time: new Date().toLocaleTimeString(),
        },
      ]);
    };

    ws.onmessage = (event) => {
      try {
        const streamEvent: StreamEvent = JSON.parse(event.data);
        console.log('[CLIENT] Message received:', streamEvent);

        // 1. Node updates
        if (streamEvent.type === 'node_update') {
          setActiveWorkerNode(streamEvent.node || null);

          const msgText =
            streamEvent.message ||
            streamEvent.content ||
            (streamEvent.output?.messages && streamEvent.output.messages[0]?.content) ||
            '';

          if (msgText) {
            console.log('[CLIENT] Response rendered from node:', streamEvent.node);
            setMessages((prev) => [
              ...prev,
              {
                role: 'assistant',
                text: typeof msgText === 'string' ? msgText : JSON.stringify(msgText),
                node: streamEvent.node,
                time: new Date().toLocaleTimeString(),
              },
            ]);
          }

          // If itinerary was updated in node output
          if (streamEvent.output?.itinerary && Array.isArray(streamEvent.output.itinerary)) {
            console.log('[CLIENT] Itinerary segments received:', streamEvent.output.itinerary.length);
            setItinerary(streamEvent.output.itinerary);
          }
        }

        // 2. Status update
        else if (streamEvent.type === 'status') {
          const statusText = streamEvent.message || streamEvent.content || 'Processing...';
          console.log('[CLIENT] Status update:', statusText);
          setActiveWorkerNode('supervisor');
        }

        // 3. Dynamic Human-in-the-Loop Interrupt
        else if (streamEvent.type === 'interrupt' && streamEvent.interrupt_value) {
          console.log('[CLIENT] Interrupt event received:', streamEvent.interrupt_value);
          setActiveInterrupt(streamEvent.interrupt_value);
          setIsProcessing(false);
        }

        // 4. Audio / Voice playback
        else if (streamEvent.type === 'voice_chunk' && streamEvent.audio_base64) {
          try {
            const audio = new Audio(`data:audio/wav;base64,${streamEvent.audio_base64}`);
            audio.play().catch(() => {});
          } catch (e) {
            console.debug('Audio playback note:', e);
          }
        }

        // 5. Turn complete
        else if (streamEvent.type === 'turn_complete') {
          console.log('[CLIENT] Turn completed for trip:', tripId);
          setIsProcessing(false);
          setActiveWorkerNode(null);
        }

        // 6. Stream error
        else if (streamEvent.type === 'error') {
          const errText = streamEvent.message || streamEvent.content || 'Stream error occurred';
          console.error('[CLIENT] Stream error:', errText);
          setMessages((prev) => [
            ...prev,
            {
              role: 'system',
              text: `Error: ${errText}`,
              time: new Date().toLocaleTimeString(),
            },
          ]);
          setIsProcessing(false);
        }
      } catch (err) {
        console.error('[CLIENT] Failed to parse WebSocket message:', err);
      }
    };

    ws.onclose = (event) => {
      console.log('[CLIENT] WebSocket closed:', event.code, event.reason);
      setIsConnected(false);
      setActiveWorkerNode(null);
    };

    ws.onerror = (err) => {
      console.error('[CLIENT] WebSocket error:', err);
      setIsConnected(false);
    };

    setSocket(ws);

    return () => {
      console.log('[CLIENT] Closing WebSocket connection');
      ws.close();
    };
  }, [tripId, wsBaseUrl]);

  // Auto-scroll messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeInterrupt]);

  // Send message over WebSocket
  const handleSendPrompt = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!promptInput.trim() || !socket || !isConnected) return;

    const userText = promptInput.trim();
    console.log('[CLIENT] Sending message:', userText);

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: userText, time: new Date().toLocaleTimeString() },
    ]);

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

    setPromptInput('');
    setIsProcessing(true);
  };

  // Human-in-the-Loop decision submission
  const handleDecision = (approved: boolean) => {
    if (!activeInterrupt || !socket || !isConnected) return;

    const payload = {
      type: 'decision',
      approved,
      action_id: activeInterrupt.action_id,
      actor: userId,
      trip_id: tripId,
    };

    console.log('[CLIENT] Submitting HITL decision:', payload);
    socket.send(JSON.stringify(payload));

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
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-5rem)]">
      {/* Left Column: Itinerary Timeline */}
      <div className="lg:col-span-5 flex flex-col bg-slate-900/60 backdrop-blur border border-slate-800 rounded-2xl p-5 overflow-hidden">
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
              <p className="text-sm">No scheduled segments yet.</p>
              <p className="text-xs text-slate-500">Ask ZICO to search flights or plan your itinerary.</p>
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
                          <span className="font-mono text-slate-400">({seg.location.iata_code})</span>
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
                    <span>{new Date(seg.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
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
      <div className="lg:col-span-7 flex flex-col bg-slate-900/60 backdrop-blur border border-slate-800 rounded-2xl p-5 overflow-hidden">
        {/* Header with Connection Status */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            <div>
              <h2 className="text-lg font-semibold text-slate-100">Orchestration Stream</h2>
              <p className="text-xs text-slate-400 font-mono">
                {isConnected ? 'LIVE WEBSOCKET STREAMING' : 'DISCONNECTED'}
              </p>
            </div>
          </div>
          {activeWorkerNode && (
            <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-blue-950/80 border border-blue-800/60 text-xs text-blue-300">
              <Radio className="w-3.5 h-3.5 animate-spin text-blue-400" />
              <span>Executing: {activeWorkerNode}</span>
            </div>
          )}
        </div>

        {/* Stream Messages Container */}
        <div className="flex-1 overflow-y-auto py-4 space-y-4 pr-1">
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
                  <span className="text-[10px] opacity-60 block text-right mt-1.5">{msg.time}</span>
                </div>
              )}
            </div>
          ))}

          {/* Dynamic Human-in-the-Loop Interrupt Approval Card */}
          {activeInterrupt && (
            <div className="p-5 rounded-2xl bg-amber-950/40 border border-amber-800/80 shadow-lg animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h4 className="text-sm font-semibold text-amber-200">
                    Human Authorization Required: {activeInterrupt.action_type}
                  </h4>
                  <p className="text-xs text-amber-300/80 mt-1">
                    {activeInterrupt.description}
                  </p>
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

          {isProcessing && !activeInterrupt && (
            <div className="flex items-center gap-2 text-xs text-slate-400 animate-pulse pl-1">
              <Radio className="w-3.5 h-3.5 animate-spin text-blue-400" />
              <span>ZICO is reasoning across agents...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Controls */}
        <form onSubmit={handleSendPrompt} className="pt-3 border-t border-slate-800 flex items-center gap-2">
          <input
            type="text"
            value={promptInput}
            onChange={(e) => setPromptInput(e.target.value)}
            placeholder="Ask ZICO (e.g. 'Search flights from Mumbai to Pune')..."
            className="flex-1 bg-slate-800/80 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
            disabled={!isConnected || isProcessing}
          />
          <button
            type="submit"
            disabled={!isConnected || !promptInput.trim() || isProcessing}
            className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white shadow-md transition-all"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
