import asyncio
import base64
from datetime import date, datetime
from enum import Enum
import json
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.graph.engine import graph_engine
from app.services.voice import get_voice_service

logger = logging.getLogger(__name__)
router = APIRouter()

KNOWN_GRAPH_NODES = {
    "input_node",
    "supervisor_node",
    "flight_search_worker",
    "policy_rag_worker",
    "disruption_worker",
    "booking_approval_node",
    "validator_node",
}


def _safe_serialize(obj: Any) -> Any:
    """
    Recursively and safely serializes Pydantic models, datetimes, Enums,
    and arbitrary objects into JSON-compliant data structures.
    """
    if obj is None:
        return None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, (int, float, bool, str)):
        return obj
    return str(obj)


async def _safe_send_json(websocket: WebSocket, payload: Dict[str, Any]) -> None:
    """Safely serializes and sends a JSON frame over the WebSocket."""
    serialized = _safe_serialize(payload)
    text = json.dumps(serialized, default=str)
    await websocket.send_text(text)


@router.websocket("/ws/stream/{trip_id}")
async def websocket_stream_endpoint(websocket: WebSocket, trip_id: str):
    """
    Real-time bidirectional WebSocket endpoint streaming LangGraph execution events,
    dynamic interrupts/HITL approval checkpoints, token streams, tool invocations,
    and visual state mutations.
    """
    await websocket.accept()
    print(f"[WS] Connected client for trip_id: {trip_id}", flush=True)
    logger.info(f"WebSocket client connected for trip session: {trip_id}")

    thread_config = {"configurable": {"thread_id": trip_id}}
    voice_service = get_voice_service()

    # -----------------------------------------------------------------------
    # Global WebSocket Try/Except Wrapping the Receive Loop
    # -----------------------------------------------------------------------
    try:
        while True:
            # 1. Receive incoming message from client
            try:
                raw_data = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected for trip: {trip_id}")
                break

            print(f"[WS RX] Raw received payload: {raw_data}", flush=True)

            # 2. Strict JSON Parsing with immediate error frame push
            try:
                payload = json.loads(raw_data)
            except Exception as json_err:
                print(f"[WS ERROR] JSON parse error: {json_err}", flush=True)
                await _safe_send_json(websocket, {
                    "type": "error",
                    "message": f"Invalid JSON payload: {str(json_err)}",
                    "content": f"Invalid JSON payload: {str(json_err)}",
                })
                continue

            # 3. Process the incoming request in an isolated try/except block
            try:
                msg_type = payload.get("type", "prompt")
                user_id = payload.get("user_id", "default_traveler")
                active_trip_id = payload.get("trip_id") or trip_id
                user_content = (
                    payload.get("content")
                    or payload.get("message")
                    or payload.get("text")
                    or payload.get("query")
                    or ""
                )

                # ---------------------------------------------------------
                # Case 1: Audio Input -> Transcribe first then run Graph
                # ---------------------------------------------------------
                if msg_type == "voice_input":
                    audio_b64 = payload.get("audio_base64", "")
                    if audio_b64:
                        try:
                            audio_bytes = base64.b64decode(audio_b64)
                            transcript = await voice_service.transcribe_audio(audio_bytes)
                            await _safe_send_json(websocket, {
                                "type": "transcript",
                                "text": transcript,
                                "content": transcript,
                            })
                            input_query = transcript
                        except Exception as exc:
                            print(f"[WS ERROR] Voice transcription error: {exc}", file=sys.stderr, flush=True)
                            logger.error(f"Voice transcription failed: {exc}")
                            await _safe_send_json(websocket, {
                                "type": "error",
                                "message": f"Speech transcription error: {str(exc)}",
                                "content": f"Speech transcription error: {str(exc)}",
                            })
                            continue
                    else:
                        input_query = user_content

                    target_input = {
                        "messages": [HumanMessage(content=input_query)],
                        "trip_id": active_trip_id,
                        "user_id": user_id,
                    }

                # ---------------------------------------------------------
                # Case 2: Resume from Human-in-the-Loop Decision Command
                # ---------------------------------------------------------
                elif msg_type == "decision":
                    approved = bool(payload.get("approved", False))
                    actor = payload.get("actor", user_id)
                    action_id = payload.get("action_id", "")

                    resume_payload = {
                        "approved": approved,
                        "actor": actor,
                        "action_id": action_id,
                    }
                    target_input = Command(resume=resume_payload)
                    input_query = None

                # ---------------------------------------------------------
                # Case 3: Standard User Prompt
                # ---------------------------------------------------------
                else:
                    input_query = user_content
                    target_input = {
                        "messages": [HumanMessage(content=input_query)],
                        "trip_id": active_trip_id,
                        "user_id": user_id,
                    }

                print(f"[WS GRAPH] Starting stream for query: {input_query!r}", flush=True)

                # Send initial status feedback
                await _safe_send_json(websocket, {
                    "type": "status",
                    "status": "processing",
                    "node": "input_node",
                    "content": "Thinking...",
                    "message": f"Orchestrating request: '{input_query}'",
                })

                # ---------------------------------------------------------
                # Execute LangGraph Streaming via astream_events(..., version="v2")
                # ---------------------------------------------------------
                accumulated_ai_text = ""
                last_streamed_itinerary_count = -1

                async for event in graph_engine.astream_events(
                    target_input,
                    config=thread_config,
                    version="v2",
                ):
                    event_type = event.get("event", "")
                    event_name = event.get("name", "")
                    event_data = event.get("data", {})
                    event_metadata = event.get("metadata", {})
                    langgraph_node = event_metadata.get("langgraph_node", "")

                    # 1. Node Start -> emit "status" frame
                    if event_type == "on_chain_start":
                        node_candidate = langgraph_node or event_name
                        if node_candidate in KNOWN_GRAPH_NODES or langgraph_node:
                            await _safe_send_json(websocket, {
                                "type": "status",
                                "node": node_candidate,
                                "content": "Thinking...",
                                "message": "Thinking...",
                            })

                    # 2. Tool Invocation -> emit "tool_call" frame
                    elif event_type == "on_tool_start":
                        tool_input = event_data.get("input", event_data)
                        await _safe_send_json(websocket, {
                            "type": "tool_call",
                            "tool": event_name,
                            "input": _safe_serialize(tool_input),
                        })

                    # 3. Chat Model Token Stream -> emit "token" frame
                    elif event_type == "on_chat_model_stream":
                        chunk_obj = event_data.get("chunk")
                        chunk_text = ""
                        if hasattr(chunk_obj, "content"):
                            c = chunk_obj.content
                            if isinstance(c, str):
                                chunk_text = c
                            elif isinstance(c, list):
                                chunk_text = "".join(
                                    item.get("text", "") if isinstance(item, dict) else str(item)
                                    for item in c
                                )
                        elif isinstance(chunk_obj, str):
                            chunk_text = chunk_obj

                        if chunk_text:
                            accumulated_ai_text += chunk_text
                            await _safe_send_json(websocket, {
                                "type": "token",
                                "content": chunk_text,
                            })

                    # 4. Node End / Chain End -> check state mutation & emit "state_update"
                    elif event_type == "on_chain_end":
                        node_candidate = langgraph_node or event_name
                        if node_candidate in KNOWN_GRAPH_NODES or langgraph_node:
                            # Fetch current graph state snapshot
                            try:
                                current_state = graph_engine.get_state(thread_config)
                                state_values = current_state.values if current_state else {}
                                itinerary = state_values.get("itinerary", [])
                                serialized_itinerary = _safe_serialize(itinerary)

                                # Push state_update on itinerary change or completion
                                if len(itinerary) != last_streamed_itinerary_count or itinerary:
                                    last_streamed_itinerary_count = len(itinerary)
                                    await _safe_send_json(websocket, {
                                        "type": "state_update",
                                        "itinerary": serialized_itinerary,
                                    })

                                # Extract message content from node output if any
                                node_output = event_data.get("output", {})
                                msg_snippet = ""
                                if isinstance(node_output, dict) and "messages" in node_output:
                                    for m in node_output["messages"]:
                                        if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
                                            msg_snippet = m.content if isinstance(m.content, str) else str(m.content)

                                # Also emit node_update for legacy UI/test compatibility
                                await _safe_send_json(websocket, {
                                    "type": "node_update",
                                    "node": node_candidate,
                                    "output": _safe_serialize(node_output),
                                    "content": msg_snippet or "Node completed",
                                    "message": msg_snippet or "Node completed",
                                })
                            except Exception as state_exc:
                                logger.debug(f"Notice getting state in on_chain_end: {state_exc}")

                # Check if graph paused on dynamic interrupt
                try:
                    current_state = graph_engine.get_state(thread_config)
                    if current_state and current_state.tasks and any(len(t.interrupts) > 0 for t in current_state.tasks):
                        for task in current_state.tasks:
                            for inter in task.interrupts:
                                prompt_msg = (
                                    inter.value.get("prompt", "Traveler approval required")
                                    if isinstance(inter.value, dict)
                                    else "Approval required"
                                )
                                await _safe_send_json(websocket, {
                                    "type": "interrupt",
                                    "node": task.name,
                                    "interrupt_value": _safe_serialize(inter.value),
                                    "prompt": prompt_msg,
                                    "content": prompt_msg,
                                })
                except Exception as interrupt_exc:
                    logger.debug(f"Notice inspecting graph interrupts: {interrupt_exc}")

                # Optional TTS generation if requested
                if accumulated_ai_text and payload.get("enable_tts", False):
                    try:
                        audio_bytes = await voice_service.synthesize_speech(accumulated_ai_text)
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        await _safe_send_json(websocket, {
                            "type": "voice_chunk",
                            "audio_base64": audio_b64,
                        })
                    except Exception as tts_exc:
                        logger.debug(f"TTS streaming notice: {tts_exc}")

                # Signal turn completion
                await _safe_send_json(websocket, {
                    "type": "turn_complete",
                    "trip_id": active_trip_id,
                })
                print(f"[WS SUCCESS] Completed stream turn for trip: {active_trip_id}", flush=True)

            except WebSocketDisconnect:
                print(f"[WS DISCONNECT] Client disconnected mid-stream for trip: {trip_id}", flush=True)
                break
            except Exception as loop_err:
                print(f"[WS ERROR] Error in stream processing: {loop_err}", file=sys.stderr, flush=True)
                traceback.print_exc()
                logger.error(f"Error during graph execution stream: {loop_err}", exc_info=True)
                try:
                    await _safe_send_json(websocket, {
                        "type": "error",
                        "message": str(loop_err),
                        "content": str(loop_err),
                    })
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[WS CLOSED] Connection closed cleanly for trip: {trip_id}", flush=True)
    except Exception as global_err:
        print(f"[WS FATAL] Global WebSocket handler exception: {global_err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        logger.error(f"Global WebSocket handler exception: {global_err}", exc_info=True)
        try:
            await _safe_send_json(websocket, {
                "type": "error",
                "message": str(global_err),
                "content": str(global_err),
            })
        except Exception:
            pass
