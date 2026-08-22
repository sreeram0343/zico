import asyncio
import base64
import json
import logging
import sys
import traceback
from typing import Any, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.graph.engine import graph_engine
from app.services.voice import get_voice_service

logger = logging.getLogger(__name__)
router = APIRouter()


def _serialize_graph_event(event: Any) -> Dict[str, Any]:
    """Helper to safely serialize LangGraph streaming events and Pydantic models to JSON."""
    if isinstance(event, dict):
        serialized: Dict[str, Any] = {}
        for k, v in event.items():
            if hasattr(v, "model_dump"):
                serialized[k] = v.model_dump()
            elif isinstance(v, list):
                serialized[k] = [
                    item.model_dump() if hasattr(item, "model_dump") else str(item)
                    for item in v
                ]
            else:
                serialized[k] = str(v) if not isinstance(v, (int, float, bool)) else v
        return serialized
    elif hasattr(event, "model_dump"):
        return event.model_dump()
    return {"data": str(event)}


@router.websocket("/ws/stream/{trip_id}")
async def websocket_stream_endpoint(websocket: WebSocket, trip_id: str):
    """
    Real-time bidirectional WebSocket endpoint streaming LangGraph execution events,
    dynamic interrupt/HITL approval checkpoints, and voice/TTS payloads.
    """
    await websocket.accept()
    print(f"[WS] Connected client for trip_id: {trip_id}", flush=True)
    logger.info(f"WebSocket client connected for trip session: {trip_id}")

    thread_config = {"configurable": {"thread_id": trip_id}}
    voice_service = get_voice_service()

    try:
        while True:
            # 1. Receive incoming message from client
            raw_data = await websocket.receive_text()
            print(f"[WS RX] Raw received payload: {raw_data}", flush=True)

            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError as json_err:
                print(f"[WS ERROR] JSON parse error: {json_err}", flush=True)
                await websocket.send_json({
                    "type": "error",
                    "content": f"Invalid JSON payload: {str(json_err)}",
                    "message": f"Invalid JSON payload: {str(json_err)}",
                })
                continue

            # Extract fields flexibly supporting all naming conventions
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

            # -------------------------------------------------------------
            # Case 1: Audio Input -> Transcribe first then run Graph
            # -------------------------------------------------------------
            if msg_type == "voice_input":
                audio_b64 = payload.get("audio_base64", "")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        transcript = await voice_service.transcribe_audio(audio_bytes)
                        await websocket.send_json({
                            "type": "transcript",
                            "text": transcript,
                            "content": transcript,
                        })
                        input_query = transcript
                    except Exception as exc:
                        print(f"[WS ERROR] Voice transcription error: {exc}", file=sys.stderr, flush=True)
                        logger.error(f"Voice transcription failed: {exc}")
                        await websocket.send_json({
                            "type": "error",
                            "content": f"Speech transcription error: {str(exc)}",
                            "message": f"Speech transcription error: {str(exc)}",
                        })
                        continue
                else:
                    input_query = user_content

            # -------------------------------------------------------------
            # Case 2: Resume from Human-in-the-Loop Decision Command
            # -------------------------------------------------------------
            elif msg_type == "decision":
                approved = bool(payload.get("approved", False))
                actor = payload.get("actor", user_id)
                action_id = payload.get("action_id", "")

                resume_payload = {
                    "approved": approved,
                    "actor": actor,
                    "action_id": action_id,
                }
                graph_input = Command(resume=resume_payload)
                input_query = None

            # -------------------------------------------------------------
            # Case 3: Standard User Prompt
            # -------------------------------------------------------------
            else:
                input_query = user_content
                graph_input = {
                    "messages": [HumanMessage(content=input_query)],
                    "trip_id": active_trip_id,
                    "user_id": user_id,
                }

            # -------------------------------------------------------------
            # Execute Streaming over LangGraph with Robust Exception Handling
            # -------------------------------------------------------------
            try:
                target_input = graph_input if input_query is None else {
                    "messages": [HumanMessage(content=input_query)],
                    "trip_id": active_trip_id,
                    "user_id": user_id,
                }

                print(f"[WS GRAPH] Starting stream for query: {input_query!r}", flush=True)

                # Send initial status feedback to client
                await websocket.send_json({
                    "type": "status",
                    "status": "processing",
                    "content": f"Orchestrating request: '{input_query}'",
                    "message": f"Orchestrating request: '{input_query}'",
                })

                async for event in graph_engine.astream(
                    target_input,
                    thread_config,
                    stream_mode="updates",
                ):
                    print(f"[WS EVENT] Event from nodes: {list(event.keys())}", flush=True)

                    for node_name, node_output in event.items():
                        msg_snippet = ""
                        if isinstance(node_output, dict) and "messages" in node_output:
                            for m in node_output["messages"]:
                                if isinstance(m, AIMessage) or getattr(m, "type", "") == "ai":
                                    msg_snippet = m.content if isinstance(m.content, str) else str(m.content)

                        # Emit serialized node update to client
                        serialized_output = _serialize_graph_event(node_output)
                        await websocket.send_json({
                            "type": "node_update",
                            "node": node_name,
                            "output": serialized_output,
                            "content": msg_snippet,
                            "message": msg_snippet,
                        })

                        # If an AI message was produced, synthesize speech chunk if requested
                        if msg_snippet and payload.get("enable_tts", False):
                            try:
                                audio_bytes = await voice_service.synthesize_speech(msg_snippet)
                                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                await websocket.send_json({
                                    "type": "voice_chunk",
                                    "node": node_name,
                                    "audio_base64": audio_b64,
                                })
                            except Exception as tts_exc:
                                logger.debug(f"TTS streaming notice: {tts_exc}")

                # Check if graph paused on dynamic interrupt
                current_state = graph_engine.get_state(thread_config)
                if current_state.tasks and any(len(t.interrupts) > 0 for t in current_state.tasks):
                    for task in current_state.tasks:
                        for inter in task.interrupts:
                            prompt_msg = (
                                inter.value.get("prompt", "Traveler approval required")
                                if isinstance(inter.value, dict)
                                else "Approval required"
                            )
                            await websocket.send_json({
                                "type": "interrupt",
                                "node": task.name,
                                "interrupt_value": inter.value,
                                "prompt": prompt_msg,
                                "content": prompt_msg,
                            })

                # Signal turn completion
                await websocket.send_json({
                    "type": "turn_complete",
                    "trip_id": active_trip_id,
                })
                print(f"[WS SUCCESS] Completed stream turn for trip: {active_trip_id}", flush=True)

            except WebSocketDisconnect:
                print(f"[WS DISCONNECT] Client disconnected mid-stream for trip: {active_trip_id}", flush=True)
                break
            except Exception as e:
                # Log full error traceback to stdout / stderr
                print(f"[WS ERROR] Error in LangGraph execution stream: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
                logger.error(f"Error during graph execution stream: {e}", exc_info=True)
                # Send explicit error message to client
                await websocket.send_json({
                    "type": "error",
                    "content": str(e),
                    "message": str(e),
                })

    except WebSocketDisconnect:
        print(f"[WS CLOSED] Connection closed cleanly for trip: {trip_id}", flush=True)
    except Exception as exc:
        print(f"[WS FATAL] Unexpected WebSocket handler exception: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        logger.error(f"Unexpected WebSocket handler exception: {exc}")
