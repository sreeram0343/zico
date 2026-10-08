import base64
import json
import logging
import sys
import traceback
import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.core.exceptions import ProviderError, ZicoError, sanitize_error_message
from app.core.request_context import (
    clear_request_context,
    get_request_id,
    get_session_id,
    set_request_context,
)
from app.graph.engine import graph_engine
from app.services.voice import get_voice_service

logger = logging.getLogger(__name__)
router = APIRouter()

KNOWN_GRAPH_NODES = {
    "input_node",
    "supervisor_node",
    "flight_search_worker",
    "research_worker",
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
                await _safe_send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": f"Invalid JSON payload: {str(json_err)}",
                        "content": f"Invalid JSON payload: {str(json_err)}",
                    },
                )
                continue

            # 3. Process the incoming request in an isolated try/except block
            req_id = payload.get("request_id") or f"req_{uuid.uuid4().hex[:12]}"
            sess_id = payload.get("session_id") or payload.get("trip_id") or trip_id
            set_request_context(request_id=req_id, session_id=sess_id)
            thread_config = {
                "configurable": {"thread_id": trip_id},
                "metadata": {"request_id": req_id, "session_id": sess_id},
            }
            has_emitted_assistant_message = False

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

                is_voice_turn = False
                # ---------------------------------------------------------
                # Case 1: Audio Input -> Transcribe first then run Graph
                # ---------------------------------------------------------
                if msg_type == "voice_input":
                    is_voice_turn = True
                    audio_b64 = payload.get("audio_base64", "")
                    if audio_b64:
                        try:
                            logger.info(
                                "transcription_started transport=ws request_id=%s session_id=%s",
                                req_id,
                                sess_id,
                            )
                            audio_bytes = base64.b64decode(audio_b64)
                            transcript = await voice_service.transcribe_audio(
                                audio_bytes, filename="voice_input.webm"
                            )
                            logger.info(
                                "transcription_completed transport=ws request_id=%s session_id=%s",
                                req_id,
                                sess_id,
                            )
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "transcript",
                                    "text": transcript,
                                    "content": transcript,
                                    "request_id": req_id,
                                    "session_id": sess_id,
                                },
                            )
                            input_query = transcript
                            logger.info(
                                "chat_from_voice_started transport=ws request_id=%s session_id=%s",
                                req_id,
                                sess_id,
                            )
                        except Exception as exc:
                            logger.error(
                                "transcription_failed transport=ws request_id=%s session_id=%s error=%s",
                                req_id,
                                sess_id,
                                exc,
                            )
                            err_str = str(exc).lower()
                            if any(
                                k in err_str
                                for k in [
                                    "quota",
                                    "429",
                                    "insufficient_quota",
                                    "credit_balance_exhausted",
                                    "rate_limit",
                                ]
                            ):
                                safe_voice_msg = "ZICO voice transcription is temporarily unavailable because the AI service quota is exhausted. Please type your query or try again later."
                                status_code = 429
                            elif any(k in err_str for k in ["timeout", "timed out"]):
                                safe_voice_msg = "Voice transcription timed out while processing audio. Please try again."
                                status_code = 504
                            else:
                                safe_voice_msg = "Voice transcription failed. Please try typing your request or try again."
                                status_code = 500

                            print(
                                f"[WS ERROR] Voice transcription error: {sanitize_error_message(str(exc))}",
                                file=sys.stderr,
                                flush=True,
                            )
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "error",
                                    "error_code": "ZICO_VOICE_ERROR",
                                    "status_code": status_code,
                                    "message": safe_voice_msg,
                                    "content": safe_voice_msg,
                                    "request_id": req_id,
                                    "session_id": sess_id,
                                    "trip_id": active_trip_id,
                                },
                            )
                            continue
                    else:
                        input_query = user_content

                    target_input = {
                        "messages": [HumanMessage(content=input_query)],
                        "trip_id": active_trip_id,
                        "user_id": user_id,
                        "request_id": req_id,
                        "session_id": sess_id,
                        "flight_search_results": {},
                        "quick_actions": [],
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
                    if payload.get("is_voice"):
                        is_voice_turn = True
                        logger.info(
                            "chat_from_voice_started transport=ws request_id=%s session_id=%s",
                            req_id,
                            sess_id,
                        )
                    input_query = user_content
                    target_input = {
                        "messages": [HumanMessage(content=input_query)],
                        "trip_id": active_trip_id,
                        "user_id": user_id,
                        "request_id": req_id,
                        "session_id": sess_id,
                        "flight_search_results": {},
                        "quick_actions": [],
                    }

                print(f"[WS GRAPH] Starting stream for query: {input_query!r} (request_id={req_id})", flush=True)

                # Send initial status feedback
                await _safe_send_json(
                    websocket,
                    {
                        "type": "status",
                        "status": "processing",
                        "node": "input_node",
                        "content": "Thinking...",
                        "message": f"Orchestrating request: '{input_query}'",
                        "request_id": req_id,
                        "session_id": sess_id,
                    },
                )

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
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "status",
                                    "node": node_candidate,
                                    "content": "Thinking...",
                                    "message": "Thinking...",
                                    "request_id": req_id,
                                    "session_id": sess_id,
                                },
                            )

                    # 2. Tool Invocation -> emit "tool_call" frame
                    elif event_type == "on_tool_start":
                        tool_input = event_data.get("input", event_data)
                        await _safe_send_json(
                            websocket,
                            {
                                "type": "tool_call",
                                "tool": event_name,
                                "input": _safe_serialize(tool_input),
                                "request_id": req_id,
                                "session_id": sess_id,
                            },
                        )

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
                            has_emitted_assistant_message = True
                            await _safe_send_json(
                                websocket,
                                {
                                    "type": "token",
                                    "content": chunk_text,
                                    "request_id": req_id,
                                    "session_id": sess_id,
                                },
                            )

                    # 4. Node End / Chain End -> check state mutation & emit "state_update"
                    elif event_type == "on_chain_end":
                        node_candidate = langgraph_node or event_name
                        if node_candidate in KNOWN_GRAPH_NODES or langgraph_node:
                            # Fetch current graph state snapshot
                            try:
                                current_state = graph_engine.get_state(thread_config)
                                state_values = current_state.values if current_state else {}
                                itinerary = state_values.get("itinerary", [])
                                flight_search_results = state_values.get("flight_search_results", {})
                                quick_actions = state_values.get("quick_actions", [])
                                serialized_itinerary = _safe_serialize(itinerary)
                                serialized_flight_search_results = _safe_serialize(flight_search_results)
                                serialized_quick_actions = _safe_serialize(quick_actions)

                                # Push state_update on itinerary change or completion
                                if len(itinerary) != last_streamed_itinerary_count or itinerary:
                                    last_streamed_itinerary_count = len(itinerary)
                                    await _safe_send_json(
                                        websocket,
                                        {
                                            "type": "state_update",
                                            "itinerary": serialized_itinerary,
                                            "request_id": req_id,
                                            "session_id": sess_id,
                                        },
                                    )

                                # Extract message content from node output if any
                                node_output = event_data.get("output", {})
                                msg_snippet = ""
                                if isinstance(node_output, dict) and "messages" in node_output:
                                    for m in node_output["messages"]:
                                        if (
                                            isinstance(m, AIMessage)
                                            or getattr(m, "type", "") == "ai"
                                        ):
                                            msg_snippet = (
                                                m.content
                                                if isinstance(m.content, str)
                                                else str(m.content)
                                            )
                                if msg_snippet:
                                    has_emitted_assistant_message = True

                                # Extract flight_search_results and quick_actions from the node output if available
                                output_flights = None
                                output_actions = None
                                if isinstance(node_output, dict):
                                    output_flights = node_output.get("flight_search_results")
                                    output_actions = node_output.get("quick_actions")

                                current_flight_results = (
                                    _safe_serialize(output_flights)
                                    if output_flights is not None
                                    else (_safe_serialize(flight_search_results) if node_candidate == "flight_search_worker" else None)
                                )
                                current_quick_actions = (
                                    _safe_serialize(output_actions)
                                    if output_actions is not None
                                    else _safe_serialize(quick_actions)
                                )

                                # Also emit node_update for state synchronization without fake message content
                                await _safe_send_json(
                                    websocket,
                                    {
                                        "type": "node_update",
                                        "node": node_candidate,
                                        "output": _safe_serialize(node_output),
                                        "content": msg_snippet,
                                        "message": msg_snippet,
                                        "flight_search_results": current_flight_results,
                                        "quick_actions": current_quick_actions,
                                        "request_id": req_id,
                                        "session_id": sess_id,
                                    },
                                )
                            except Exception as state_exc:
                                logger.debug(f"Notice getting state in on_chain_end: {state_exc}")

                # Check if graph paused on dynamic interrupt
                try:
                    current_state = graph_engine.get_state(thread_config)
                    if (
                        current_state
                        and current_state.tasks
                        and any(len(t.interrupts) > 0 for t in current_state.tasks)
                    ):
                        for task in current_state.tasks:
                            for inter in task.interrupts:
                                prompt_msg = (
                                    inter.value.get("prompt", "Traveler approval required")
                                    if isinstance(inter.value, dict)
                                    else "Approval required"
                                )
                                await _safe_send_json(
                                    websocket,
                                    {
                                        "type": "interrupt",
                                        "node": task.name,
                                        "interrupt_value": _safe_serialize(inter.value),
                                        "prompt": prompt_msg,
                                        "content": prompt_msg,
                                        "request_id": req_id,
                                        "session_id": sess_id,
                                    },
                                )
                except Exception as interrupt_exc:
                    logger.debug(f"Notice inspecting graph interrupts: {interrupt_exc}")

                # Optional TTS generation if requested
                if accumulated_ai_text and payload.get("enable_tts", False):
                    try:
                        audio_bytes = await voice_service.synthesize_speech(accumulated_ai_text)
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        await _safe_send_json(
                            websocket,
                            {
                                "type": "voice_chunk",
                                "audio_base64": audio_b64,
                                "request_id": req_id,
                                "session_id": sess_id,
                            },
                        )
                    except Exception as tts_exc:
                        logger.debug(f"TTS streaming notice: {tts_exc}")

                # Verify assistant response was generated
                if not accumulated_ai_text and not has_emitted_assistant_message:
                    logger.warning(
                        "Stream turn ended with empty assistant response for trip=%s request_id=%s session_id=%s",
                        active_trip_id,
                        req_id,
                        sess_id,
                    )
                    print(
                        f"[WS ERROR] Empty assistant response for trip: {active_trip_id}",
                        file=sys.stderr,
                        flush=True,
                    )
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "error_code": "EMPTY_RESPONSE",
                            "status_code": 500,
                            "message": "ZICO is temporarily unable to process this request because the AI service is unavailable. Please try again later.",
                            "content": "ZICO is temporarily unable to process this request because the AI service is unavailable. Please try again later.",
                            "request_id": req_id,
                            "session_id": sess_id,
                            "trip_id": active_trip_id,
                        },
                    )
                    continue

                if is_voice_turn:
                    logger.info(
                        "chat_from_voice_completed transport=ws request_id=%s session_id=%s",
                        req_id,
                        sess_id,
                    )

                # Signal turn completion
                await _safe_send_json(
                    websocket,
                    {
                        "type": "turn_complete",
                        "trip_id": active_trip_id,
                        "request_id": req_id,
                        "session_id": sess_id,
                    },
                )
                print(f"[WS SUCCESS] Completed stream turn for trip: {active_trip_id}", flush=True)

            except WebSocketDisconnect:
                print(
                    f"[WS DISCONNECT] Client disconnected mid-stream for trip: {trip_id}",
                    flush=True,
                )
                break
            except Exception as loop_err:
                err_str = str(loop_err).lower()
                is_quota = (
                    (isinstance(loop_err, ProviderError) and "quota" in str(loop_err.message).lower())
                    or any(
                        k in err_str
                        for k in [
                            "insufficient_quota",
                            "credit_balance_exhausted",
                            "quota",
                            "429",
                            "rate_limit",
                        ]
                    )
                )
                is_timeout = (
                    (isinstance(loop_err, ProviderError) and "timeout" in str(loop_err.message).lower())
                    or any(k in err_str for k in ["timeout", "timed out"])
                )

                if is_quota:
                    safe_error_msg = (
                        "ZICO is temporarily unable to process this request because the AI service is unavailable. Please try again later."
                    )
                    error_code = "ZICO_PROVIDER_ERROR"
                    status_code = 429
                elif is_timeout:
                    safe_error_msg = (
                        "ZICO request timed out while contacting the AI service. Please try again later."
                    )
                    error_code = "ZICO_TIMEOUT_ERROR"
                    status_code = 504
                elif isinstance(loop_err, ZicoError):
                    safe_error_msg = loop_err.safe_message
                    error_code = loop_err.code
                    status_code = loop_err.http_status_code
                else:
                    safe_error_msg = (
                        "An unexpected error occurred while processing the travel request."
                    )
                    error_code = "ZICO_INTERNAL_ERROR"
                    status_code = 500

                clean_err_desc = sanitize_error_message(str(loop_err))
                print(
                    f"[WS ERROR] Error in stream processing: {clean_err_desc}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.error(
                    "Error during graph execution stream trip=%s request_id=%s session_id=%s error_type=%s: %s",
                    active_trip_id,
                    req_id,
                    sess_id,
                    type(loop_err).__name__,
                    clean_err_desc,
                    exc_info=True,
                )
                try:
                    await _safe_send_json(
                        websocket,
                        {
                            "type": "error",
                            "error_code": error_code,
                            "status_code": status_code,
                            "message": safe_error_msg,
                            "content": safe_error_msg,
                            "request_id": req_id,
                            "session_id": sess_id,
                            "trip_id": active_trip_id,
                        },
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        print(f"[WS CLOSED] Connection closed cleanly for trip: {trip_id}", flush=True)
    except Exception as global_err:
        clean_global_err = sanitize_error_message(str(global_err))
        print(
            f"[WS FATAL] Global WebSocket handler exception: {clean_global_err}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        logger.error(f"Global WebSocket handler exception: {clean_global_err}", exc_info=True)
        try:
            await _safe_send_json(
                websocket,
                {
                    "type": "error",
                    "error_code": "ZICO_INTERNAL_ERROR",
                    "status_code": 500,
                    "message": "An internal server error occurred.",
                    "content": "An internal server error occurred.",
                },
            )
        except Exception:
            pass
    finally:
        clear_request_context()
