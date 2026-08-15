from typing import Any, Dict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.core.config import settings


# 1. Define allowed routing destinations
class RouteDecision(BaseModel):
    next_step: Literal[
        "flight_search_worker",
        "policy_rag_worker",
        "disruption_worker",
        "FINISH",
    ]
    reasoning: str = Field(
        description="Short rationale for choosing this worker or finishing"
    )


# 2. System prompt enforcing role separation
SUPERVISOR_SYSTEM_PROMPT = """You are the orchestrator for ZICO travel companion.
Decide which worker needs to act next based on traveler state and latest message.

Available Workers:
- flight_search_worker: For flight searches, availability, schedules, fares, and flight lookup queries.
- policy_rag_worker: For baggage allowance, cancellation policies, airline rules, and travel policy documents.
- disruption_worker: For handling flight delays, cancellations, rebooking assistance, weather issues, or urgent travel disruptions.
- FINISH: Select FINISH when the conversation has completed, the user's intent is answered, or no further specialized worker action is required.
"""


# 3. Supervisor Node callable
def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the conversation history and routes execution to the appropriate worker node or FINISH.
    """
    # 1. Initialize ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)

    # 2. Bind structured output
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | structured_llm

    # Extract messages from state dictionary or state object
    messages = state["messages"] if isinstance(state, dict) else getattr(state, "messages", state["messages"])

    # 3. Invoke with state["messages"]
    decision: RouteDecision = chain.invoke({"messages": messages})

    # 4. Return update dict: {"next_node": decision.next_step}
    return {"next_node": decision.next_step}
