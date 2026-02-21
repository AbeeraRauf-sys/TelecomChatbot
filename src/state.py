"""Graph state: messages, customer data, and routing."""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Routes: retention -> Problem Solver; cancel -> Processor; tech/billing -> end with message
ROUTE_RETENTION = "retention"
ROUTE_CANCEL = "cancel"
ROUTE_TECH = "tech"
ROUTE_BILLING = "billing"
ROUTE_END = "end"


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_data: dict[str, Any] | None
    next_route: str | None
    _api_time_this_turn: float  # accumulated LLM API time for this invoke (for logging)
