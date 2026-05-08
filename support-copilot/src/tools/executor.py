from __future__ import annotations

from src.tools import tools
from src.tools.schema import trace
from src.tools.schema_loader import validate_tool_arguments


class ToolExecutor:
    def __init__(self) -> None:
        self.traces: list[dict] = []

    def call(self, name: str, **arguments) -> dict:
        validate_tool_arguments(name, arguments)
        fn = {
            "RouteDomain": tools.route_domain,
            "SearchKB": tools.search_kb,
            "GetPolicy": tools.get_policy,
            "CreateTicket": tools.create_ticket,
            "RejectQuery": tools.reject_query,
        }[name]
        result = fn(**arguments)
        self.traces.append(trace(name, arguments, result))
        return result
