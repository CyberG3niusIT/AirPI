from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from core.policy import PolicyDecision, Action

class BackendType(str, Enum):
    LOCAL = "local"
    EXTERNAL = "external"
    NONE = "none"

class RouteResult(BaseModel):
    backend: BackendType
    model_name: Optional[str] = None
    blocked: bool = False
    reason: str = ""

class ModelRouter:
    def __init__(self, default_local_model: str = "default-local-model"):
        self.default_local_model = default_local_model

    def route(self, decision: PolicyDecision, request_data: Dict[str, Any]) -> RouteResult:
        if decision.action == Action.BLOCK:
            return RouteResult(backend=BackendType.NONE, blocked=True, reason=decision.reason)

        if decision.action == Action.REQUIRE_APPROVAL:
            return RouteResult(backend=BackendType.NONE, blocked=True, reason=f"Pending Approval: {decision.reason}")

        if decision.action == Action.ALLOW_EXTERNAL:
            # Future: resolve specific external model
            return RouteResult(backend=BackendType.EXTERNAL, model_name=request_data.get("model", "default-external"), reason=decision.reason)

        if decision.action == Action.ALLOW_LOCAL:
            return RouteResult(backend=BackendType.LOCAL, model_name=request_data.get("model", self.default_local_model), reason=decision.reason)

        return RouteResult(backend=BackendType.NONE, blocked=True, reason="Unknown policy action")
