from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Action(str, Enum):
    ALLOW_LOCAL = "allow_local"
    ALLOW_EXTERNAL = "allow_external"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"

class PolicyDecision(BaseModel):
    action: Action
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PolicyEngine:
    def __init__(self, default_action: Action = Action.ALLOW_LOCAL):
        self.default_action = default_action

    def evaluate(self, request_data: Dict[str, Any]) -> PolicyDecision:
        # Minimal skeleton: Check if 'block' is explicitly in a simulated tag
        if request_data.get("tags", []) and "malicious" in request_data["tags"]:
            return PolicyDecision(action=Action.BLOCK, reason="Blocked by malicious tag")

        if request_data.get("requires_external", False):
            return PolicyDecision(action=Action.ALLOW_EXTERNAL, reason="External backend requested")

        if request_data.get("sensitive", False):
            return PolicyDecision(action=Action.REQUIRE_APPROVAL, reason="Sensitive data requires manual approval")

        return PolicyDecision(action=self.default_action, reason="Default policy applied")
