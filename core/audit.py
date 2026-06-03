from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging

class AuditRecord(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_type: str
    policy_decision: str
    backend_selected: str
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AuditLogger:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("AirPI.Audit")
        else:
            self.logger = logger

    def log(self, record: AuditRecord):
        # Currently, structured logging to memory/stdout.
        # Future: store persistently in sqlite or send to external sink.
        log_msg = f"Audit: [{record.request_type}] Policy={record.policy_decision} Backend={record.backend_selected} Reason='{record.reason}'"
        self.logger.info(log_msg)
        return log_msg
