import pytest
from core.policy import PolicyEngine, Action
from core.router import ModelRouter, BackendType
from core.audit import AuditLogger, AuditRecord

def test_policy_allow_local():
    engine = PolicyEngine()
    decision = engine.evaluate({"prompt": "Hello"})
    assert decision.action == Action.ALLOW_LOCAL
    assert decision.reason == "Default policy applied"

def test_policy_allow_external():
    engine = PolicyEngine()
    decision = engine.evaluate({"requires_external": True})
    assert decision.action == Action.ALLOW_EXTERNAL

def test_policy_block():
    engine = PolicyEngine()
    decision = engine.evaluate({"tags": ["malicious"]})
    assert decision.action == Action.BLOCK

def test_policy_require_approval():
    engine = PolicyEngine()
    decision = engine.evaluate({"sensitive": True})
    assert decision.action == Action.REQUIRE_APPROVAL

def test_router_respects_policy():
    router = ModelRouter()

    # Test Allow Local
    decision_local = PolicyEngine().evaluate({"model": "test-local"})
    route_local = router.route(decision_local, {"model": "test-local"})
    assert route_local.backend == BackendType.LOCAL
    assert route_local.model_name == "test-local"
    assert not route_local.blocked

    # Test Block
    decision_block = PolicyEngine().evaluate({"tags": ["malicious"]})
    route_block = router.route(decision_block, {})
    assert route_block.backend == BackendType.NONE
    assert route_block.blocked
    assert "malicious" in route_block.reason

def test_audit_record_creation():
    logger = AuditLogger()
    record = AuditRecord(
        request_type="chat",
        policy_decision="block",
        backend_selected="none",
        reason="Malicious intent detected"
    )

    log_output = logger.log(record)
    assert "Audit:" in log_output
    assert "Policy=block" in log_output
    assert "Backend=none" in log_output
    assert "Malicious" in log_output
