# Implementation Plan: AirPI Local AI Control Plane

This document describes the technical roadmap for evolving AirPI from its current state (a stateful LLM inference server) into a robust Local AI Control Plane.

## 1. Current State (What exists)

- **Local AI Runtime:** A functioning inference layer based on `llama.cpp` (`model_manager.py`, `server.py`).
- **Memory System:** A basic memory persistence, extraction, and knowledge graph system (`memory/` folder).
- **Control Surfaces:** Basic CLI and Web UI implementations.
- **Integrations:** API contract exists for Ollama compatibility.

## 2. Target Architecture (What is documented but not fully implemented)

The target architecture defines several layers that are conceptually defined but need explicit implementation:

- **Policy Engine:** To evaluate requests and determine if they should be executed, blocked, or forwarded.
- **Model Router:** To intelligently route requests to the appropriate backend (local or external) based on policy decisions.
- **Audit Layer:** To record the 'why' and 'where' of every request for transparency and debugging.

## 3. Module Development Plan

### Phase 1: Core Foundation (Current Phase)
**Goal:** Establish the basic structures for Policy, Routing, and Audit.

- `core/policy.py`: Basic `PolicyEngine` to yield `PolicyDecision` (Allow Local, Allow External, Block, Require Approval).
- `core/router.py`: Basic `ModelRouter` to resolve a `PolicyDecision` into a concrete `RouteResult`.
- `core/audit.py`: Basic `AuditLogger` to structure and log the decisions in a standard format.

**Acceptance Criteria:**
- `core` modules exist and are covered by unit tests.
- Core classes use Pydantic for strict data validation.
- No existing functionality is broken.

**Risks:**
- Scope creep. (Mitigation: keep logic minimal, focus on interfaces).

### Phase 2: Integration into Intake
**Goal:** Connect the API endpoints (`server.py`) to the new core modules.

- Refactor `/api/generate` and `/api/chat` to pass requests through `PolicyEngine` first.
- Pass the decision to `ModelRouter`.
- Execute via `ModelManager` (if local).
- Record the transaction via `AuditLogger`.

**Acceptance Criteria:**
- API endpoints correctly block or route requests based on basic static policies.
- Audit logs are visible in server output.

**Risks:**
- Performance overhead. (Mitigation: Policy evaluation must be fast and synchronous for now).

### Phase 3: External Routing and Advanced Policies
**Goal:** Implement true external routing and dynamic policies.

- Extend `ModelRouter` to support calling external APIs (e.g., OpenAI, Anthropic) if the policy allows.
- Create a configuration mechanism (YAML/JSON) for users to define custom policies.
- Persist Audit logs to SQLite.

**Acceptance Criteria:**
- AirPI can successfully forward requests to external providers when permitted.
- Custom policies can be loaded at startup.
- Audit history can be queried.

## 4. Testing Strategy

Each phase requires specific tests:
- **Phase 1:** Unit tests for `PolicyEngine`, `ModelRouter`, and `AuditLogger` (ensuring correct logic paths).
- **Phase 2:** Integration tests for the API endpoints (ensuring requests are evaluated and logged).
- **Phase 3:** Mocked external API tests and policy configuration parsing tests.