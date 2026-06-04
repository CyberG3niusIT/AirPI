# Architecture: AirPI

AirPI is structured as a modular, local AI control plane. Its architecture enforces a strict separation of concerns, ensuring that requests are predictably evaluated, securely routed, transparently logged, and efficiently processed.

## High-Level Architecture Overview

The flow of data through the AirPI system follows a clear, secure path:

1. **Request Intake:** A user or connected service sends an AI request to AirPI via the API or UI.
2. **Evaluation:** The system assesses the request to determine its nature, required resources, and sensitive content.
3. **Policy Enforcement:** The Policy Engine evaluates the request against predefined rules. It decides whether the request should be processed locally, routed to an external model, blocked, or if user clarification is required.
4. **Routing:** Based on the policy decision, the Model Router directs the request to the appropriate backend (local or external).
5. **Execution:** The chosen model processes the request.
6. **Memory Control:** The Memory Control layer inspects the result to determine if any information should be retained, strictly following privacy policies.
7. **Auditing:** The Audit Layer logs the entire decision-making process, routing choices, and outcomes.
8. **Response:** The final processed response is returned to the user or service.

## Core Layers

The architecture separates the system into distinct, manageable layers.

### 1. API (Intake & Interface)
- **Role:** Handles incoming requests and provides standardized interfaces.
- **Components:** Ollama-compatible REST API, WebSocket streams.
- **Functionality:** Ensures seamless integration with external tools and acts as the entry point for all operations.

### 2. Policy Engine (Evaluation & Decision)
- **Role:** The core security and compliance checkpoint.
- **Functionality:** Applies rules to every incoming request. It enforces data privacy boundaries, blocks unauthorized content, and dictates whether a request is allowed to leave the local network.

### 3. Model Router (Execution Management)
- **Role:** Directs traffic to the appropriate computational resource.
- **Functionality:** Selects the optimal model based on the Policy Engine's decision, request complexity, and current system load. Handles fallbacks and manages connections to external APIs if permitted.

### 4. Local AI Runtime (Processing)
- **Role:** The execution environment for on-device inference.
- **Components:** llama.cpp (C++ core), KV-cache management, thread allocation.
- **Functionality:** Processes requests that are routed locally. Optimized for stable, continuous operation on Edge hardware (e.g., Raspberry Pi 5).

### 5. Memory Control (State Management)
- **Role:** Manages what AirPI remembers.
- **Functionality:** A strictly controlled system that determines what facts or context are stored. Prioritizes explicit retention rules, deduplication, and complete deletability over unconstrained data collection.

### 6. Audit Layer (Logging & Compliance)
- **Role:** Ensures system transparency.
- **Functionality:** Logs routing decisions, policy enforcement actions, model selection, and memory changes. Crucial for verifying system behavior and ensuring compliance with local rules.

### 7. UI & CLI (Control Surfaces)
- **Role:** Operational management interfaces.
- **Functionality:** Web UI and CLI are designed as administrative control panels and diagnostic tools, not as end-user toys. They allow administrators to inspect logs, manage memory, and configure policies.

### 8. Integrations (Ecosystem)
- **Role:** Connects AirPI to broader infrastructure.
- **Components:** Integration with PI Guardian, Home Assistant, and other network services.
- **Functionality:** Allows AirPI to serve as the intelligence backend for network-wide security systems and automation, cleanly separating AI execution from network management.
