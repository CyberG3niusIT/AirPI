# Product Identity: AirPI

## What is AirPI?

AirPI is the local Edge-AI node and control plane for private, small-scale business, and semi-professional infrastructures.

It functions as a central infrastructure building block that securely accepts, assesses, processes locally, conditionally routes, logs, and secures AI requests through defined policies within its own network. It ensures that AI operates in a controlled, transparent, and cloud-independent manner. AirPI is to AI operations what Pi-hole is to DNS and advertising: a locally hostable infrastructure layer for true sovereign control.

The core formula of AirPI is:
**AirPI = Local AI Runtime + Policy Engine + Model Router + Memory Control + Audit Layer**

### Key Characteristics
1. **Local AI Inference Node:** Processes tasks securely on-premise.
2. **Ollama-compatible API Layer:** Seamlessly integrates with existing tools relying on the Ollama standard.
3. **Controlled Interface:** Acts as a gateway between local services, users, devices, and external AI models.
4. **Privacy-first Preprocessing:** Handles data locally before any external routing.
5. **Policy-based Decision Layer:** Evaluates requests and enforces processing rules.
6. **Local Memory with Control:** Manages state and memory purposefully without unverified retention.
7. **Auditable Decisions and Logs:** Maintains transparent records of operations and model routing.
8. **Integration Hub:** Provides a stable endpoint for PI Guardian, Home Assistant, CLI, Web-UI, and future external models.
9. **Reliable Infrastructure Component:** Designed for continuous operation in home networks, small businesses, medical practices, schools, or lab environments.

## What AirPI is NOT

To maintain focus and product integrity, AirPI explicitly avoids being:

1. Just another local ChatGPT clone.
2. A toy web UI or purely experimental interface.
3. A simple Raspberry Pi technical demonstration.
4. An unstructured prompt playground.
5. A cloud-dependent autonomous agent.
6. A hobbyist project lacking a serious security model.
7. An AI assistant without clearly defined system boundaries.
8. An uncontrolled or continuously growing memory system.
9. A feature-creep project with a flashy roadmap but no solid infrastructural core.

## The Problem AirPI Solves

The integration of AI into networks currently forces users to choose between two problematic extremes: either rely entirely on opaque, cloud-based models that harvest data, or run isolated, unmanaged local models that lack routing, memory control, and auditability.

AirPI bridges this gap. It provides a secure, auditable, and easily manageable local node that standardizes how AI is consumed across an organization or household. It ensures sensitive data remains local, while still allowing controlled, rule-based access to more powerful external models when explicitly permitted.

## Target Audience

AirPI is designed for:
- Privacy-conscious individuals and households.
- Small and medium-sized businesses (SMBs).
- Medical practices, legal offices, and other confidentiality-bound professions.
- Educational institutions (schools, universities).
- Research and development laboratories.
- Homelab enthusiasts seeking robust, production-ready AI infrastructure.

## System Boundaries and Security Principles

### Security Principles
- **Local-first:** Default processing always occurs on local hardware.
- **Controlled Cloud Access:** External APIs are only utilized under strict, user-defined routing rules.
- **No Silent Data Leaks:** Sensitive content is never forwarded without explicit policy clearance.
- **Managed Retention:** Memory storage is strictly controlled, verifiable, and completely deletable.
- **Realistic Guarantees:** AirPI provides a secure framework but does not make false promises of absolute privacy or absolute autonomy. It enables control, not magic.
- **No Unverified Background Actions:** All processing is triggered by explicit requests and documented in audit logs.

### Model Handling
- **Local Models:** Primary processing targets. Handled entirely on-device (e.g., via llama.cpp on Raspberry Pi).
- **External Models:** Accessed via the Model Router only when the Policy Engine determines the request requires it and the data is safe to send.

## The Role of Components

While AirPI includes user-facing components, they serve specific infrastructural roles rather than being the product's primary purpose:

- **Web-UI & CLI:** These are operational control surfaces and diagnostic interfaces for managing the local AI infrastructure, not end-user entertainment chat apps.
- **Memory & Knowledge Graph:** These are state management and audit tools. They provide a controlled, inspectable, and manageable history of the AI's understanding, rather than an unconstrained "second brain."
- **Integration with PI Guardian:** PI Guardian acts as a crucial partner in the ecosystem. AirPI functions as the execution and routing backend, while PI Guardian handles network-level or advanced threat mitigation, forming a cohesive security architecture.
