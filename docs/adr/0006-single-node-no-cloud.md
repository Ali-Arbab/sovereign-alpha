# ADR-0006: Single-node closed-loop, zero cloud dependency

**Status:** Accepted
**Date:** 2026-04-28

## Context
The economic premise of Sovereign Alpha (master directive §1) is that a fixed-cost ₹10,00,000 hardware substrate amortizes to **zero marginal cost per re-run**. Variable cloud-API token costs across a full 10-year, 1.8M-document corpus, multiplied by iterative prompt-tuning cycles, would compound into millions of rupees per research iteration. Distributed compute introduces network jitter, security surface, and a vendor-lock-in axis the research thesis does not need.

## Decision
**All tokenization, inference, temporal joins, friction modeling, and rendering occur within a single-node, closed-loop environment.** No external API calls. No telemetry. No phone-home dependencies. No reliance on hosted vector DBs, hosted LLMs, or cloud blob storage.

## Alternatives considered
- **OpenAI / Anthropic API for inference.** Best models, but per-token cost compounds catastrophically over re-inference loops. Also: the prompt-as-alpha thesis requires reasoning-trace persistence, which most APIs strip or rate-limit.
- **AWS / GCP for distributed inference.** Could parallelize across GPUs, but: GPU instance prices, egress costs, and the need to ship 1.8M docs into the cloud each run defeat the economics.
- **Self-hosted vLLM cluster.** A coherent direction post-Sovereign-state if compute scales, but a single 5090 / 32B-q6 fits the bootstrap thesis.

## Consequences
- Hardware is a real ~₹10L upfront cost.
- Iterative velocity is unbounded; marginal cost = electricity.
- Network failures cannot interrupt a research run (the iterative loop has no network dependency).
- Storage is on-prem: 4TB Gen5 NVMe holds the data lake; backups are the operator's responsibility.
- ADR-0003 (ZMQ + MessagePack bridge) is consistent — even the UE5 client is on the same machine.
