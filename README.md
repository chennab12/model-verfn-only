# Hugging Face Model Qualification Lab V5

A lightweight Streamlit learning and qualification app that progresses organically from environment compatibility through production qualification.

Default model:

`Qwen/Qwen2.5-0.5B-Instruct`

## Full phase sequence

1. **Phase 0 — Preflight**
   - Python/package compatibility
   - Hugging Face model/config access
   - disk/RAM headroom
   - CPU/CUDA/MPS/XPU visibility
   - actual device tensor-compute smoke test

2. **Phase 2 — Functional Run Verification**
   - tokenizer/model load
   - device placement
   - prompt formatting/tokenization
   - inference
   - decode
   - functional PASS/FAIL
   - each internal step shows exact Python operation, actual runtime output, purpose, behind-the-scenes concept, and B60/B70 differences

3. **Phase 3 — Robustness / Repeatability Qualification**
   - repeated deterministic runs
   - prompt variation
   - longer-context sanity
   - pass rate / determinism / memory evidence

4. **Phase 4 — Correctness / Parity**
   - CPU reference
   - target backend run
   - decoded-text comparison
   - top-1 next-token match
   - top-5 overlap
   - max/mean absolute logit difference

5. **Phase 5 — Performance Benchmark**
   - controlled workload definition
   - warm-up
   - prefill latency
   - lightweight one-token TTFT proxy
   - end-to-end latency
   - approximate TPOT / decode throughput
   - generated tokens/sec
   - run-to-run coefficient of variation
   - accelerator memory evidence
   - save benchmark as Phase 7 regression baseline

6. **Phase 6 — Optimization**
   - baseline: `use_cache=False + torch.no_grad()`
   - candidate: `use_cache=True + torch.inference_mode()`
   - before/after latency
   - throughput gain
   - speedup
   - deterministic output parity
   - optimization promoted only when a measured gain exists

7. **Phase 7 — Regression**
   - load known-good Phase 5 baseline
   - rerun same workload
   - latency delta
   - throughput delta
   - configurable regression threshold
   - output parity guard
   - release PASS/FAIL gate

8. **Phase 8 — Production Qualification**
   - evidence completeness
   - upstream functional/robustness/parity gates
   - performance SLO
   - regression release gate
   - resource headroom evidence
   - reproducibility metadata
   - scoped production go/no-go

## Reference and TPM tabs

- **Generic vs B70** — generic/local step vs Intel Arc Pro step and why it differs
- **Environment** — OS/framework/device/runtime snapshot
- **TPM Checklist** — phase ordering and exit criteria
- **Summary** — one consolidated TPM implementation map

## Summary tab

The right-most Summary tab includes a table with:

- phase / reference area
- organic implementation gist
- current status
- key metrics
- most important TPM takeaway
- top 1% interview question
- ideal expected answer

It also includes dashboard visuals for:

- phase readiness
- Phase 5 latency / throughput metrics
- Phase 6 optimization gains
- Phase 7 regression deltas
- Phase 8 production readiness

## Intel Arc Pro B60 / B70

Both use the Intel XPU path in this learning app. The Hugging Face model/tokenizer logic remains largely the same. The practical differences are hardware capacity and compute headroom.

The app annotations use these key learning distinctions:

- B60: 24 GB GDDR6 class capacity
- B70: 32 GB GDDR6 class capacity
- B70 generally offers greater memory/compute headroom

Always capture the actual device identity and software-stack versions from the tested system rather than assuming a configuration.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Important benchmarking note

This is a **lightweight TPM learning / qualification tool**, not a substitute for a production-grade benchmark harness such as a dedicated vLLM/TGI/Triton/MLPerf-style system.

The TTFT value in this app is explicitly labeled **TTFT proxy** because it measures a non-streaming one-token generation request. For rigorous serving benchmarks, instrument the real serving stack and measure the first streamed token at the client/request boundary.

## Qualification mental model

```text
Preflight compatibility
        ↓
Single-run functional verification
        ↓
Repeatability / robustness qualification
        ↓
CPU-vs-target parity
        ↓
Performance benchmark baseline
        ↓
Controlled optimization
        ↓
Regression gate
        ↓
Production qualification
        ↓
Scale-out / serving / long-term operations
```
