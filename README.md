# Model Verification Only — V6 fixed

Lightweight Hugging Face model qualification app for `Qwen/Qwen2.5-0.5B-Instruct`, covering:

0. Preflight
1. Functional verification
2. Phase 3 robustness / repeatability
3. Phase 4 CPU-vs-target parity
4. Phase 5 performance benchmark
5. Phase 6 optimization
6. Phase 7 regression
7. Phase 8 production qualification
8. Generic vs B70 reference
9. Environment
10. TPM checklist
11. Summary dashboard

## Reliability fixes in this build

- Replaced deprecated Streamlit `use_container_width` calls with `width="stretch"`.
- Uses Transformers v5 `dtype=` automatically; falls back to `torch_dtype=` only on Transformers v4.
- Resets sampling-only generation settings for deterministic runs so Qwen's saved sampling config does not emit irrelevant warnings when `do_sample=False`.
- Adds explicit CPU/CUDA/XPU memory cleanup between phase runs.
- Phase 4 no longer holds CPU-reference and target model copies in memory simultaneously.
- Phase 5 baseline stores model/workload/device/dtype metadata.
- Phase 7 refuses an invalid apples-to-oranges regression comparison if baseline workload/config differs.
- Summary readiness visualization excludes reference-only tabs from the readiness score.
- Streamlit file watching is disabled to avoid PyTorch `torch.classes` watcher warnings.
- Hosted Streamlit is treated as CPU-only unless PyTorch really reports an XPU.
- Dependency ranges are constrained to compatible current families.

## Dependencies

`requirements.txt`:

```text
streamlit>=1.57,<1.63
transformers>=5.10,<5.18
torch>=2.10,<2.14
accelerate>=1.10,<2
safetensors>=0.6,<1
huggingface-hub>=1.0,<2
psutil>=6,<8
pandas>=2.2,<3
```

Transformers currently documents Python 3.10+ and PyTorch 2.5+ as supported. This package uses narrower ranges so Streamlit Cloud does not unexpectedly jump across major APIs.

## Deploy to Streamlit Community Cloud

Upload these files to the repository root:

```text
app.py
requirements.txt
self_check.py
.streamlit/config.toml
README.md
```

Set the main file to `app.py`, then reboot the app after replacing the files.

Streamlit Community Cloud will normally execute this app on CPU. Intel Arc Pro B60/B70 functional testing requires running the same package on a machine/container where that Intel GPU and its driver/runtime are actually attached.

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe self_check.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Before debugging a tab

Run:

```powershell
python self_check.py
```

It verifies imports, minimum versions, app syntax, required phase functions, and checks that deprecated Streamlit width calls are absent. It does not download the model.

## Important Phase 7 rule

A regression baseline is valid only if the candidate uses the same:

- model ID
- prompt/workload
- `max_new_tokens`
- target device
- dtype selection

If any of these differ, V6 rejects the regression comparison instead of reporting a misleading delta.


## Phase 4 parity hotfix

On CPU-only hosted environments, Phase 4 now skips duplicate CPU-vs-CPU model loading and returns an educational dry-run verdict. On real CUDA/XPU/MPS hardware, it still performs the full CPU-reference vs target-backend comparison, with aggressive cleanup between model loads.


## V8 later-phase reliability fixes

### One shared benchmark engine

Phases 5, 6, and 7 now call the same `quick_benchmark()` implementation.

It performs:
- warm-up
- prefill timing
- one-token TTFT proxy
- full deterministic generation timing
- tokens/sec
- approximate decode tokens/sec
- approximate TPOT
- run-to-run latency CV
- device-memory evidence
- deterministic output capture
- mandatory model/tensor cleanup before returning

This prevents Phase 5, 6, and 7 from silently using different measurement methods.

### Phase 3 memory hardening

Repeat and prompt-suite tensors are released after each iteration, and the loaded model is explicitly released before the tab returns.

### Phase 7 baseline validity

The saved baseline signature now includes:
- model ID
- prompt
- max output tokens
- resolved device
- requested device
- dtype choice
- benchmark engine version
- KV-cache setting
- inference-mode setting

A candidate that does not match the baseline definition is rejected rather than compared.

### Phase 8 fail-closed parity logic

A Streamlit Cloud CPU-only Phase 4 dry-run now has mode:

`educational_skip`

Phase 8 does **not** count this as accelerator parity.

Production qualification requires:

`phase4.summary.mode == "real_backend_parity"`

plus an acceptable parity status.

### Summary status semantics

The Summary tab now distinguishes:
- PASS
- WARN
- FAIL
- SKIPPED
- NOT RUN

A CPU-only parity dry-run is shown as **SKIPPED**, not PASS.
