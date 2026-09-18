# Hugging Face Model Functional Verifier V2

A lightweight Streamlit app focused only on Hugging Face causal-LM **functional verification**, now with a setup compatibility **Preflight** stage.

Default:
`Qwen/Qwen2.5-0.5B-Instruct`

## Preflight checks

Before model loading, the app checks:

1. Python version
2. required packages
3. Qwen2.5 / Transformers compatibility
4. PyTorch import/version
5. requested device is actually available
6. real 16×16 tensor matrix multiply on the requested device
7. Hugging Face repository access
8. model config / architecture resolves
9. remote-code requirement signal
10. repository download-size estimate
11. disk-space headroom
12. host RAM headroom
13. accelerator-memory visibility when available
14. Intel XPU readiness
15. B70 identity warning when XPU is present but the reported device is not clearly B70

A **HARD FAIL** blocks the Run Verification button.

## Why the XPU compute test matters

This is stronger than:

```python
torch.xpu.is_available()
```

The preflight also creates two tiny tensors on XPU and performs a matrix multiply.
That catches cases where the device is visible but the compute/runtime path is broken.

## Qwen2.5 guard

Qwen documents that `transformers < 4.37.0` can fail with `KeyError: 'qwen2'`.
The preflight checks this explicitly.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Intel Arc Pro B70

For a real B70 verification:

1. run this app on the machine/container with the B70 attached
2. install the current Intel graphics driver/runtime
3. install a current PyTorch XPU build appropriate for the OS
4. select `XPU`
5. run Preflight
6. require the XPU tensor-compute smoke test to PASS
7. then run model functional verification

Streamlit Community Cloud normally cannot verify a physical B70 because the hardware is not attached there.

## Notes

Disk and RAM headroom are conservative heuristics used to catch obvious setup problems. They are not formal model-memory requirements.


## V3 TPM transparent Run Verification tab

Only the **Run Verification** tab now expands every internal step and displays:

- exact Python operation being run
- actual runtime output
- AI/ML core concept represented by that step
- true purpose
- what happens behind the scenes
- equivalent Intel Arc Pro B60 behavior
- equivalent Intel Arc Pro B70 behavior
- why B60/B70 differ

The visible execution sequence includes:

1. import PyTorch/Transformers
2. select device
3. select numerical precision
4. load tokenizer
5. load model architecture + weights
6. place model on device
7. apply chat template
8. tokenize to tensors
9. move input tensors to device
10. autoregressive generation
11. isolate new tokens
12. decode token IDs
13. functional PASS/FAIL gate

### B60 vs B70 learning point

The Hugging Face/PyTorch functional flow is essentially the same because both use Intel's XPU path.

The practical hardware differences are capacity and compute:

- Arc Pro B60: 24 GB GDDR6, 456 GB/s memory bandwidth, 160 XMX engines
- Arc Pro B70: 32 GB GDDR6, 608 GB/s memory bandwidth, 256 XMX engines

This means the main functional-verification differences are likely to appear around model/context memory headroom and device/runtime behavior, rather than different Hugging Face code.
