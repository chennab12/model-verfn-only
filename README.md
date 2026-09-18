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
