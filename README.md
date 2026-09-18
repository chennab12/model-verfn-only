# Hugging Face Model Functional Verifier V4

This version extends the TPM-transparent verifier with **two new tabs**:

- **Phase 3 Qualification**
- **Phase 4 Parity**

These sit after the existing single-run **Run Verification** tab and before any future benchmarking phase.

## Tabs

1. **Preflight**
   - environment compatibility
   - package checks
   - Hugging Face model access
   - disk/RAM headroom
   - device/XPU smoke test

2. **Run Verification**
   - single-run functional smoke test
   - 13 detailed internal steps
   - exact operation
   - actual runtime output
   - true purpose
   - behind-the-scenes explanation
   - B60/B70 comparison

3. **Phase 3 Qualification**
   - repeated deterministic runs
   - prompt-variation stability
   - longer-context sanity
   - aggregated robustness metrics
   - detailed step-by-step TPM explanation
   - B60/B70 annotations

4. **Phase 4 Parity**
   - CPU reference path
   - target-backend path
   - decoded-text comparison
   - top-1 next-token comparison
   - top-5 overlap
   - max/mean absolute logit difference
   - detailed step-by-step TPM explanation
   - B60/B70 annotations

5. **Generic vs B70**
6. **Environment**
7. **TPM Checklist**

## Why this order matters

Recommended progression:

1. Preflight
2. Single-run functional verification
3. Phase 3 robustness qualification
4. Phase 4 CPU-vs-target parity
5. Benchmarking
6. Optimization
7. Regression automation

This helps ensure that future benchmark numbers are based on a setup that is already:
- compatible
- functional
- stable enough
- behaviorally credible

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes

- Phase 3 and Phase 4 are still lightweight qualification steps, not full benchmark suites.
- The app remains practical for TPM learning and demo-scale model verification.
- Intel Arc Pro B60 and B70 both use the XPU path. The main differences are capacity and compute headroom, not different Hugging Face APIs.
