# Model Verification Only — V9 user-friendly / hosted-safe

This revision fixes the later-phase Streamlit Cloud failure modes and adds usability guidance throughout the app.

## Important fixes
- Phase 5 benchmark uses a hosted-safe benchmark engine with automatic low-memory workload reduction.
- On constrained CPU-only hosts, Auto dtype uses FP16 to reduce model RAM footprint; explicit FP32 remains selectable.
- Separate prefill forward is skipped on constrained hosts and clearly labeled as a proxy.
- Phase 6 loads the model once, then measures baseline and optimized configurations on the same loaded model.
- Successful Phase 5 automatically becomes the Phase 7 baseline.
- Phase 7 locks its prompt to the baseline and treats missing/mismatched prerequisites as SKIPPED instead of a crash.
- Phase 8 treats missing accelerator parity as NOT READY / demo-only rather than an execution failure.
- Stable Transformers 4.x dependency range is used for Qwen2.5 compatibility.
- Every tab now includes a brief goal note, intuitive icons, and an Assumptions & caveats section.

## Community Cloud caveat
Streamlit Community Cloud is resource-constrained and commonly CPU-only. The app automatically uses a smaller benchmark workload there. Those CPU-hosted timing values are useful for learning and workflow validation, not vendor-grade B60/B70 benchmarking.

## Deploy
Replace your repository files with:
- app.py
- requirements.txt
- self_check.py
- .streamlit/config.toml

Then reboot the Streamlit app so dependencies reinstall cleanly.
