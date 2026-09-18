import os
import sys
import time
import shutil
import traceback
import tempfile
import platform
import importlib.util
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="HF Model Functional Verifier",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# -----------------------------
# Generic helpers
# -----------------------------
def package_status(name):
    return importlib.util.find_spec(name) is not None

def safe_version(pkg):
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return "not installed"

def version_tuple(text):
    nums = []
    for part in str(text).replace("+", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            nums.append(int(digits))
        else:
            break
    return tuple(nums[:3])

def gb(n):
    return n / (1024 ** 3)

def choose_device():
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_device(prefer):
    auto = choose_device()
    return auto if prefer == "Auto" else prefer.lower()

def selected_dtype(device, choice):
    import torch
    if choice == "FP32":
        return torch.float32
    if choice == "FP16":
        return torch.float16
    if choice == "BF16":
        return torch.bfloat16
    # conservative default
    return torch.float32 if device == "cpu" else torch.float16

def add_check(checks, name, status, detail, severity="HARD"):
    checks.append({
        "Check": name,
        "Status": status,
        "Severity": severity,
        "Detail": detail,
    })

def preflight_passed(checks):
    return not any(c["Severity"] == "HARD" and c["Status"] == "FAIL" for c in checks)

# -----------------------------
# Preflight
# -----------------------------
def run_preflight(model_id, prefer_device, dtype_choice, trust_remote_code):
    checks = []
    info = {
        "model_size_bytes": None,
        "disk_free_bytes": None,
        "ram_available_bytes": None,
        "device": None,
        "device_name": None,
        "model_type": None,
        "architecture": None,
    }

    # 1. Python
    py = sys.version_info
    if (py.major, py.minor) >= (3, 10):
        add_check(checks, "Python version", "PASS", f"Python {py.major}.{py.minor}.{py.micro}")
    else:
        add_check(checks, "Python version", "FAIL", f"Python {py.major}.{py.minor}.{py.micro}; Python 3.10+ is recommended for this package.")

    # 2. Required packages
    required = ["torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "psutil"]
    missing = [p for p in required if not package_status(p)]
    if missing:
        add_check(checks, "Required packages", "FAIL", "Missing: " + ", ".join(missing))
        return {"checks": checks, "info": info, "ready": False}
    else:
        add_check(checks, "Required packages", "PASS", "All required Python packages are importable.")

    import torch
    import transformers
    import psutil
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    # 3. Transformers compatibility, with a specific Qwen2.5 guard.
    tf_ver = version_tuple(transformers.__version__)
    if "Qwen2.5" in model_id or "qwen2.5" in model_id.lower():
        if tf_ver >= (4, 37, 0):
            add_check(checks, "Transformers / Qwen2.5 compatibility", "PASS",
                      f"transformers={transformers.__version__}; Qwen2.5 requires >=4.37.")
        else:
            add_check(checks, "Transformers / Qwen2.5 compatibility", "FAIL",
                      f"transformers={transformers.__version__}; Qwen2.5 can fail with versions below 4.37.")
    else:
        add_check(checks, "Transformers version", "PASS", f"transformers={transformers.__version__}", severity="SOFT")

    # 4. PyTorch basic import/version
    add_check(checks, "PyTorch import", "PASS", f"torch={torch.__version__}")

    # 5. Device existence
    device = resolve_device(prefer_device)
    info["device"] = device

    availability = True
    detail = f"Selected device: {device}"
    if device == "xpu":
        availability = hasattr(torch, "xpu") and torch.xpu.is_available()
        if availability:
            try:
                info["device_name"] = torch.xpu.get_device_name(0)
                detail += f" · {info['device_name']}"
            except Exception:
                pass
    elif device == "cuda":
        availability = torch.cuda.is_available()
        if availability:
            info["device_name"] = torch.cuda.get_device_name(0)
            detail += f" · {info['device_name']}"
    elif device == "mps":
        availability = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        info["device_name"] = "Apple MPS" if availability else None
    elif device == "cpu":
        info["device_name"] = platform.processor() or "CPU"
    else:
        availability = False

    add_check(checks, "Target device detected", "PASS" if availability else "FAIL",
              detail if availability else f"Requested {device}, but PyTorch reports it unavailable.")

    # 6. Actual tensor smoke test -- stronger than availability alone.
    if availability:
        try:
            dtype = selected_dtype(device, dtype_choice)
            a = torch.randn((16, 16), device=device, dtype=dtype)
            b = torch.randn((16, 16), device=device, dtype=dtype)
            c = a @ b
            # force async backends to finish
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "xpu" and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            finite = bool(torch.isfinite(c).all().item())
            add_check(checks, "Device compute smoke test", "PASS" if finite else "FAIL",
                      f"16×16 matrix multiply completed on {device} with {str(dtype).replace('torch.','')}.")
        except Exception as e:
            add_check(checks, "Device compute smoke test", "FAIL", f"{type(e).__name__}: {e}")

    # 7. Model repository access and size estimate
    api = HfApi()
    model_info = None
    try:
        model_info = api.model_info(model_id, files_metadata=True)
        add_check(checks, "Hugging Face model access", "PASS",
                  f"Repository resolved: {model_info.id}")
        total = 0
        for f in (model_info.siblings or []):
            size = getattr(f, "size", None)
            if size:
                total += size
        if total > 0:
            info["model_size_bytes"] = total
            add_check(checks, "Model download size", "PASS",
                      f"Repository files total about {gb(total):.2f} GB.", severity="SOFT")
        else:
            add_check(checks, "Model download size", "WARN",
                      "Repository resolved, but file-size metadata was unavailable.", severity="SOFT")
    except Exception as e:
        add_check(checks, "Hugging Face model access", "FAIL",
                  f"Could not resolve model repository: {type(e).__name__}: {e}")

    # 8. Config / architecture compatibility
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        info["model_type"] = getattr(cfg, "model_type", None)
        arch = getattr(cfg, "architectures", None)
        info["architecture"] = ", ".join(arch) if arch else "not declared"
        add_check(checks, "Model config / architecture", "PASS",
                  f"model_type={info['model_type']}; architectures={info['architecture']}")
        auto_map = getattr(cfg, "auto_map", None)
        if auto_map and not trust_remote_code:
            add_check(checks, "Remote-code requirement", "WARN",
                      "Model config advertises custom auto_map code, but Trust remote code is OFF. "
                      "If standard Transformers cannot resolve the architecture, enable it only if you trust the repository.",
                      severity="SOFT")
        else:
            add_check(checks, "Remote-code requirement", "PASS",
                      "No unresolved remote-code requirement detected.", severity="SOFT")
    except Exception as e:
        add_check(checks, "Model config / architecture", "FAIL",
                  f"AutoConfig failed: {type(e).__name__}: {e}")

    # 9. Disk headroom
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(cache_root)
    except Exception:
        usage = shutil.disk_usage(tempfile.gettempdir())
    info["disk_free_bytes"] = usage.free

    if info["model_size_bytes"]:
        recommended = info["model_size_bytes"] * 1.5
        ok = usage.free >= recommended
        add_check(checks, "Disk-space headroom", "PASS" if ok else "FAIL",
                  f"Free disk: {gb(usage.free):.1f} GB; recommended >= {gb(recommended):.1f} GB "
                  f"(about 1.5× repository size).")
    else:
        add_check(checks, "Disk-space headroom", "WARN",
                  f"Free disk: {gb(usage.free):.1f} GB. Model size unknown, so headroom could not be proven.",
                  severity="SOFT")

    # 10. Host RAM headroom
    vm = psutil.virtual_memory()
    info["ram_available_bytes"] = vm.available
    if info["model_size_bytes"]:
        # conservative host-side loading/caching heuristic, not a model requirement
        recommended_ram = max(2 * 1024**3, info["model_size_bytes"] * 2.0)
        status = "PASS" if vm.available >= recommended_ram else "WARN"
        add_check(checks, "Host RAM headroom", status,
                  f"Available RAM: {gb(vm.available):.1f} GB; heuristic target >= {gb(recommended_ram):.1f} GB.",
                  severity="SOFT")
    else:
        add_check(checks, "Host RAM headroom", "PASS",
                  f"Available RAM: {gb(vm.available):.1f} GB.", severity="SOFT")

    # 11. Accelerator memory visibility
    try:
        if device == "cuda" and torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            add_check(checks, "Accelerator memory visibility", "PASS",
                      f"CUDA memory: {gb(free_b):.1f} GB free / {gb(total_b):.1f} GB total.", severity="SOFT")
        elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            props = torch.xpu.get_device_properties(0)
            total = getattr(props, "total_memory", None)
            used = torch.xpu.device_memory_used(0) if hasattr(torch.xpu, "device_memory_used") else None
            if total:
                free = total - used if used is not None else None
                msg = f"XPU memory: {gb(total):.1f} GB total"
                if free is not None:
                    msg += f" · ~{gb(free):.1f} GB not currently reported used"
                add_check(checks, "Accelerator memory visibility", "PASS", msg, severity="SOFT")
            else:
                add_check(checks, "Accelerator memory visibility", "WARN",
                          "XPU is usable, but total-memory metadata was unavailable.", severity="SOFT")
        else:
            add_check(checks, "Accelerator memory visibility", "PASS",
                      "Not required for CPU/MPS preflight.", severity="SOFT")
    except Exception as e:
        add_check(checks, "Accelerator memory visibility", "WARN",
                  f"Could not query accelerator memory: {e}", severity="SOFT")

    # 12. B70/XPU-specific readiness signal
    if prefer_device == "XPU" or device == "xpu":
        xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        name = info.get("device_name") or ""
        add_check(checks, "Intel XPU readiness", "PASS" if xpu_ok else "FAIL",
                  f"torch.xpu.is_available()={xpu_ok}; device={name or 'not available'}")
        if xpu_ok and "B70" not in name.upper():
            add_check(checks, "B70 identity", "WARN",
                      f"An Intel XPU is available, but the reported device name is '{name}'. "
                      "If this run is intended to qualify Arc Pro B70 specifically, confirm the physical device.",
                      severity="SOFT")
        elif xpu_ok:
            add_check(checks, "B70 identity", "PASS",
                      f"Reported XPU device includes B70: {name}", severity="SOFT")

    ready = preflight_passed(checks)
    return {"checks": checks, "info": info, "ready": ready}

# -----------------------------
# Runtime snapshot
# -----------------------------
def runtime_snapshot():
    data = {
        "OS": platform.platform(),
        "Python": sys.version.split()[0],
        "torch": safe_version("torch"),
        "transformers": safe_version("transformers"),
        "accelerate": safe_version("accelerate"),
        "safetensors": safe_version("safetensors"),
        "huggingface_hub": safe_version("huggingface_hub"),
    }
    try:
        import torch
        data["Auto-selected device"] = choose_device()
        data["CUDA available"] = str(torch.cuda.is_available())
        data["XPU available"] = str(hasattr(torch, "xpu") and torch.xpu.is_available())
        data["MPS available"] = str(bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            data["XPU device"] = str(torch.xpu.get_device_name(0))
    except Exception as e:
        data["Torch check"] = str(e)
    return data

# -----------------------------
# Functional verification
# -----------------------------
def run_verification(model_id, prompt, max_new_tokens, prefer_device, dtype_choice, trust_remote_code=False):
    result = {
        "steps": [], "output": "", "device": None, "dtype": None,
        "load_seconds": None, "generation_seconds": None, "error": None
    }
    def add(step, status, detail):
        result["steps"].append({"Step": step, "Status": status, "Detail": detail})
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        add("Import libraries", "PASS", f"torch={torch.__version__}")

        device = resolve_device(prefer_device)
        result["device"] = device
        add("Use preflight-selected device", "PASS", device)

        dtype = selected_dtype(device, dtype_choice)
        result["dtype"] = str(dtype).replace("torch.", "")

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        add("Load tokenizer", "PASS", tokenizer.__class__.__name__)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.to(device)
        result["load_seconds"] = time.perf_counter() - t0
        add("Load model", "PASS",
            f"{model.__class__.__name__} on {device} as {result['dtype']}")

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        add("Prepare prompt", "PASS", f"Input tokens: {inputs['input_ids'].shape[-1]}")

        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "xpu" and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        result["generation_seconds"] = time.perf_counter() - t1
        add("Generate", "PASS", f"Completed in {result['generation_seconds']:.2f}s")

        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        text_out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result["output"] = text_out

        if not text_out:
            add("Validate output", "FAIL", "Decoded output was empty.")
            result["error"] = "Empty generated output."
            return result

        add("Validate output", "PASS",
            f"Generated {len(new_tokens)} new token(s); non-empty output.")
        add("Functional verdict", "PASS",
            "Tokenizer + model + device placement + generation + output validation succeeded.")
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        add("Functional verdict", "FAIL", result["error"])
        return result

COMPARISON = [
    {
        "Step": "Preflight: software compatibility",
        "Generic / local step": "Check Python, torch, Transformers, disk/RAM and model-repo access.",
        "Intel Arc Pro B70 step": "Do the same plus require a working XPU build/runtime.",
        "Why different": "B70 depends on the Intel XPU software path in addition to normal Python/model compatibility."
    },
    {
        "Step": "Preflight: device compute",
        "Generic / local step": "Run a tiny matrix multiply on the selected backend.",
        "Intel Arc Pro B70 step": "Run the same operation on xpu and synchronize it.",
        "Why different": "torch.xpu.is_available() proves discovery; a real tensor operation proves the path can execute work."
    },
    {
        "Step": "Check packages",
        "Generic / local step": "Verify torch, transformers, accelerate and safetensors.",
        "Intel Arc Pro B70 step": "Verify Intel-GPU-capable PyTorch and torch.xpu.",
        "Why different": "B70 is addressed through XPU, not CUDA."
    },
    {
        "Step": "Load tokenizer",
        "Generic / local step": "AutoTokenizer.from_pretrained(model_id).",
        "Intel Arc Pro B70 step": "Same.",
        "Why different": "Tokenizer work is host-side."
    },
    {
        "Step": "Load model",
        "Generic / local step": "AutoModelForCausalLM.from_pretrained(...).",
        "Intel Arc Pro B70 step": "Same weights, then move the model to xpu.",
        "Why different": "Model semantics stay the same; compute backend changes."
    },
    {
        "Step": "Smoke generation",
        "Generic / local step": "Run a short deterministic prompt.",
        "Intel Arc Pro B70 step": "Run exactly the same prompt on XPU.",
        "Why different": "Keeping the workload constant isolates backend-specific failures."
    },
    {
        "Step": "Verdict",
        "Generic / local step": "PASS when load + generation + non-empty output succeed.",
        "Intel Arc Pro B70 step": "PASS only when those gates succeed specifically on B70/XPU.",
        "Why different": "CPU functionality does not prove accelerator functionality."
    },
]

# -----------------------------
# UI
# -----------------------------
st.title("✅ Hugging Face Model Functional Verifier")
st.write(
    "A lightweight functional checker with a **preflight gate** so setup problems are caught "
    "before downloading/loading the full model."
)

with st.sidebar:
    st.header("Input")
    model_id = st.text_input("Hugging Face model", value=DEFAULT_MODEL)
    device = st.selectbox("Device", ["Auto", "CPU", "CUDA", "MPS", "XPU"])
    dtype_choice = st.selectbox("Dtype", ["Auto", "FP32", "FP16", "BF16"])
    max_new_tokens = st.slider("Max new tokens", 8, 128, 32, 8)
    trust_remote_code = st.toggle("Trust remote code", value=False)
    st.caption("For Intel Arc Pro B70, choose XPU explicitly when you want to qualify B70.")

tabs = st.tabs([
    "0 · Preflight",
    "1 · Run Verification",
    "2 · Generic vs B70",
    "3 · Environment",
    "4 · TPM Checklist",
])

with tabs[0]:
    st.subheader("Setup compatibility preflight")
    st.write(
        "Run this first. **HARD FAIL** blocks functional verification; "
        "**WARN** means the setup may still work but deserves attention."
    )

    if st.button("🔎 Run preflight", type="primary", use_container_width=True):
        with st.spinner("Checking setup compatibility..."):
            st.session_state["preflight"] = run_preflight(
                model_id.strip(), device, dtype_choice, trust_remote_code
            )

    pf = st.session_state.get("preflight")
    if pf:
        if pf["ready"]:
            st.success("✅ PREFLIGHT READY — no hard compatibility blockers detected.")
        else:
            st.error("❌ PREFLIGHT BLOCKED — fix HARD FAIL items before model verification.")

        st.dataframe(pf["checks"], use_container_width=True, hide_index=True)

        hard_fails = [c for c in pf["checks"] if c["Severity"] == "HARD" and c["Status"] == "FAIL"]
        warnings = [c for c in pf["checks"] if c["Status"] == "WARN"]

        a,b,c = st.columns(3)
        a.metric("Hard failures", len(hard_fails))
        b.metric("Warnings", len(warnings))
        c.metric("Selected device", pf["info"].get("device") or "—")

        if hard_fails:
            st.subheader("Fix first")
            for item in hard_fails:
                st.write(f"• **{item['Check']}** — {item['Detail']}")

with tabs[1]:
    st.subheader("Functional smoke test")
    pf = st.session_state.get("preflight")
    ready = bool(pf and pf.get("ready"))

    if not pf:
        st.warning("Run **0 · Preflight** first.")
    elif not ready:
        st.error("Functional verification is blocked because preflight has a HARD FAIL.")

    prompt = st.text_area(
        "Prompt",
        value="Reply with one short sentence explaining what a GPU does.",
        height=90
    )
    st.caption("This is functional verification, not a benchmark.")

    if st.button(
        "▶ Run functional verification",
        use_container_width=True,
        disabled=not ready
    ):
        with st.spinner("Loading and running model..."):
            st.session_state["last_result"] = run_verification(
                model_id.strip(), prompt.strip(), max_new_tokens,
                device, dtype_choice, trust_remote_code
            )

    result = st.session_state.get("last_result")
    if result:
        passed = result["steps"] and result["steps"][-1]["Status"] == "PASS"
        st.success("✅ FUNCTIONAL PASS") if passed else st.error("❌ FUNCTIONAL FAIL")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Device", result.get("device") or "—")
        c2.metric("Dtype", result.get("dtype") or "—")
        c3.metric("Load time", f"{result['load_seconds']:.2f}s" if result.get("load_seconds") is not None else "—")
        c4.metric("Generate time", f"{result['generation_seconds']:.2f}s" if result.get("generation_seconds") is not None else "—")
        st.dataframe(result["steps"], use_container_width=True, hide_index=True)
        if result.get("output"):
            st.subheader("Generated output")
            st.code(result["output"], language="text")
        if result.get("error"):
            st.error(result["error"])
            with st.expander("Traceback"):
                st.code(result.get("traceback",""), language="text")

with tabs[2]:
    st.subheader("Generic vs Intel Arc Pro B70")
    st.dataframe(COMPARISON, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Current runtime")
    snap = runtime_snapshot()
    st.dataframe(
        [{"Item": k, "Value": v} for k, v in snap.items()],
        use_container_width=True,
        hide_index=True
    )

with tabs[4]:
    st.subheader("Minimal TPM functional gate")
    checks = [
        "Preflight has no HARD FAIL.",
        "Model repository and AutoConfig resolve.",
        "Transformers version supports the model architecture.",
        "Target accelerator is detected.",
        "A tiny tensor compute operation succeeds on the target device.",
        "Disk/RAM headroom is reasonable.",
        "Tokenizer loads.",
        "Model weights load.",
        "Model and input tensors are on the same device.",
        "One deterministic generation completes.",
        "Decoded output is non-empty.",
        "Environment and exact failure evidence are captured.",
        "Only after functional PASS should benchmarking or optimization begin.",
    ]
    for i, item in enumerate(checks, 1):
        st.write(f"**{i}.** {item}")

st.divider()
st.caption(
    "Functional verification only. Preflight heuristics for disk/RAM are conservative setup checks, "
    "not formal hardware-capacity guarantees."
)
