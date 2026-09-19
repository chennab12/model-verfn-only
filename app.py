import os
import re
import sys
import time
import math
import shutil
import psutil
import traceback
import tempfile
import platform
import importlib.util
import gc
import copy
import inspect
import statistics
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="HF Model Functional Verifier",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Reduce non-actionable library noise in hosted deployments.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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

def hosted_safe_mode(device=None):
    """Detect constrained CPU-only hosts such as Streamlit Community Cloud."""
    try:
        dev = device or choose_device()
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        return dev == "cpu" and total_gb <= 5.0
    except Exception:
        return False

def selected_dtype(device, choice):
    import torch
    if choice == "FP32":
        return torch.float32
    if choice == "FP16":
        return torch.float16
    if choice == "BF16":
        return torch.bfloat16
    # Auto: on constrained CPU hosts, FP16 materially reduces the ~0.5B model
    # memory footprint. Explicit FP32 remains available from the sidebar.
    if device == "cpu" and hosted_safe_mode(device):
        return torch.float16
    return torch.float32 if device == "cpu" else torch.float16


def model_load_kwargs(dtype, trust_remote_code=False):
    """Return warning-free Transformers kwargs across v4/v5."""
    import transformers
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if version_tuple(transformers.__version__) >= (5, 0, 0):
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    return kwargs


def load_causal_model(model_id, dtype, trust_remote_code=False):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_load_kwargs(dtype, trust_remote_code),
    )
    model.eval()
    return model


def deterministic_generate(model, **kwargs):
    """Greedy generation without sampling-only Qwen warnings."""
    cfg = copy.deepcopy(model.generation_config)
    cfg.do_sample = False
    # Keep neutral/default sampling values so Transformers validation stays quiet
    # across stable v4 releases while greedy decoding remains active.
    neutral = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "typical_p": 1.0,
        "min_p": None,
    }
    for attr, value in neutral.items():
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)
    kwargs.pop("do_sample", None)
    return model.generate(generation_config=cfg, **kwargs)


def cleanup_accelerator(device=None):
    """Best-effort cleanup to keep repeated Streamlit runs from retaining memory."""
    gc.collect()
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            if hasattr(torch.xpu, "empty_cache"):
                torch.xpu.empty_cache()
    except Exception:
        pass

def add_check(checks, name, status, detail, severity="HARD"):
    checks.append({
        "Check": name,
        "Status": status,
        "Severity": severity,
        "Detail": detail,
    })

def preflight_passed(checks):
    return not any(c["Severity"] == "HARD" and c["Status"] == "FAIL" for c in checks)

def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def sample_memory_report(device):
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return f"CUDA memory free={gb(free_b):.2f} GB / total={gb(total_b):.2f} GB"
        if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            props = torch.xpu.get_device_properties(0)
            total = getattr(props, "total_memory", None)
            used = torch.xpu.device_memory_used(0) if hasattr(torch.xpu, "device_memory_used") else None
            if total is not None:
                if used is not None:
                    return f"XPU memory used~={gb(used):.2f} GB / total={gb(total):.2f} GB"
                return f"XPU total memory={gb(total):.2f} GB"
        return "No accelerator memory metric required for CPU/MPS."
    except Exception as e:
        return f"Memory query skipped: {e}"

def tab_note(icon, title, body):
    st.info(f"{icon} **{title}:** {body}")

def assumptions_box(items):
    with st.expander("ℹ️ Assumptions & caveats", expanded=False):
        for item in items:
            st.write(f"• {item}")

def fmt_metric(value, suffix="", digits=2, missing="—"):
    if value is None:
        return missing
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)

def render_step_expanders(result, output_label="Final generated output", metrics=None):
    if result is None:
        return
    final_status = "PASS"
    if result.get("steps"):
        final_status = result["steps"][-1]["status"]
    if final_status == "PASS":
        st.success("✅ PASS")
    elif final_status == "WARN":
        st.warning("⚠️ WARN — review the evidence below")
    elif final_status == "SKIPPED":
        st.info("⏭️ SKIPPED / NOT APPLICABLE")
    else:
        st.error("❌ FAIL")
    if metrics:
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, value)
    st.markdown("### Behind-the-scenes execution steps")
    for step in result.get("steps", []):
        icon = "✅" if step["status"] == "PASS" else "❌" if step["status"] == "FAIL" else "⚠️"
        with st.expander(f"{icon} Step {step['number']} · {step['name']}", expanded=True):
            m1, m2 = st.columns([1, 1])
            with m1:
                st.markdown("#### Exact core operation")
                st.code(step["command"], language="python")
                st.markdown("#### Actual runtime output")
                if step["status"] == "PASS":
                    st.success(step["output"])
                elif step["status"] == "WARN":
                    st.warning(step["output"])
                else:
                    st.error(step["output"])
                st.markdown("#### AI/ML core concept")
                st.info(step["concept"])
            with m2:
                st.markdown("#### True purpose")
                st.write(step["purpose"])
                st.markdown("#### What happens behind the scenes")
                st.write(step["behind_scenes"])

            st.markdown("#### Intel Arc Pro comparison")
            compare_rows = [
                {"Target": "Intel Arc Pro B60", "What changes": step["b60"]},
                {"Target": "Intel Arc Pro B70", "What changes": step["b70"]},
                {"Target": "Why the difference", "What changes": step["why_diff"]},
            ]
            st.dataframe(compare_rows, width="stretch", hide_index=True)

    if result.get("output"):
        st.subheader(output_label)
        st.code(result["output"], language="text")
    if result.get("samples"):
        st.subheader("Sample outputs / evidence")
        for item in result["samples"]:
            st.code(item, language="text")
    if result.get("error"):
        st.error(result["error"])
        with st.expander("Technical traceback"):
            st.code(result.get("traceback", ""), language="text")

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

    py = sys.version_info
    if (py.major, py.minor) >= (3, 10):
        add_check(checks, "Python version", "PASS", f"Python {py.major}.{py.minor}.{py.micro}")
    else:
        add_check(checks, "Python version", "FAIL", f"Python {py.major}.{py.minor}.{py.micro}; Python 3.10+ is recommended.")

    required = ["torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "psutil"]
    missing = [p for p in required if not package_status(p)]
    if missing:
        add_check(checks, "Required packages", "FAIL", "Missing: " + ", ".join(missing))
        return {"checks": checks, "info": info, "ready": False}
    else:
        add_check(checks, "Required packages", "PASS", "All required Python packages are importable.")

    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    tf_ver = version_tuple(transformers.__version__)
    if "Qwen2.5" in model_id or "qwen2.5" in model_id.lower():
        if tf_ver >= (4, 37, 0):
            add_check(checks, "Transformers / Qwen2.5 compatibility", "PASS",
                      f"transformers={transformers.__version__}; Qwen2.5 requires >=4.37.")
        else:
            add_check(checks, "Transformers / Qwen2.5 compatibility", "FAIL",
                      f"transformers={transformers.__version__}; Qwen2.5 can fail below 4.37.")
    else:
        add_check(checks, "Transformers version", "PASS", f"transformers={transformers.__version__}", severity="SOFT")

    add_check(checks, "PyTorch import", "PASS", f"torch={torch.__version__}")

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

    if availability:
        try:
            dtype = selected_dtype(device, dtype_choice)
            a = torch.randn((16, 16), device=device, dtype=dtype)
            b = torch.randn((16, 16), device=device, dtype=dtype)
            c = a @ b
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "xpu" and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            finite = bool(torch.isfinite(c).all().item())
            add_check(checks, "Device compute smoke test", "PASS" if finite else "FAIL",
                      f"16×16 matrix multiply completed on {device} with {str(dtype).replace('torch.','')}.")
        except Exception as e:
            add_check(checks, "Device compute smoke test", "FAIL", f"{type(e).__name__}: {e}")

    api = HfApi()
    try:
        model_info = api.model_info(model_id, files_metadata=True)
        add_check(checks, "Hugging Face model access", "PASS", f"Repository resolved: {model_info.id}")
        total = 0
        for f in (model_info.siblings or []):
            size = getattr(f, "size", None)
            if size:
                total += size
        if total > 0:
            info["model_size_bytes"] = total
            add_check(checks, "Model download size", "PASS", f"Repository files total about {gb(total):.2f} GB.", severity="SOFT")
        else:
            add_check(checks, "Model download size", "WARN", "Repository resolved, but file-size metadata was unavailable.", severity="SOFT")
    except Exception as e:
        add_check(checks, "Hugging Face model access", "FAIL", f"Could not resolve model repository: {type(e).__name__}: {e}")

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
                      "Model config advertises custom auto_map code, but Trust remote code is OFF.",
                      severity="SOFT")
        else:
            add_check(checks, "Remote-code requirement", "PASS",
                      "No unresolved remote-code requirement detected.", severity="SOFT")
    except Exception as e:
        add_check(checks, "Model config / architecture", "FAIL",
                  f"AutoConfig failed: {type(e).__name__}: {e}")

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
                  f"Free disk: {gb(usage.free):.1f} GB; recommended >= {gb(recommended):.1f} GB.")
    else:
        add_check(checks, "Disk-space headroom", "WARN",
                  f"Free disk: {gb(usage.free):.1f} GB. Model size unknown, so headroom could not be proven.",
                  severity="SOFT")

    vm = psutil.virtual_memory()
    info["ram_available_bytes"] = vm.available
    if info["model_size_bytes"]:
        recommended_ram = max(2 * 1024**3, info["model_size_bytes"] * 2.0)
        status = "PASS" if vm.available >= recommended_ram else "WARN"
        add_check(checks, "Host RAM headroom", status,
                  f"Available RAM: {gb(vm.available):.1f} GB; heuristic target >= {gb(recommended_ram):.1f} GB.",
                  severity="SOFT")
    else:
        add_check(checks, "Host RAM headroom", "PASS",
                  f"Available RAM: {gb(vm.available):.1f} GB.", severity="SOFT")

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
                msg = f"XPU memory: {gb(total):.1f} GB total"
                if used is not None:
                    msg += f" · ~{gb(used):.1f} GB currently used"
                add_check(checks, "Accelerator memory visibility", "PASS", msg, severity="SOFT")
            else:
                add_check(checks, "Accelerator memory visibility", "WARN", "XPU usable, but total-memory metadata unavailable.", severity="SOFT")
        else:
            add_check(checks, "Accelerator memory visibility", "PASS", "Not required for CPU/MPS preflight.", severity="SOFT")
    except Exception as e:
        add_check(checks, "Accelerator memory visibility", "WARN", f"Could not query accelerator memory: {e}", severity="SOFT")

    if prefer_device == "XPU" or device == "xpu":
        xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
        name = info.get("device_name") or ""
        add_check(checks, "Intel XPU readiness", "PASS" if xpu_ok else "FAIL",
                  f"torch.xpu.is_available()={xpu_ok}; device={name or 'not available'}")
        if xpu_ok and "B70" not in name.upper():
            add_check(checks, "B70 identity", "WARN",
                      f"Intel XPU available, but reported device name is '{name}'. Confirm physical hardware.",
                      severity="SOFT")
        elif xpu_ok:
            add_check(checks, "B70 identity", "PASS", f"Reported XPU device includes B70: {name}", severity="SOFT")

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
# Phase 2: single-run functional verification
# -----------------------------
def run_verification(model_id, prompt, max_new_tokens, prefer_device, dtype_choice, trust_remote_code=False):
    result = {
        "steps": [], "output": "", "device": None, "dtype": None,
        "load_seconds": None, "generation_seconds": None,
        "error": None, "traceback": None
    }

    def add_step(number, name, status, command, output, purpose, behind_scenes, concept, b60, b70, why_diff):
        result["steps"].append({
            "number": number, "name": name, "status": status, "command": command,
            "output": output, "purpose": purpose, "behind_scenes": behind_scenes,
            "concept": concept, "b60": b60, "b70": b70, "why_diff": why_diff
        })

    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        add_step(1, "Import PyTorch + Transformers", "PASS",
                 "import torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM",
                 f"torch={torch.__version__}; transformers={transformers.__version__}",
                 "Prove the core AI/ML stack is importable before touching model files.",
                 "PyTorch provides tensor execution and device management; Transformers provides model/tokenizer abstractions.",
                 "Framework/runtime layer",
                 "Same imports on B60.", "Same imports on B70.",
                 "Hardware does not change the Python import surface.")

        device = resolve_device(prefer_device)
        result["device"] = device
        add_step(2, "Select execution device", "PASS",
                 f"device = '{device}'", f"Selected device: {device}",
                 "Decide where tensor math and inference will run.",
                 "All tensors and model weights must end up on the same backend.",
                 "Device abstraction / accelerator backend",
                 "Use .to('xpu') for B60.", "Use .to('xpu') for B70.",
                 "B60/B70 share the same logical XPU path.")

        dtype = selected_dtype(device, dtype_choice)
        result["dtype"] = str(dtype).replace("torch.", "")
        add_step(3, "Choose numerical precision", "PASS",
                 f"dtype = torch.{result['dtype']}", f"Selected dtype: {result['dtype']}",
                 "Choose memory/computation precision for weights and inference.",
                 "Precision affects memory footprint and may affect backend support/performance.",
                 "Precision / datatype",
                 "Lower precision can help B60 fit more workload in 24 GB.", "B70 also benefits, but has 32 GB headroom.",
                 "The API is identical; the main difference is memory capacity and compute headroom.")

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        tok_elapsed = time.perf_counter() - t0
        add_step(4, "Load tokenizer", "PASS",
                 f"AutoTokenizer.from_pretrained('{model_id}')",
                 f"{tokenizer.__class__.__name__} loaded in {tok_elapsed:.2f}s",
                 "Load the text-to-token converter used by the model.",
                 "Vocabulary, merges, tokenizer config, and chat template are loaded on the host side.",
                 "Tokenization",
                 "Same host-side tokenizer on B60.", "Same host-side tokenizer on B70.",
                 "Tokenization is largely hardware-independent.")

        t1 = time.perf_counter()
        model = load_causal_model(model_id, dtype, trust_remote_code)
        load_elapsed = time.perf_counter() - t1
        param_count = sum(p.numel() for p in model.parameters())
        add_step(5, "Load model configuration + weights", "PASS",
                 "AutoModelForCausalLM.from_pretrained(...)\nmodel.eval()",
                 f"{model.__class__.__name__}; parameters={param_count:,}; host load={load_elapsed:.2f}s",
                 "Instantiate the neural-network architecture and load pretrained parameters.",
                 "Transformers builds the model graph from config and loads checkpoint tensors.",
                 "Model architecture + pretrained parameters",
                 "Same Qwen architecture; B60 may hit capacity limits earlier for larger models.", "Same architecture; B70 has more capacity margin.",
                 "The Hugging Face loading logic is identical. Differences appear when weights move to accelerator memory.")

        t2 = time.perf_counter()
        model.to(device)
        move_elapsed = time.perf_counter() - t2
        result["load_seconds"] = tok_elapsed + load_elapsed + move_elapsed
        add_step(6, "Place model on target device", "PASS",
                 f"model.to('{device}')", f"Model moved to {device} in {move_elapsed:.2f}s",
                 "Put model parameters onto the hardware that will execute inference.",
                 "PyTorch allocates device memory and transfers/materializes model tensors there.",
                 "Tensor placement / device memory",
                 "B60 uses 24 GB GDDR6 XPU memory.", "B70 uses 32 GB GDDR6 XPU memory.",
                 "This is where B70's larger VRAM can reduce out-of-memory risk.")

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            template_note = "chat template applied"
        else:
            formatted = prompt
            template_note = "raw prompt used"
        add_step(7, "Apply chat/instruction template", "PASS",
                 "tokenizer.apply_chat_template(messages, add_generation_prompt=True)",
                 f"{template_note}; formatted length={len(formatted)} chars",
                 "Convert the user message into the text pattern expected by the instruction-tuned model.",
                 "The template adds role markers/control tokens that align with instruction-tuning data.",
                 "Prompt formatting / instruction tuning",
                 "Same formatting on B60.", "Same formatting on B70.",
                 "Prompt formatting must stay constant for fair backend comparison.")

        inputs = tokenizer(formatted, return_tensors="pt")
        token_count = int(inputs["input_ids"].shape[-1])
        add_step(8, "Tokenize prompt into tensors", "PASS",
                 "inputs = tokenizer(formatted_prompt, return_tensors='pt')",
                 f"input_tokens={token_count}; first IDs={inputs['input_ids'][0][:12].tolist()}",
                 "Convert text into integer token IDs and tensor structures.",
                 "The tokenizer maps text pieces to vocabulary IDs; PyTorch wraps them as tensors.",
                 "Tokens / tensors / sequence length",
                 "Same token IDs on B60.", "Same token IDs on B70.",
                 "Tokenization output is hardware-independent.")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        add_step(9, "Move input tensors to target device", "PASS",
                 f"inputs = {{k: v.to('{device}') for k, v in inputs.items()}}",
                 f"Tensor shapes={ {k:list(v.shape) for k,v in inputs.items()} }",
                 "Ensure inputs and model parameters are on the same backend.",
                 "Prompt tensors are copied into device-visible memory before inference begins.",
                 "Host-to-device transfer",
                 "Same XPU transfer on B60.", "Same XPU transfer on B70.",
                 "Main difference is performance/headroom, not logic.")

        t3 = time.perf_counter()
        with torch.no_grad():
            out = deterministic_generate(model, 
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "xpu" and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        result["generation_seconds"] = time.perf_counter() - t3
        add_step(10, "Run autoregressive generation", "PASS",
                 "with torch.no_grad(): model.generate(..., do_sample=False, max_new_tokens=N)",
                 f"generation_time={result['generation_seconds']:.2f}s; total_output_tokens={int(out.shape[-1])}",
                 "Execute the forward passes needed to produce new tokens.",
                 "The model processes the full prompt (prefill) and then predicts one next token at a time (decode).",
                 "Inference · prefill · decode",
                 "B60 runs the same XPU kernels with less compute/memory than B70.", "B70 runs the same logical graph with more compute/memory headroom.",
                 "The functional flow is identical; capacity and speed differ.")

        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        add_step(11, "Separate newly generated tokens", "PASS",
                 "new_tokens = output_ids[0][input_length:]",
                 f"new_tokens={int(new_tokens.shape[-1])}; token IDs={new_tokens[:16].tolist()}",
                 "Separate the completion from the original prompt portion.",
                 "Generated sequences often contain prompt + completion; slicing isolates just the model answer.",
                 "Sequence slicing",
                 "Same slicing on B60.", "Same slicing on B70.",
                 "No meaningful hardware difference.")

        text_out = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        result["output"] = text_out
        add_step(12, "Decode token IDs back to text", "PASS" if text_out else "FAIL",
                 "tokenizer.decode(new_tokens, skip_special_tokens=True)",
                 text_out if text_out else "<empty output>",
                 "Convert generated token IDs back into readable natural language.",
                 "The tokenizer reverses vocabulary IDs into text pieces and removes control tokens.",
                 "Decoding / post-processing",
                 "Same host-side decode on B60.", "Same host-side decode on B70.",
                 "Decoding is mostly CPU-side and hardware-independent.")

        if not text_out:
            result["error"] = "Empty generated output."
            add_step(13, "Apply functional PASS/FAIL gate", "FAIL",
                     "assert generated_text.strip() != ''",
                     "FAIL: generation completed but output is empty.",
                     "Produce a narrow functional verdict.",
                     "The verifier only checks that the inference path completed and returned usable text.",
                     "Functional qualification gate",
                     "B60 PASS uses same gate.", "B70 PASS uses same gate.",
                     "Acceptance criteria are the same across hardware.")
            return result

        add_step(13, "Apply functional PASS/FAIL gate", "PASS",
                 "PASS = tokenizer_load && model_load && device_placement && generation && non_empty_output",
                 "PASS: end-to-end inference succeeded and produced non-empty text.",
                 "Produce a narrow functional verdict before moving to broader qualification phases.",
                 "This confirms basic software and inference functionality but not stability, parity, or performance.",
                 "Functional qualification",
                 "B60 PASS requires the same gates on B60/XPU.", "B70 PASS requires the same gates on B70/XPU.",
                 "A model can pass here and still fail later in stability/parity/performance work.")
        del model, tokenizer, inputs
        cleanup_accelerator(device)
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        cleanup_accelerator(locals().get("device", locals().get("target_device", None)))
        result["steps"].append({
            "number": len(result["steps"]) + 1,
            "name": "Failure captured",
            "status": "FAIL",
            "command": "exception handler",
            "output": result["error"],
            "purpose": "Capture the exact point and reason the flow stopped.",
            "behind_scenes": "The traceback preserves the failing Python call chain for root-cause classification.",
            "concept": "Failure isolation / reproducibility",
            "b60": "Classify whether failure is XPU/runtime, memory, dtype, operator, driver or model-level on B60.",
            "b70": "Use the same classification on B70.",
            "why_diff": "The debugging method is the same; hardware capacity may change which failure appears."
        })
        return result

# -----------------------------
# Phase 3: robustness / repeatability qualification
# -----------------------------
def run_phase3_qualification(model_id, base_prompt, max_new_tokens, prefer_device, dtype_choice,
                             trust_remote_code=False, repeats=3, long_repeat_factor=60):
    result = {
        "steps": [], "output": "", "samples": [], "device": None, "dtype": None,
        "error": None, "traceback": None, "summary": {}
    }

    def add_step(number, name, status, command, output, purpose, behind_scenes, concept, b60, b70, why_diff):
        result["steps"].append({
            "number": number, "name": name, "status": status, "command": command,
            "output": output, "purpose": purpose, "behind_scenes": behind_scenes,
            "concept": concept, "b60": b60, "b70": b70, "why_diff": why_diff
        })

    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM

        add_step(1, "Import PyTorch + Transformers", "PASS",
                 "import torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM",
                 f"torch={torch.__version__}; transformers={transformers.__version__}",
                 "Initialize the runtime for repeated qualification tests.",
                 "Phase 3 still depends on the same tensor/runtime/model abstractions as Phase 2.",
                 "Framework/runtime reuse",
                 "Same import flow for B60.", "Same import flow for B70.",
                 "Phase 3 changes test breadth, not the fundamental API surface.")

        device = resolve_device(prefer_device)
        result["device"] = device
        dtype = selected_dtype(device, dtype_choice)
        result["dtype"] = str(dtype).replace("torch.", "")
        add_step(2, "Create qualification test matrix", "PASS",
                 "repeats=N; prompts=[base, alternate, long-context]",
                 f"repeats={repeats}; max_new_tokens={max_new_tokens}; long_repeat_factor={long_repeat_factor}",
                 "Define what robustness means for this lightweight tool.",
                 "The model will be exercised repeatedly, with prompt variations and with a longer-context stress input.",
                 "Qualification matrix / test design",
                 "B60 uses the same matrix but may hit memory/stability boundaries sooner.",
                 "B70 uses the same matrix with more headroom for long inputs.",
                 "Same logic; different hardware headroom can change where failures appear.")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = load_causal_model(model_id, dtype, trust_remote_code)
        model.to(device)
        add_step(3, "Load tokenizer + model once for qualification", "PASS",
                 "tokenizer = AutoTokenizer.from_pretrained(...)\nmodel = AutoModelForCausalLM.from_pretrained(...)\nmodel.to(device)",
                 f"Loaded {model.__class__.__name__} on {device} as {result['dtype']}",
                 "Reuse a single model instance while evaluating repeatability and stability.",
                 "Loading once removes repeated initialization noise so the phase focuses more on run-to-run behavior.",
                 "Stable loaded inference session",
                 "Same logic on B60.", "Same logic on B70.",
                 "Difference shows up as memory pressure and execution margin, not API changes.")

        prompts = [
            ("base", base_prompt),
            ("alternate", "Explain tokenization in one short sentence."),
            ("long_context", "AI inference system design. " * long_repeat_factor),
        ]
        add_step(4, "Prepare prompt suite", "PASS",
                 "prompts = [base_prompt, alternate_prompt, long_context_prompt]",
                 f"Prompt categories: {[name for name, _ in prompts]}",
                 "Create a small but varied prompt suite for lightweight qualification.",
                 "Phase 3 should test more than one example so failures are less likely to be hidden by a single lucky prompt.",
                 "Prompt coverage",
                 "Long prompts can expose B60 memory pressure earlier.",
                 "B70 can often tolerate longer prompts/context with more headroom.",
                 "Prompt length affects KV-cache size and memory usage on both devices.")

        # Repeatability test
        base_outputs = []
        repeat_times = []
        for i in range(repeats):
            messages = [{"role": "user", "content": base_prompt}]
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = base_prompt
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.no_grad():
                out = deterministic_generate(model, 
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "xpu" and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            elapsed = time.perf_counter() - t0
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            base_outputs.append(decoded)
            repeat_times.append(elapsed)
            result["samples"].append(f"Repeat {i+1}: {decoded}")
            del out, new_tokens, inputs
            gc.collect()
        unique_outputs = len(set(base_outputs))
        deterministic = unique_outputs == 1 and all(x != "" for x in base_outputs)
        add_step(5, "Run deterministic repeatability loop", "PASS" if deterministic else "WARN",
                 "for i in range(repeats): model.generate(..., do_sample=False)",
                 f"runs={repeats}; unique_outputs={unique_outputs}; avg_time={sum(repeat_times)/len(repeat_times):.2f}s",
                 "Check that the same deterministic prompt produces stable behavior across repeated runs.",
                 "With sampling disabled, a healthy runtime should usually produce the same output for the same prompt/backend/settings.",
                 "Repeatability / deterministic inference",
                 "B60 may still pass determinism, but can reveal thermal/runtime/memory instability earlier under repetition.",
                 "B70 runs the same test with more hardware headroom.",
                 "Differences usually come from runtime stability or capacity pressure rather than Hugging Face logic.")

        # Prompt suite stability
        suite_results = []
        for name, ptext in prompts:
            messages = [{"role": "user", "content": ptext}]
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = ptext
            inputs = tokenizer(text, return_tensors="pt")
            tok_len = int(inputs["input_ids"].shape[-1])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = deterministic_generate(model, 
                    **inputs, max_new_tokens=min(max_new_tokens, 16), do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            suite_results.append((name, tok_len, bool(decoded), decoded[:120]))
            del out, new_tokens, inputs
            gc.collect()
        suite_ok = all(x[2] for x in suite_results)
        suite_summary = "; ".join([f"{n}:tok={t},ok={ok}" for n, t, ok, _ in suite_results])
        add_step(6, "Run prompt-variation stability suite", "PASS" if suite_ok else "WARN",
                 "for prompt in prompts: model.generate(...)",
                 suite_summary,
                 "Ensure the model works across multiple input styles, not just one example.",
                 "Different prompt structures can trigger different tokenization lengths and decode paths.",
                 "Input coverage / scenario stability",
                 "Longer variants may expose B60 memory limits or operator issues sooner.",
                 "B70 generally has more room for prompt/context growth.",
                 "Same logical graph; larger inputs stress memory/cache behavior differently.")

        # Long context token boundary
        long_prompt_tokens = [x[1] for x in suite_results if x[0] == "long_context"][0]
        add_step(7, "Evaluate long-context boundary evidence", "PASS" if long_prompt_tokens > 0 else "FAIL",
                 "token_count = tokenizer(long_context_prompt).input_ids.shape[-1]",
                 f"long_context_tokens={long_prompt_tokens}; memory_report={sample_memory_report(device)}",
                 "Measure whether the run covered a meaningfully longer prompt than the simple smoke test.",
                 "Longer prompt length grows prefill work and usually grows attention/KV-cache memory demands.",
                 "Context length / prefill pressure",
                 "B60 is more likely to hit memory pressure earlier as context grows.",
                 "B70's 32 GB often gives more margin for context expansion.",
                 "The main difference is available memory/capacity, not a different model API.")

        success_count = sum(1 for _, _, ok, _ in suite_results if ok)
        all_runs = repeats + len(suite_results)
        pass_rate = (repeats + success_count) / all_runs if all_runs else 0.0
        deterministic_rate = 1.0 if deterministic else max(0.0, (repeats - unique_outputs + 1) / max(repeats, 1))
        result["summary"] = {
            "pass_rate": pass_rate,
            "deterministic": deterministic,
            "unique_repeat_outputs": unique_outputs,
            "avg_repeat_time_s": round(sum(repeat_times)/len(repeat_times), 3),
            "memory": sample_memory_report(device),
        }
        add_step(8, "Aggregate robustness metrics", "PASS",
                 "pass_rate, deterministic_flag, repeat_time_avg, memory_report",
                 f"pass_rate={pass_rate:.2%}; deterministic={deterministic}; avg_repeat_time={result['summary']['avg_repeat_time_s']}s",
                 "Summarize the qualification evidence into a compact stability scorecard.",
                 "This aggregates run-level evidence rather than relying on a single successful output.",
                 "Qualification metrics / run aggregation",
                 "B60 qualification may degrade earlier for larger prompt sizes or tighter memory margin.",
                 "B70 usually has better capacity margin, which can improve robustness on larger tests.",
                 "Metrics are interpreted the same way, but available headroom can change outcomes.")

        verdict_pass = suite_ok and deterministic and pass_rate >= 0.90
        result["output"] = (
            f"Qualification verdict={'PASS' if verdict_pass else 'WARN/FAIL'} | "
            f"pass_rate={pass_rate:.2%} | deterministic={deterministic} | "
            f"avg_repeat_time={result['summary']['avg_repeat_time_s']}s"
        )
        add_step(9, "Apply Phase 3 qualification verdict", "PASS" if verdict_pass else "WARN",
                 "PASS if repeatability + prompt-variation stability + long-context sanity are acceptable",
                 result["output"],
                 "Provide a bridge verdict between single-run functionality and later benchmarking.",
                 "Phase 3 asks: can the model run repeatedly and credibly across small variations without crashing or drifting unexpectedly?",
                 "Robustness / repeatability qualification",
                 "B60 PASS means the workload is stable on B60, not merely on CPU.",
                 "B70 PASS means the workload is stable on B70, not merely on CPU.",
                 "A model can function once yet still fail qualification when prompts vary or repetition exposes instability.")
        del model, tokenizer
        cleanup_accelerator(device)
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        cleanup_accelerator(locals().get("device", locals().get("target_device", None)))
        result["steps"].append({
            "number": len(result["steps"]) + 1,
            "name": "Failure captured",
            "status": "FAIL",
            "command": "exception handler",
            "output": result["error"],
            "purpose": "Capture the exact point and reason the qualification flow stopped.",
            "behind_scenes": "The traceback helps classify whether the qualification failure came from model load, memory, runtime, or generation.",
            "concept": "Failure isolation / robustness debugging",
            "b60": "Classify whether repetition or longer prompts exposed B60-specific limits.",
            "b70": "Classify whether the same issue appears on B70 or only on smaller headroom hardware.",
            "why_diff": "Qualification often reveals capacity/stability gaps that a single-run smoke test misses."
        })
        return result

# -----------------------------
# Phase 4: correctness / parity
# -----------------------------
def run_phase4_parity(model_id, prompt, max_new_tokens, prefer_device, dtype_choice, trust_remote_code=False):
    result = {
        "steps": [], "output": "", "samples": [], "device": None, "dtype": None,
        "error": None, "traceback": None, "summary": {}
    }

    def add_step(number, name, status, command, output, purpose, behind_scenes, concept, b60, b70, why_diff):
        result["steps"].append({
            "number": number, "name": name, "status": status, "command": command,
            "output": output, "purpose": purpose, "behind_scenes": behind_scenes,
            "concept": concept, "b60": b60, "b70": b70, "why_diff": why_diff
        })

    try:
        import torch
        import transformers
        from transformers import AutoTokenizer

        add_step(
            1, "Import PyTorch + Transformers", "PASS",
            "import torch\\nfrom transformers import AutoTokenizer",
            f"torch={torch.__version__}; transformers={transformers.__version__}",
            "Initialize the libraries needed for reference-vs-target comparison.",
            "Phase 4 compares behavior across two execution backends only when two distinct backends actually exist.",
            "Framework/runtime comparison setup",
            "Same setup for B60.", "Same setup for B70.",
            "Hardware changes the target backend, not the parity methodology."
        )

        target_device = resolve_device(prefer_device)
        result["device"] = target_device
        target_dtype = selected_dtype(target_device, dtype_choice)
        result["dtype"] = str(target_dtype).replace("torch.", "")

        add_step(
            2, "Define reference and target paths", "PASS",
            "reference_device='cpu'; target_device=selected_backend",
            f"reference=cpu/fp32; target={target_device}/{result['dtype']}",
            "Create a CPU reference path and a distinct target path when one is available.",
            "CPU is the practical reference. A meaningful backend parity test requires the target backend to be different from CPU.",
            "Reference-vs-target methodology",
            "B60 becomes the XPU target when physically available.",
            "B70 becomes the XPU target when physically available.",
            "CPU-vs-CPU does not validate accelerator correctness, so duplicating the model would waste memory without adding evidence."
        )

        if target_device == "cpu":
            result["summary"] = {
                "mode": "educational_skip",
                "skipped_duplicate_cpu_run": True,
                "same_text_exact": None,
                "same_text_normalized": None,
                "top1_match": None,
                "top5_overlap": None,
                "max_abs_diff": None,
                "mean_abs_diff": None,
            }

            add_step(
                3, "Detect same-backend parity condition", "PASS",
                "if target_device == 'cpu': skip_duplicate_backend_run()",
                "Target backend is CPU, which is the same backend as the CPU reference.",
                "Prevent an expensive comparison that cannot prove accelerator parity.",
                "A parity test is useful only when comparing genuinely different execution paths.",
                "Parity test validity",
                "On a B60 machine, target resolves to XPU and this skip does not happen.",
                "On a B70 machine, target resolves to XPU and this skip does not happen.",
                "Streamlit Community Cloud is commonly CPU-only, so CPU-vs-CPU would only duplicate work and memory."
            )

            add_step(
                4, "Skip duplicate model loading", "PASS",
                "# intentionally do not load CPU reference + CPU target copies",
                "Duplicate CPU model load skipped to avoid unnecessary RAM pressure.",
                "Avoid the main cause of Phase 4 hanging on constrained hosted environments.",
                "Loading the same model twice in one process can push a small Streamlit instance toward memory pressure, swapping, or process termination.",
                "Memory-aware qualification",
                "B60 path loads CPU reference once, releases it, then loads B60/XPU target.",
                "B70 path loads CPU reference once, releases it, then loads B70/XPU target.",
                "The safe hosted path avoids a second CPU copy because it provides no additional parity evidence."
            )

            add_step(
                5, "Explain the meaningful parity path", "PASS",
                "CPU reference  ->  XPU target  ->  compare text/logits",
                "Meaningful parity requires CPU vs CUDA/XPU, not CPU vs CPU.",
                "Teach the TPM what should happen on real accelerator infrastructure.",
                "The same prompt, tokenizer, generation settings, and model stay constant while only the backend changes.",
                "Controlled backend comparison",
                "B60: compare CPU/fp32 reference against B60/XPU target.",
                "B70: compare CPU/fp32 reference against B70/XPU target.",
                "Keeping workload constant isolates backend/runtime differences."
            )

            result["output"] = (
                "PARITY DRY-RUN / SKIPPED: selected target is CPU, the same as the CPU reference. "
                "A second model load was intentionally skipped to avoid unnecessary Streamlit Cloud RAM pressure. "
                "Run this same tab on CUDA/XPU hardware for a meaningful CPU-vs-accelerator parity comparison."
            )

            add_step(
                6, "Apply hosted CPU parity verdict", "PASS",
                "return educational_skip when reference_backend == target_backend",
                result["output"],
                "Finish Phase 4 safely without pretending CPU-vs-CPU proves accelerator correctness.",
                "This is an educational PASS for the workflow itself, not a hardware parity qualification.",
                "Scoped parity verdict",
                "B60 hardware still requires a real CPU-vs-B60/XPU run.",
                "B70 hardware still requires a real CPU-vs-B70/XPU run.",
                "Skipping invalid work is better than reporting a misleading parity result."
            )
            return result

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        add_step(
            3, "Load shared tokenizer", "PASS",
            f"AutoTokenizer.from_pretrained('{model_id}')",
            f"{tokenizer.__class__.__name__} loaded",
            "Guarantee CPU and target runs use identical preprocessing.",
            "One shared tokenizer prevents differences caused by prompt preprocessing.",
            "Shared preprocessing baseline",
            "Same tokenizer for CPU and B60.", "Same tokenizer for CPU and B70.",
            "Parity requires identical token IDs across both paths."
        )

        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt
        cpu_inputs = tokenizer(formatted, return_tensors="pt")
        input_len = int(cpu_inputs["input_ids"].shape[-1])

        add_step(
            4, "Prepare identical reference prompt", "PASS",
            "formatted = tokenizer.apply_chat_template(...)\\ncpu_inputs = tokenizer(formatted, return_tensors='pt')",
            f"formatted_length={len(formatted)} chars; input_tokens={input_len}",
            "Ensure both runs see exactly the same prompt and token IDs.",
            "A controlled experiment changes only the execution backend.",
            "Controlled experimental input",
            "Same token IDs for B60 comparison.", "Same token IDs for B70 comparison.",
            "Any input mismatch would invalidate the comparison."
        )

        cpu_model = load_causal_model(model_id, torch.float32, trust_remote_code).to("cpu")
        with torch.no_grad():
            cpu_logits = cpu_model(**cpu_inputs).logits[0, -1, :].float().cpu()
            cpu_gen = deterministic_generate(
                cpu_model, **cpu_inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        cpu_new = cpu_gen[0][input_len:]
        cpu_text = tokenizer.decode(cpu_new, skip_special_tokens=True).strip()
        cpu_top5_ids = torch.topk(cpu_logits, k=5).indices.tolist()
        cpu_logits_ref = cpu_logits.detach().clone()
        result["samples"].append(f"CPU output: {cpu_text}")

        add_step(
            5, "Run CPU reference inference", "PASS",
            "cpu_model(...).logits\\ncpu_model.generate(...)",
            f"cpu_output_len={len(cpu_new)}; cpu_top5={cpu_top5_ids}; cpu_text={cpu_text[:120]}",
            "Capture reference text and next-token evidence before target execution.",
            "The CPU path acts as a practical correctness baseline.",
            "Reference inference / logits baseline",
            "B60 target will be compared against this CPU evidence.",
            "B70 target will be compared against this CPU evidence.",
            "CPU is widely available and easier to debug as a baseline."
        )

        del cpu_model, cpu_gen, cpu_logits, cpu_new
        gc.collect()
        cleanup_accelerator("cpu")

        add_step(
            6, "Release CPU reference model before target load", "PASS",
            "del cpu_model, cpu_gen, cpu_logits\\ngc.collect()\\ncleanup_accelerator('cpu')",
            "CPU reference model released before loading target model.",
            "Minimize peak process memory and avoid holding two model copies simultaneously.",
            "Memory lifecycle / resource cleanup",
            "Important before B60 target load on systems with constrained host RAM.",
            "Important before B70 target load as well, even though B70 has more VRAM.",
            "Host RAM pressure can still be a bottleneck regardless of accelerator VRAM."
        )

        target_model = load_causal_model(model_id, target_dtype, trust_remote_code).to(target_device)
        tgt_inputs = {k: v.to(target_device) for k, v in cpu_inputs.items()}
        with torch.no_grad():
            tgt_logits = target_model(**tgt_inputs).logits[0, -1, :].float().cpu()
            tgt_gen = deterministic_generate(
                target_model, **tgt_inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        if target_device == "cuda":
            torch.cuda.synchronize()
        elif target_device == "xpu" and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        tgt_new = tgt_gen[0][input_len:]
        tgt_text = tokenizer.decode(tgt_new, skip_special_tokens=True).strip()
        tgt_top5_ids = torch.topk(tgt_logits, k=5).indices.tolist()
        result["samples"].append(f"Target output ({target_device}): {tgt_text}")

        add_step(
            7, "Run target-backend inference", "PASS",
            "target_model(...).logits\\ntarget_model.generate(...)",
            f"target_output_len={len(tgt_new)}; target_top5={tgt_top5_ids}; target_text={tgt_text[:120]}",
            "Capture equivalent evidence on the accelerator backend.",
            "The same model/input executes using the target runtime/device kernels.",
            "Target inference / backend-specific execution",
            "On B60 this is the XPU/B60 execution path.",
            "On B70 this is the XPU/B70 execution path.",
            "Only the backend/hardware should differ from the CPU reference."
        )

        same_text_exact = cpu_text == tgt_text
        same_text_norm = normalize_text(cpu_text) == normalize_text(tgt_text)
        top1_match = int(cpu_top5_ids[0] == tgt_top5_ids[0])
        top5_overlap = len(set(cpu_top5_ids).intersection(set(tgt_top5_ids)))
        max_abs_diff = float((cpu_logits_ref - tgt_logits).abs().max().item())
        mean_abs_diff = float((cpu_logits_ref - tgt_logits).abs().mean().item())

        result["summary"] = {
            "mode": "real_backend_parity",
            "skipped_duplicate_cpu_run": False,
            "same_text_exact": same_text_exact,
            "same_text_normalized": same_text_norm,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        }

        add_step(
            8, "Compare CPU vs target evidence", "PASS",
            "compare(decoded_text, top1, top5, logits_diff)",
            f"same_text_exact={same_text_exact}; normalized_match={same_text_norm}; "
            f"top1_match={top1_match}; top5_overlap={top5_overlap}; "
            f"max_abs_diff={max_abs_diff:.6f}; mean_abs_diff={mean_abs_diff:.6f}",
            "Convert the two executions into practical parity metrics.",
            "Text comparison gives a human-readable signal; top-token/logit comparisons expose lower-level numeric behavior.",
            "Correctness / parity metrics",
            "B60 can differ numerically from CPU yet remain acceptably close.",
            "B70 can also differ numerically from CPU; exact equality is not always required.",
            "Different precisions and accelerator kernels can create small numeric differences."
        )

        verdict_pass = bool(tgt_text) and (same_text_norm or top1_match == 1 or top5_overlap >= 3)
        result["output"] = (
            f"Parity verdict={'PASS' if verdict_pass else 'WARN/FAIL'} | "
            f"normalized_text_match={same_text_norm} | top1_match={top1_match} | "
            f"top5_overlap={top5_overlap} | max_abs_diff={max_abs_diff:.6f}"
        )

        add_step(
            9, "Apply Phase 4 parity verdict", "PASS" if verdict_pass else "WARN",
            "PASS if target behavior is credible vs CPU using text + token evidence",
            result["output"],
            "Provide a pre-benchmark confidence gate for target-backend correctness.",
            "The goal is credible similarity, not mandatory bitwise identity across CPU and accelerator backends.",
            "Reference parity / correctness gate",
            "B60 PASS means B60 behavior is acceptably close to CPU for this workload.",
            "B70 PASS means B70 behavior is acceptably close to CPU for this workload.",
            "A backend can be functional but still suspicious if parity evidence diverges materially."
        )

        del target_model, tokenizer, tgt_inputs, tgt_gen, tgt_logits, tgt_new, cpu_inputs, cpu_logits_ref
        gc.collect()
        cleanup_accelerator(target_device)
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        try:
            gc.collect()
            cleanup_accelerator(locals().get("target_device", None))
        except Exception:
            pass
        result["steps"].append({
            "number": len(result["steps"]) + 1,
            "name": "Failure captured",
            "status": "FAIL",
            "command": "exception handler",
            "output": result["error"],
            "purpose": "Capture the exact point and reason the parity flow stopped.",
            "behind_scenes": "The traceback helps distinguish CPU reference, memory, target runtime, device, and comparison failures.",
            "concept": "Failure isolation / parity debugging",
            "b60": "Determine whether B60 failed while CPU reference succeeded.",
            "b70": "Determine whether B70 failed while CPU reference succeeded.",
            "why_diff": "Parity is specifically meant to expose backend-specific correctness or runtime problems."
        })
        return result


def _prepare_benchmark_bundle(model_id, prompt, prefer_device, dtype_choice, trust_remote_code=False):
    import torch
    from transformers import AutoTokenizer
    device = resolve_device(prefer_device)
    dtype = selected_dtype(device, dtype_choice)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = load_causal_model(model_id, dtype, trust_remote_code).to(device)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt
    host_inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in host_inputs.items()}
    return tokenizer, model, inputs, device, dtype

def _sync_device(device):
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available() and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
    except Exception:
        pass

def benchmark_loaded_model(model, tokenizer, inputs, device, max_new_tokens, iterations=3, use_cache=True, use_inference_mode=True):
    """Measure one already-loaded model without retaining large intermediate tensors."""
    import torch
    safe = hosted_safe_mode(device)
    effective_iterations = max(1, min(int(iterations), 2 if safe else int(iterations)))
    effective_tokens = max(4, min(int(max_new_tokens), 16 if safe else int(max_new_tokens)))
    input_tokens = int(inputs["input_ids"].shape[-1])
    context = torch.inference_mode if use_inference_mode else torch.no_grad

    # Minimal warm-up.
    with context():
        warm = deterministic_generate(model, **inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id, use_cache=use_cache)
    _sync_device(device)
    del warm

    # TTFT proxy: one generated token.
    t0=time.perf_counter()
    with context():
        one=deterministic_generate(model, **inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, use_cache=use_cache)
    _sync_device(device)
    ttft_proxy_ms=(time.perf_counter()-t0)*1000.0
    del one

    # Prefill: on low-memory hosted CPU skip a separate full-logits forward, because
    # that allocation is unnecessary and can push Community Cloud over RAM limits.
    prefill_is_proxy = safe
    if safe:
        prefill_ms = ttft_proxy_ms
    else:
        forward_kwargs={"use_cache": False}
        try:
            if "logits_to_keep" in inspect.signature(model.forward).parameters:
                forward_kwargs["logits_to_keep"] = 1
        except Exception:
            pass
        p0=time.perf_counter()
        with context():
            pref=model(**inputs, **forward_kwargs)
        _sync_device(device)
        prefill_ms=(time.perf_counter()-p0)*1000.0
        del pref

    e2e=[]; counts=[]; outputs=[]
    for _ in range(effective_iterations):
        g0=time.perf_counter()
        with context():
            out=deterministic_generate(model, **inputs, max_new_tokens=effective_tokens, pad_token_id=tokenizer.eos_token_id, use_cache=use_cache)
        _sync_device(device)
        elapsed=time.perf_counter()-g0
        new=out[0][input_tokens:]
        counts.append(int(new.shape[-1])); e2e.append(elapsed)
        outputs.append(tokenizer.decode(new, skip_special_tokens=True).strip())
        del out,new
        gc.collect()

    avg_e2e_s=sum(e2e)/len(e2e)
    avg_tokens=sum(counts)/len(counts) if counts else 0.0
    tok_s=avg_tokens/avg_e2e_s if avg_e2e_s>0 else 0.0
    # Approximation only; true streaming TTFT/TPOT requires a serving stack or token timestamps.
    approx_decode_s=max(avg_e2e_s-(prefill_ms/1000.0),1e-9)
    approx_tpot_ms=approx_decode_s/max(avg_tokens,1.0)*1000.0
    approx_decode_tok_s=avg_tokens/approx_decode_s if avg_tokens else 0.0
    cv=(statistics.pstdev(e2e)/avg_e2e_s*100.0) if len(e2e)>=2 and avg_e2e_s>0 else 0.0
    nonempty=[x for x in outputs if x]
    return {
        "device":device, "dtype":str(next(model.parameters()).dtype).replace("torch.",""),
        "iterations":effective_iterations, "requested_iterations":int(iterations),
        "effective_max_new_tokens":effective_tokens, "requested_max_new_tokens":int(max_new_tokens),
        "input_tokens":input_tokens, "avg_output_tokens":avg_tokens,
        "prefill_ms":prefill_ms, "prefill_is_proxy":prefill_is_proxy,
        "ttft_proxy_ms":ttft_proxy_ms, "e2e_ms":avg_e2e_s*1000.0,
        "tokens_per_s":tok_s, "approx_decode_tokens_per_s":approx_decode_tok_s,
        "approx_tpot_ms":approx_tpot_ms, "latency_cv_pct":cv,
        "memory":sample_memory_report(device), "output":nonempty[-1] if nonempty else "",
        "unique_output_count":len(set(nonempty)) if nonempty else 0,
        "use_cache":bool(use_cache), "use_inference_mode":bool(use_inference_mode),
        "hosted_safe_mode":safe, "raw_e2e_ms":[x*1000.0 for x in e2e],
    }

def quick_benchmark(model_id, prompt, max_new_tokens, prefer_device, dtype_choice, trust_remote_code=False, iterations=3, use_cache=True, use_inference_mode=True):
    tokenizer=model=inputs=None; device=resolve_device(prefer_device)
    try:
        tokenizer,model,inputs,device,_dtype=_prepare_benchmark_bundle(model_id,prompt,prefer_device,dtype_choice,trust_remote_code)
        return benchmark_loaded_model(model,tokenizer,inputs,device,max_new_tokens,iterations,use_cache,use_inference_mode)
    finally:
        for name in ("inputs","model","tokenizer"):
            if name in locals():
                try: del locals()[name]
                except Exception: pass
        gc.collect(); cleanup_accelerator(device)


# -----------------------------
# Phase 5: performance benchmark
# -----------------------------
def run_phase5_benchmark(model_id,prompt,max_new_tokens,prefer_device,dtype_choice,trust_remote_code=False,iterations=3):
    result={"steps":[],"output":"","samples":[],"error":None,"traceback":None,"summary":{}}
    def add(n,name,status,cmd,out,purpose,behind,concept,b60,b70,why):
        result["steps"].append({"number":n,"name":name,"status":status,"command":cmd,"output":out,"purpose":purpose,"behind_scenes":behind,"concept":concept,"b60":b60,"b70":b70,"why_diff":why})
    try:
        safe=hosted_safe_mode(resolve_device(prefer_device))
        add(1,"Define a fixed workload","PASS","freeze model + prompt + tokens + dtype + device",f"requested iterations={iterations}; requested max_new_tokens={max_new_tokens}; hosted_safe_mode={safe}","Make benchmark results comparable.","A valid benchmark changes as few variables as possible.","Controlled benchmark methodology","Use the identical workload on B60.","Use the identical workload on B70.","Hardware can differ; workload definition should not.")
        bm=quick_benchmark(model_id,prompt,max_new_tokens,prefer_device,dtype_choice,trust_remote_code,iterations,True,True)
        bm["baseline_signature"]={"model_id":model_id,"prompt":prompt,"max_new_tokens":max_new_tokens,"device":resolve_device(prefer_device),"requested_device":prefer_device,"dtype_choice":dtype_choice,"benchmark_engine":"quick_benchmark_v9","use_cache":True,"use_inference_mode":True}
        result["summary"]=bm; result["output"]=bm["output"]
        add(2,"Warm up then measure","PASS","warm-up → 1-token TTFT proxy → measured generation",f"effective iterations={bm['iterations']}; effective output cap={bm['effective_max_new_tokens']} tokens","Reduce cold-start noise and gather lightweight repeatable timings.","Hosted-safe mode automatically reduces iterations/output length to stay inside small Community Cloud memory/CPU budgets.","Steady-state benchmarking","Same procedure on B60.","Same procedure on B70.","Hosted CPU needs conservative workload sizing; dedicated accelerators can run fuller workloads.")
        prefill_label="proxy" if bm.get("prefill_is_proxy") else "measured"
        add(3,"Measure prompt/prefill cost","PASS", "model forward (or hosted-safe TTFT proxy)",f"prefill={bm['prefill_ms']:.2f} ms ({prefill_label}); input_tokens={bm['input_tokens']}","Estimate the cost before iterative decoding.","On constrained CPU hosts a separate full-logits forward is skipped to avoid an unnecessary RAM spike.","Prefill / prompt processing","Measure directly on B60 when resources allow.","Measure directly on B70 when resources allow.","The concept is the same; hosted-safe mode sacrifices precision for reliability.")
        add(4,"Measure latency + throughput","PASS","generate(1 token) + generate(N tokens)",f"TTFT proxy={bm['ttft_proxy_ms']:.2f} ms; E2E={bm['e2e_ms']:.2f} ms; tok/s={bm['tokens_per_s']:.2f}; TPOT~={bm['approx_tpot_ms']:.2f} ms/tok","Capture user-visible latency and generation throughput.","TTFT/TPOT here are educational proxies because this is not a streaming serving engine.","LLM inference KPIs","B60 uses the same metrics.","B70 uses the same metrics.","Performance differs; KPI definitions do not.")
        add(5,"Check benchmark stability","PASS" if bm['latency_cv_pct']<=20 else "WARN","CV = stddev / mean × 100",f"latency CV={bm['latency_cv_pct']:.2f}%","Detect noisy results before using them as a baseline.","High variance can come from shared-cloud CPU scheduling and is not necessarily a model bug.","Measurement variance","Dedicated B60 hosts should normally be less noisy.","Dedicated B70 hosts should normally be less noisy.","Community Cloud is shared infrastructure, so timing noise is expected.")
        verdict=bm['tokens_per_s']>0 and bool(bm['output'])
        add(6,"Create benchmark baseline","PASS" if verdict else "FAIL","save metrics + workload signature",f"E2E={bm['e2e_ms']:.2f} ms; throughput={bm['tokens_per_s']:.2f} tok/s","Create evidence that later optimization/regression phases can reuse.","A baseline includes both metrics and the exact workload definition.","Performance baseline","Store B60-specific baseline.","Store B70-specific baseline.","Do not compare unlike devices/workloads as regressions.")
        return result
    except Exception as e:
        result['error']=f"{type(e).__name__}: {e}"; result['traceback']=traceback.format_exc()
        add(len(result['steps'])+1,"Benchmark could not complete","FAIL","exception handler",result['error'],"Expose a useful failure instead of leaving the tab spinning.","Most hosted failures are resource/dependency issues; the traceback is retained for diagnosis.","Benchmark failure isolation","Check B60 runtime/memory.","Check B70 runtime/memory.","Resource and backend failures require different fixes.")
        cleanup_accelerator(resolve_device(prefer_device)); return result

# -----------------------------
# Phase 6: optimization
# -----------------------------
def run_phase6_optimization(model_id,prompt,max_new_tokens,prefer_device,dtype_choice,trust_remote_code=False,iterations=2):
    result={"steps":[],"output":"","samples":[],"error":None,"traceback":None,"summary":{}}
    def add(n,name,status,cmd,out,purpose,behind,concept,b60,b70,why): result['steps'].append({"number":n,"name":name,"status":status,"command":cmd,"output":out,"purpose":purpose,"behind_scenes":behind,"concept":concept,"b60":b60,"b70":b70,"why_diff":why})
    tokenizer=model=inputs=None; device=resolve_device(prefer_device)
    try:
        add(1,"Choose one controlled optimization","PASS","baseline: cache off + no_grad; candidate: cache on + inference_mode","Same loaded model and same workload used for both measurements.","Avoid repeated model loads while isolating the optimization change.","Controlled tuning experiment","Run the same A/B test on B60.","Run the same A/B test on B70.","The gain can vary by hardware even when the optimization is identical.")
        tokenizer,model,inputs,device,_dtype=_prepare_benchmark_bundle(model_id,prompt,prefer_device,dtype_choice,trust_remote_code)
        baseline=benchmark_loaded_model(model,tokenizer,inputs,device,max_new_tokens,iterations,False,False)
        add(2,"Measure baseline","PASS","benchmark_loaded_model(..., use_cache=False, inference_mode=False)",f"E2E={baseline['e2e_ms']:.2f} ms; tok/s={baseline['tokens_per_s']:.2f}","Establish the before value.","The model remains loaded so this phase avoids a second checkpoint load and large RAM spike.","Before measurement","Measure B60 baseline locally.","Measure B70 baseline locally.","Each device needs its own baseline.")
        optimized=benchmark_loaded_model(model,tokenizer,inputs,device,max_new_tokens,iterations,True,True)
        add(3,"Measure candidate optimization","PASS","benchmark_loaded_model(..., use_cache=True, inference_mode=True)",f"E2E={optimized['e2e_ms']:.2f} ms; tok/s={optimized['tokens_per_s']:.2f}","Measure the exact same workload after the tuning change.","KV cache trades memory for less repeated attention compute; inference_mode reduces autograd overhead.","KV cache + inference mode","B60 has less memory headroom for cache.","B70 has more cache headroom.","The speed/memory tradeoff differs by device.")
        speedup=baseline['e2e_ms']/optimized['e2e_ms'] if optimized['e2e_ms'] else 0.0
        gain=(baseline['e2e_ms']-optimized['e2e_ms'])/baseline['e2e_ms']*100 if baseline['e2e_ms'] else 0.0
        tgain=(optimized['tokens_per_s']-baseline['tokens_per_s'])/baseline['tokens_per_s']*100 if baseline['tokens_per_s'] else 0.0
        parity=normalize_text(baseline['output'])==normalize_text(optimized['output'])
        result['summary']={"speedup":speedup,"latency_improvement_pct":gain,"throughput_gain_pct":tgain,"output_parity":parity,"baseline_e2e_ms":baseline['e2e_ms'],"optimized_e2e_ms":optimized['e2e_ms'],"baseline_tok_s":baseline['tokens_per_s'],"optimized_tok_s":optimized['tokens_per_s'],"hosted_safe_mode":optimized.get('hosted_safe_mode',False)}
        result['samples']=[f"Baseline output: {baseline['output']}",f"Optimized output: {optimized['output']}"]
        add(4,"Calculate before/after delta","PASS","speedup = baseline / candidate",f"speedup={speedup:.3f}x; latency gain={gain:.2f}%; throughput gain={tgain:.2f}%","Decide if the optimization produced measurable value.","On noisy shared CPU hosts small negative/positive changes should be treated cautiously.","Optimization effectiveness","Validate gain on B60.","Validate gain on B70.","Optimization is workload- and hardware-specific.")
        status="PASS" if parity and gain>0 else "WARN"
        result['output']=f"Optimization verdict={'PASS' if status=='PASS' else 'NO CLEAR GAIN / WARN'} | speedup={speedup:.3f}x | output parity={parity}"
        add(5,"Apply optimization verdict",status,"require parity; prefer measured positive gain",result['output'],"Keep only optimizations that preserve behavior and show evidence of benefit.","A WARN is not an app error; it means the candidate did not clearly improve this noisy/small workload.","Optimization qualification","Promote only proven B60 gains.","Promote only proven B70 gains.","Theoretically good tuning can be neutral or worse on a given workload.")
        return result
    except Exception as e:
        result['error']=f"{type(e).__name__}: {e}"; result['traceback']=traceback.format_exc(); add(len(result['steps'])+1,"Optimization could not complete","FAIL","exception handler",result['error'],"Return a diagnostic instead of a stuck spinner.","The exact baseline/candidate stage remains visible in the completed steps.","Optimization debugging","Check B60 setup/resources.","Check B70 setup/resources.","Failures can be backend- or capacity-specific."); return result
    finally:
        try: del inputs,model,tokenizer
        except Exception: pass
        gc.collect(); cleanup_accelerator(device)

# -----------------------------
# Phase 7: regression
# -----------------------------
def run_phase7_regression(model_id,prompt,max_new_tokens,prefer_device,dtype_choice,baseline,trust_remote_code=False,threshold_pct=10.0):
    result={"steps":[],"output":"","samples":[],"error":None,"traceback":None,"summary":{}}
    def add(n,name,status,cmd,out,purpose,behind,concept,b60,b70,why): result['steps'].append({"number":n,"name":name,"status":status,"command":cmd,"output":out,"purpose":purpose,"behind_scenes":behind,"concept":concept,"b60":b60,"b70":b70,"why_diff":why})
    if not baseline:
        result['summary']={'mode':'prerequisite_missing'}; result['output']='Regression not run: no Phase 5 baseline is available.'
        add(1,"Check baseline prerequisite","SKIPPED","require saved Phase 5 baseline",result['output'],"Avoid an invalid regression comparison.","A regression delta has no meaning without a known-good reference.","Regression prerequisite","Need a B60 baseline.","Need a B70 baseline.","Baseline is a prerequisite, not a runtime error."); return result
    try:
        signature=baseline.get('baseline_signature',{})
        current={"model_id":model_id,"prompt":prompt,"max_new_tokens":max_new_tokens,"device":resolve_device(prefer_device),"requested_device":prefer_device,"dtype_choice":dtype_choice,"benchmark_engine":"quick_benchmark_v9","use_cache":True,"use_inference_mode":True}
        mismatches=[k for k,v in current.items() if signature and signature.get(k)!=v]
        if mismatches:
            result['summary']={'mode':'baseline_mismatch','mismatches':mismatches}; result['output']='Regression skipped because the candidate workload differs from the saved baseline: '+', '.join(mismatches)
            add(1,"Validate baseline signature","SKIPPED","compare candidate signature to baseline",result['output'],"Prevent misleading regression percentages.","Model/prompt/tokens/device/dtype/benchmark method must match.","Regression test validity","B60 baseline must match B60 candidate.","B70 baseline must match B70 candidate.","A mismatch is a test-definition problem, not a model failure."); return result
        add(1,"Load known-good baseline","PASS","baseline = Phase 5 metrics",f"baseline E2E={baseline['e2e_ms']:.2f} ms; tok/s={baseline['tokens_per_s']:.2f}; threshold={threshold_pct:.1f}%","Anchor the comparison to accepted evidence.","The baseline contains the workload signature as well as performance metrics.","Known-good baseline","Use B60 baseline for B60.","Use B70 baseline for B70.","Cross-device deltas are not regressions.")
        current_bm=quick_benchmark(model_id,prompt,max_new_tokens,prefer_device,dtype_choice,trust_remote_code,2,True,True)
        lat=(current_bm['e2e_ms']-baseline['e2e_ms'])/baseline['e2e_ms']*100 if baseline['e2e_ms'] else 0.0
        thr=(current_bm['tokens_per_s']-baseline['tokens_per_s'])/baseline['tokens_per_s']*100 if baseline['tokens_per_s'] else 0.0
        parity=normalize_text(current_bm['output'])==normalize_text(baseline.get('output',''))
        pass_gate=lat<=threshold_pct and thr>=-threshold_pct and parity
        result['summary']={"mode":"real_regression","latency_delta_pct":lat,"throughput_delta_pct":thr,"output_parity":parity,"threshold_pct":threshold_pct,"current_e2e_ms":current_bm['e2e_ms'],"current_tok_s":current_bm['tokens_per_s']}
        result['samples']=[f"Baseline output: {baseline.get('output','')}",f"Current output: {current_bm['output']}"]
        add(2,"Run candidate with the same benchmark engine","PASS","quick_benchmark_v9(candidate)",f"current E2E={current_bm['e2e_ms']:.2f} ms; tok/s={current_bm['tokens_per_s']:.2f}","Measure the candidate using the same methodology.","The shared benchmark engine prevents measurement-method drift between phases.","Candidate regression run","Same B60 config.","Same B70 config.","Equivalent environments are essential.")
        add(3,"Compare against threshold","PASS" if pass_gate else "FAIL","latency Δ + throughput Δ + output parity",f"latency Δ={lat:.2f}%; throughput Δ={thr:.2f}%; output parity={parity}","Turn performance evidence into a release guardrail.","A threshold tolerates normal noise but blocks material deterioration.","Regression gate","Apply B60 threshold to B60.","Apply B70 threshold to B70.","Threshold semantics match; baselines are device-specific.")
        result['output']=f"Regression verdict={'PASS' if pass_gate else 'FAIL'} | latency={lat:.2f}% | throughput={thr:.2f}% | parity={parity}"
        add(4,"Apply regression verdict","PASS" if pass_gate else "FAIL","release only when within threshold",result['output'],"Protect known-good behavior over time.","A true FAIL here is evidence of a candidate regression, not an app crash.","Release safety gate","Block B60 candidate on failure.","Block B70 candidate on failure.","Each platform needs its own accepted baseline."); return result
    except Exception as e:
        result['error']=f"{type(e).__name__}: {e}"; result['traceback']=traceback.format_exc(); add(len(result['steps'])+1,"Regression execution could not complete","FAIL","exception handler",result['error'],"Return actionable diagnostics.","The test itself failed to execute; this is different from a measured performance regression.","Pipeline reliability","Check B60 pipeline.","Check B70 pipeline.","Execution errors and regression failures should not be conflated."); return result

# -----------------------------
# Phase 8: production qualification
# -----------------------------
def run_phase8_production_qualification(evidence,max_e2e_ms=5000.0,min_tok_s=1.0):
    result={"steps":[],"output":"","error":None,"traceback":None,"summary":{}}
    def add(n,name,status,cmd,out,purpose,behind,concept,b60,b70,why): result['steps'].append({"number":n,"name":name,"status":status,"command":cmd,"output":out,"purpose":purpose,"behind_scenes":behind,"concept":concept,"b60":b60,"b70":b70,"why_diff":why})
    try:
        pf=evidence.get('preflight'); p2=evidence.get('phase2'); p3=evidence.get('phase3'); p4=evidence.get('phase4'); p5=evidence.get('phase5'); p6=evidence.get('phase6'); p7=evidence.get('phase7')
        missing=[name for name,obj in [('Preflight',pf),('Functional',p2),('Qualification',p3),('Parity',p4),('Benchmark',p5),('Optimization',p6),('Regression',p7)] if obj is None]
        add(1,"Check evidence completeness","PASS" if not missing else "WARN","collect phases 0–7", "All required evidence is present." if not missing else "Missing: "+', '.join(missing),"Show whether a production decision is even possible.","Missing evidence yields NOT READY rather than a Python exception.","Evidence chain","Require B60 evidence chain.","Require B70 evidence chain.","Production qualification should fail closed but remain user-friendly.")
        pf_ok=bool(pf and pf.get('ready')); p2_ok=bool(p2 and p2.get('steps') and p2['steps'][-1]['status']=='PASS'); p3_ok=bool(p3 and p3.get('steps') and p3['steps'][-1]['status'] in ['PASS','WARN'])
        p4_mode=(p4 or {}).get('summary',{}).get('mode'); p4_status=p4['steps'][-1]['status'] if p4 and p4.get('steps') else None; p4_ok=bool(p4_mode=='real_backend_parity' and p4_status in ['PASS','WARN'])
        add(2,"Validate functional + real parity gates","PASS" if pf_ok and p2_ok and p3_ok and p4_ok else "WARN", "preflight + functional + robustness + real backend parity",f"preflight={pf_ok}; functional={p2_ok}; robustness={p3_ok}; parity_mode={p4_mode}; real_parity={p4_ok}","Confirm correctness evidence exists before production promotion.","On CPU-only Streamlit Cloud, Phase 4 is intentionally SKIPPED; therefore accelerator production qualification stays NOT READY without throwing an error.","Scoped production evidence","Need real CPU-vs-B60 parity for B60.","Need real CPU-vs-B70 parity for B70.","A hosted demo can be healthy without being accelerator-qualified.")
        p5s=(p5 or {}).get('summary',{}); perf_ok=bool(p5s and p5s.get('e2e_ms') is not None and p5s.get('tokens_per_s') is not None and p5s.get('e2e_ms')<=max_e2e_ms and p5s.get('tokens_per_s')>=min_tok_s)
        add(3,"Check performance SLO","PASS" if perf_ok else "WARN","E2E <= SLO and tok/s >= minimum",f"E2E={p5s.get('e2e_ms','—')} ms; SLO={max_e2e_ms:.0f} ms; throughput={p5s.get('tokens_per_s','—')}; minimum={min_tok_s}","Translate benchmark numbers into acceptance criteria.","A benchmark result alone is descriptive; an SLO makes it actionable.","Performance acceptance","Set B60 SLO for B60 product needs.","Set B70 SLO for B70 product needs.","Different product tiers may use different targets.")
        p7_mode=(p7 or {}).get('summary',{}).get('mode'); p7_ok=bool(p7 and p7.get('steps') and p7['steps'][-1]['status']=='PASS' and p7_mode=='real_regression')
        add(4,"Check regression gate","PASS" if p7_ok else "WARN","require real Phase 7 PASS",f"regression_mode={p7_mode}; pass={p7_ok}","Ensure the candidate did not regress against an accepted baseline.","Missing/mismatched baseline is shown as prerequisite work, not as an app error.","Release guardrail","B60 needs a B60 baseline.","B70 needs a B70 baseline.","Release evidence is platform-specific.")
        complete=not missing; final=complete and pf_ok and p2_ok and p3_ok and p4_ok and perf_ok and p7_ok
        demo_ready=pf_ok and p2_ok and p3_ok and bool(p5s)
        result['summary']={"ready":final,"demo_ready":demo_ready,"missing":missing,"preflight":pf_ok,"functional":p2_ok,"qualification":p3_ok,"parity":p4_ok,"parity_mode":p4_mode,"performance_slo":perf_ok,"regression":p7_ok}
        if final: status='PASS'; result['output']='Production qualification=PASS for this tested model/backend/workload.'
        else: status='WARN'; result['output']='Production qualification=NOT READY. The app may still be suitable as a CPU-hosted learning/demo environment; review the missing/blocked gates above.'
        add(5,"Apply scoped production verdict",status,"aggregate required evidence gates",result['output'],"Give a go/no-go decision without confusing missing accelerator evidence with a software crash.","Production PASS is intentionally strict; hosted demo readiness is a separate concept.","Production readiness / go-no-go","PASS applies only to tested B60 config.","PASS applies only to tested B70 config.","Readiness is scoped to model + stack + hardware + workload."); return result
    except Exception as e:
        result['error']=f"{type(e).__name__}: {e}"; result['traceback']=traceback.format_exc(); add(len(result['steps'])+1,"Production aggregation could not complete","FAIL","exception handler",result['error'],"Expose aggregation bugs rather than silently claiming readiness.","This is an app/pipeline error, not a production verdict.","Gate reliability","Block B60 promotion.","Block B70 promotion.","A broken qualification gate is itself a risk."); return result

COMPARISON = [
    {"Step": "Preflight: software compatibility", "Generic / local step": "Check Python, torch, Transformers, disk/RAM and model-repo access.", "Intel Arc Pro B70 step": "Do the same plus require a working XPU build/runtime.", "Why different": "B70 depends on the Intel XPU path in addition to normal Python/model compatibility."},
    {"Step": "Preflight: device compute", "Generic / local step": "Run a tiny matrix multiply on the selected backend.", "Intel Arc Pro B70 step": "Run the same operation on xpu and synchronize it.", "Why different": "torch.xpu.is_available() proves discovery; a real tensor op proves the path can execute work."},
    {"Step": "Phase 2", "Generic / local step": "Single-run load + generate + non-empty output.", "Intel Arc Pro B70 step": "Same, but specifically on XPU/B70.", "Why different": "CPU functionality does not prove accelerator functionality."},
    {"Step": "Phase 3", "Generic / local step": "Repeatability, prompt variations, long-context sanity.", "Intel Arc Pro B70 step": "Same qualification tests on XPU/B70.", "Why different": "Stability and memory headroom can differ under repeated or larger workloads."},
    {"Step": "Phase 4", "Generic / local step": "CPU reference vs target-backend parity.", "Intel Arc Pro B70 step": "Compare CPU reference vs B70/XPU output/logits.", "Why different": "A backend can be functional yet still differ suspiciously from the reference."},
]

# -----------------------------
# UI
# -----------------------------
st.title("✅ Hugging Face Model Functional Verifier")
st.write(
    "A lightweight, TPM-friendly model qualification lab from setup compatibility through production gates."
)
st.info("🧭 **Recommended order:** Preflight → Functional → Qualification → Parity → Benchmark → Optimization → Regression → Production → Summary.  ✅ PASS = gate met · ⚠️ WARN = review evidence · ⏭️ SKIPPED = prerequisite/not applicable · ❌ FAIL = execution or gate failure.")

with st.sidebar:
    st.header("Input")
    model_id = st.text_input("Hugging Face model", value=DEFAULT_MODEL)
    device = st.selectbox("Device", ["Auto", "CPU", "CUDA", "MPS", "XPU"])
    dtype_choice = st.selectbox("Dtype", ["Auto", "FP32", "FP16", "BF16"])
    max_new_tokens = st.slider("Max new tokens", 8, 128, 32, 8)
    trust_remote_code = st.toggle("Trust remote code", value=False)
    st.caption("For Intel Arc Pro B60/B70 qualification, choose XPU explicitly when the app is running on a machine with that Intel GPU attached.")
    try:
        import torch
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            st.caption("Hosted Streamlit environments commonly expose CPU only; B60/B70 verification requires a B60/B70 host.")
    except Exception:
        pass

tabs = st.tabs([
    "0 · Preflight",
    "1 · Run Verification",
    "2 · Phase 3 Qualification",
    "3 · Phase 4 Parity",
    "4 · Phase 5 Benchmark",
    "5 · Phase 6 Optimization",
    "6 · Phase 7 Regression",
    "7 · Phase 8 Production",
    "8 · Generic vs B70",
    "9 · Environment",
    "10 · TPM Checklist",
    "11 · Summary",
])

with tabs[0]:
    st.subheader("🩺 Setup compatibility preflight")
    tab_note("🎯","Goal","Catch environment/dependency/device problems before loading the model.")
    st.write("Run this first. **HARD FAIL** blocks later phases; **WARN** means the setup may still work but deserves attention.")
    if st.button("🔎 Run preflight", type="primary", width="stretch"):
        with st.spinner("Checking setup compatibility..."):
            st.session_state["preflight"] = run_preflight(model_id.strip(), device, dtype_choice, trust_remote_code)
    pf = st.session_state.get("preflight")
    if pf:
        st.success("✅ PREFLIGHT READY — no hard compatibility blockers detected.") if pf["ready"] else st.error("❌ PREFLIGHT BLOCKED — fix HARD FAIL items before later phases.")
        st.dataframe(pf["checks"], width="stretch", hide_index=True)
        hard_fails=[c for c in pf["checks"] if c["Severity"]=="HARD" and c["Status"]=="FAIL"]
        warnings=[c for c in pf["checks"] if c["Status"]=="WARN"]
        a,b,c=st.columns(3);a.metric("Hard failures",len(hard_fails));b.metric("Warnings",len(warnings));c.metric("Selected device",pf["info"].get("device") or "—")
        if hard_fails:
            st.subheader("Fix first")
            for item in hard_fails: st.write(f"• **{item['Check']}** — {item['Detail']}")

    assumptions_box(["Preflight checks setup compatibility; it does not prove the model itself is functional.", "Community Cloud is resource-constrained and normally CPU-only; XPU checks require a B60/B70 host.", "Disk/RAM checks are conservative heuristics, not formal capacity guarantees."])

with tabs[1]:
    st.subheader("✅ Phase 2 — functional smoke test · TPM transparent mode")
    tab_note("🎯","Goal","Prove one end-to-end model inference works and show what happens internally.")
    pf=st.session_state.get("preflight"); ready=bool(pf and pf.get("ready"))
    if not pf: st.warning("Run **0 · Preflight** first.")
    elif not ready: st.error("Functional verification is blocked because preflight has a HARD FAIL.")
    prompt=st.text_area("Prompt",value="Reply with one short sentence explaining what a GPU does.",height=90,key="phase2_prompt")
    st.caption("Single-run functionality only. This does not yet prove stability, parity, or performance.")
    if st.button("▶ Run functional verification",width="stretch",disabled=not ready):
        with st.spinner("Loading and running model..."):
            st.session_state["phase2_result"]=run_verification(model_id.strip(),prompt.strip(),max_new_tokens,device,dtype_choice,trust_remote_code)
    result=st.session_state.get("phase2_result")
    if result:
        metrics={"Device":result.get("device") or "—","Dtype":result.get("dtype") or "—",
                 "Load time":f"{result['load_seconds']:.2f}s" if result.get("load_seconds") is not None else "—",
                 "Generate time":f"{result['generation_seconds']:.2f}s" if result.get("generation_seconds") is not None else "—"}
        render_step_expanders(result,"Final generated output",metrics)

    assumptions_box(["Functional PASS proves one deterministic inference path completed on this exact stack.", "It does not prove repeatability, parity, benchmark performance, or production readiness.", "Auto dtype may switch to FP16 on very small CPU hosts to reduce memory pressure; choose FP32 explicitly if you require it."])

with tabs[2]:
    st.subheader("🔁 Phase 3 — robustness / repeatability qualification")
    tab_note("🎯","Goal","Check that functionality is repeatable across several runs and prompt shapes.")
    pf=st.session_state.get("preflight");ready=bool(pf and pf.get("ready"))
    if not pf: st.warning("Run **0 · Preflight** first.")
    q_prompt=st.text_area("Base prompt for qualification",value="Explain in one short sentence what inference means in AI.",height=90,key="phase3_prompt")
    c1,c2=st.columns(2)
    with c1: repeats=st.slider("Repeat count",2,5,3,1)
    with c2: long_repeat_factor=st.slider("Long-context factor",20,120,60,10)
    if st.button("▶ Run Phase 3 qualification",width="stretch",disabled=not ready):
        with st.spinner("Running repeatability and stability checks..."):
            st.session_state["phase3_result"]=run_phase3_qualification(model_id.strip(),q_prompt.strip(),max_new_tokens,device,dtype_choice,trust_remote_code,repeats,long_repeat_factor)
    result=st.session_state.get("phase3_result")
    if result:
        sm=result.get("summary",{})
        render_step_expanders(result,"Phase 3 verdict",{"Device":result.get("device") or "—","Dtype":result.get("dtype") or "—","Pass rate":f"{sm.get('pass_rate',0):.2%}","Deterministic":str(sm.get('deterministic','—'))})

    assumptions_box(["Phase 3 is intentionally lightweight; it is not an exhaustive stress test.", "Shared-cloud scheduling can cause timing variation even when the model is healthy.", "Long-context factor is a small sanity test, not validation of the model's maximum advertised context window."])

with tabs[3]:
    st.subheader("⚖️ Phase 4 — correctness / parity against CPU reference")
    tab_note("🎯","Goal","Compare a real target backend against a CPU reference without wasting RAM on CPU-vs-CPU.")
    st.caption(
        "If the selected target resolves to CPU (common on Streamlit Community Cloud), "
        "the app now skips the duplicate CPU-vs-CPU model load to avoid RAM pressure. "
        "A full parity comparison runs only when the target is a distinct CUDA/XPU/MPS backend."
    )
    pf=st.session_state.get("preflight");ready=bool(pf and pf.get("ready"))
    p_prompt=st.text_area("Prompt for parity comparison",value="In one short sentence, explain what tokenization does in an LLM.",height=90,key="phase4_prompt")
    if st.button("▶ Run Phase 4 parity",width="stretch",disabled=not ready):
        with st.spinner("Running CPU-vs-target parity check..."):
            st.session_state["phase4_result"]=run_phase4_parity(model_id.strip(),p_prompt.strip(),max_new_tokens,device,dtype_choice,trust_remote_code)
    result=st.session_state.get("phase4_result")
    if result:
        sm=result.get("summary",{})
        render_step_expanders(result,"Phase 4 verdict",{"Target":result.get("device") or "—","Dtype":result.get("dtype") or "—","Top1 match":str(sm.get('top1_match','—')),"Top5 overlap":str(sm.get('top5_overlap','—'))})

    assumptions_box(["On CPU-only hosts this phase is SKIPPED because CPU-vs-CPU does not validate accelerator parity.", "Real B60/B70 parity requires the app to run on that XPU hardware and software stack.", "Small numerical differences across backends/precisions can be acceptable; bitwise equality is not always required."])

with tabs[4]:
    st.subheader("⏱️ Phase 5 — performance benchmark")
    tab_note("🎯","Goal","Create a lightweight, reproducible performance baseline for this exact workload.")
    pf=st.session_state.get("preflight");ready=bool(pf and pf.get("ready"))
    b_prompt=st.text_area("Benchmark prompt",value="Explain in two short sentences why KV cache matters during LLM inference.",height=90,key="phase5_prompt")
    iterations=st.slider("Measured iterations",2,5,3,1,key="phase5_iters")
    st.caption("Measures prefill latency, end-to-end latency, tokens/sec, approximate decode throughput, variability, and resource evidence.")
    if st.button("▶ Run Phase 5 benchmark",width="stretch",disabled=not ready):
        with st.spinner("Running benchmark..."):
            st.session_state["phase5_result"]=run_phase5_benchmark(model_id.strip(),b_prompt.strip(),max_new_tokens,device,dtype_choice,trust_remote_code,iterations)
    result=st.session_state.get("phase5_result")
    if result:
        sm=result.get("summary",{})
        render_step_expanders(result,"Phase 5 benchmark baseline",{
            "Prefill":fmt_metric(sm.get('prefill_ms')," ms",1),"TTFT proxy":fmt_metric(sm.get('ttft_proxy_ms')," ms",1),
            "TPOT~":fmt_metric(sm.get('approx_tpot_ms')," ms/tok",1),"Tokens/s":fmt_metric(sm.get('tokens_per_s'),"",2)})
        if sm and result.get("steps") and result["steps"][-1]["status"]=="PASS":
            st.session_state["regression_baseline"]=dict(sm)
            st.success("💾 This successful Phase 5 result was automatically saved as the Phase 7 regression baseline.")

    assumptions_box(["TTFT and TPOT shown here are educational proxies because this is not a streaming production server.", "On low-memory CPU hosts the separate prefill forward is replaced by a proxy to avoid RAM spikes.", "Community Cloud benchmark numbers are learning/demo evidence, not vendor-grade accelerator benchmark results."])

with tabs[5]:
    st.subheader("⚙️ Phase 6 — optimization / tuning")
    tab_note("🎯","Goal","Measure whether one controlled optimization actually helps without changing output.")
    pf=st.session_state.get("preflight");ready=bool(pf and pf.get("ready"))
    o_prompt=st.text_area("Optimization comparison prompt",value="Explain in one short sentence what a KV cache stores.",height=90,key="phase6_prompt")
    opt_iters=st.slider("Iterations per baseline/candidate",1,3,2,1,key="phase6_iters")
    st.caption("Compares a simple baseline against `use_cache=True + torch.inference_mode()` while holding workload constant.")
    if st.button("▶ Run Phase 6 optimization",width="stretch",disabled=not ready):
        with st.spinner("Running baseline and optimized configurations..."):
            st.session_state["phase6_result"]=run_phase6_optimization(model_id.strip(),o_prompt.strip(),max_new_tokens,device,dtype_choice,trust_remote_code,opt_iters)
    result=st.session_state.get("phase6_result")
    if result:
        sm=result.get("summary",{})
        render_step_expanders(result,"Phase 6 optimization verdict",{
            "Speedup":f"{sm.get('speedup',0):.3f}x","Latency gain":f"{sm.get('latency_improvement_pct',0):.1f}%",
            "Throughput gain":f"{sm.get('throughput_gain_pct',0):.1f}%","Output parity":str(sm.get('output_parity','—'))})

    assumptions_box(["A WARN means the optimization produced no clear gain; it is not necessarily an app error.", "Phase 6 now loads the model only once and compares both configurations in the same session to reduce memory pressure.", "Optimization gains must be revalidated on the actual B60/B70 target hardware."])

with tabs[6]:
    st.subheader("🛡️ Phase 7 — regression gate")
    tab_note("🎯","Goal","Compare the current candidate against the automatically saved known-good Phase 5 baseline.")
    pf=st.session_state.get("preflight");ready=bool(pf and pf.get("ready"))
    baseline=st.session_state.get("regression_baseline")
    if not baseline:
        phase5=st.session_state.get("phase5_result")
        if phase5 and phase5.get("summary"):
            st.info("A Phase 5 result exists. Save it as the regression baseline from the Phase 5 tab before running this phase.")
        else:
            st.warning("Run Phase 5 first and save its result as the regression baseline.")
    r_prompt=(baseline or {}).get("baseline_signature",{}).get("prompt", "Explain in two short sentences why KV cache matters during LLM inference.")
    st.text_area("Regression workload prompt (locked to baseline)",value=r_prompt,height=90,key="phase7_prompt_display",disabled=True)
    threshold=st.slider("Allowed regression threshold (%)",1.0,30.0,10.0,1.0)
    if st.button("▶ Run Phase 7 regression",width="stretch",disabled=not (ready and baseline)):
        with st.spinner("Comparing current candidate against baseline..."):
            st.session_state["phase7_result"]=run_phase7_regression(model_id.strip(),r_prompt.strip(),max_new_tokens,device,dtype_choice,baseline,trust_remote_code,threshold)
    result=st.session_state.get("phase7_result")
    if result:
        sm=result.get("summary",{})
        render_step_expanders(result,"Phase 7 regression verdict",{
            "Latency Δ":f"{sm.get('latency_delta_pct',0):.1f}%","Throughput Δ":f"{sm.get('throughput_delta_pct',0):.1f}%",
            "Threshold":f"{sm.get('threshold_pct',threshold):.1f}%","Output parity":str(sm.get('output_parity','—'))})

    assumptions_box(["Phase 5 automatically saves a successful regression baseline.", "Regression only makes sense when model, prompt, output length, device, dtype, and benchmark method match the baseline.", "SKIPPED means the comparison definition is invalid or a prerequisite is missing—not that the model regressed."])

with tabs[7]:
    st.subheader("🚦 Phase 8 — production qualification")
    tab_note("🎯","Goal","Aggregate all prior evidence into a scoped go/no-go decision—without mistaking missing prerequisites for app errors.")
    max_e2e=st.number_input("Maximum acceptable E2E latency (ms)",min_value=1.0,value=5000.0,step=100.0)
    min_tok=st.number_input("Minimum acceptable generated tokens/sec",min_value=0.1,value=1.0,step=0.5)
    evidence={"preflight":st.session_state.get("preflight"),"phase2":st.session_state.get("phase2_result"),
              "phase3":st.session_state.get("phase3_result"),"phase4":st.session_state.get("phase4_result"),
              "phase5":st.session_state.get("phase5_result"),"phase6":st.session_state.get("phase6_result"),
              "phase7":st.session_state.get("phase7_result")}
    st.caption("Aggregates the entire evidence chain into a scoped go/no-go decision for this model + stack + hardware + workload.")
    if st.button("▶ Run Phase 8 production qualification",width="stretch"):
        st.session_state["phase8_result"]=run_phase8_production_qualification(evidence,max_e2e,min_tok)
    result=st.session_state.get("phase8_result")
    if result:
        sm=result.get("summary",{})
        render_step_expanders(result,"Phase 8 production verdict",{
            "Ready":str(sm.get('ready','—')),"Performance SLO":str(sm.get('performance_slo','—')),
            "Regression":str(sm.get('regression','—')),"Reproducible":str(sm.get('reproducible','—'))})

    assumptions_box(["Production qualification is intentionally fail-closed: missing real accelerator parity keeps accelerator production status NOT READY.", "A CPU-hosted Streamlit deployment can still be healthy as a learning/demo environment.", "Real production readiness also needs serving, concurrency, security, observability, failover, capacity, and operational testing beyond this lightweight app."])

with tabs[8]:
    st.subheader("🧩 Generic vs Intel Arc Pro B70")
    tab_note("💡","How to read this","The model workflow is mostly the same; device/runtime, memory headroom, and evidence collection differ.")
    st.dataframe(COMPARISON,width="stretch",hide_index=True)

    assumptions_box(["This reference table explains conceptual differences; it does not execute B70 hardware remotely.", "Intel XPU behavior depends on the installed driver/runtime/PyTorch combination."])

with tabs[9]:
    st.subheader("🖥️ Current runtime")
    tab_note("💡","Why it matters","Framework, Python, and device versions are part of reproducibility.")
    snap=runtime_snapshot()
    st.dataframe([{"Item":k,"Value":v} for k,v in snap.items()],width="stretch",hide_index=True)

    assumptions_box(["Environment metadata is diagnostic evidence and should be captured alongside benchmark results.", "Hosted environments can change underlying CPU resources over time."])

with tabs[10]:
    st.subheader("🧭 Minimal TPM progression gate")
    tab_note("🧠","Remember","Each phase answers a different question; do not jump to optimization before proving the earlier gates.")
    checks=[
        "Phase 0 — Preflight: setup compatibility has no HARD FAIL.",
        "Phase 2 — Functional: tokenizer + model + generation + non-empty output succeed.",
        "Phase 3 — Qualification: repeated runs and prompt variations remain stable enough.",
        "Phase 4 — Parity: target behavior is credible against a CPU reference.",
        "Phase 5 — Benchmark: reproducible latency/throughput/resource baseline exists.",
        "Phase 6 — Optimization: tuning has measured benefit and acceptable functional parity.",
        "Phase 7 — Regression: candidate stays within known-good thresholds.",
        "Phase 8 — Production: all required gates, SLOs and evidence are satisfied.",
    ]
    for i,item in enumerate(checks,1): st.write(f"**{i}.** {item}")

    assumptions_box(["The checklist is a learning sequence; real organizations may add security, compliance, load, resilience, and serving-specific gates.", "Each phase should have explicit entry/exit criteria before automation in CI/CD."])

with tabs[11]:
    st.subheader("📊 Executive Summary · TPM implementation map")
    tab_note("🎯","Goal","Consolidate phase status, key metrics, TPM takeaways, and interview-ready concepts in one view.")
    pf=st.session_state.get("preflight");p2=st.session_state.get("phase2_result");p3=st.session_state.get("phase3_result")
    p4=st.session_state.get("phase4_result");p5=st.session_state.get("phase5_result");p6=st.session_state.get("phase6_result")
    p7=st.session_state.get("phase7_result");p8=st.session_state.get("phase8_result")

    def stat(result, allow_warn=False):
        if result is None:
            return "NOT RUN"
        if isinstance(result,dict) and "ready" in result:
            return "PASS" if result.get("ready") else "FAIL"
        if isinstance(result,dict):
            mode = result.get("summary",{}).get("mode")
            if mode in {"educational_skip","prerequisite_missing","baseline_mismatch"}:
                return "SKIPPED"
        steps=result.get("steps",[]) if isinstance(result,dict) else []
        if not steps:
            return "NOT RUN"
        s=steps[-1].get("status","UNKNOWN")
        if allow_warn and s=="WARN":
            return "WARN"
        return s

    p5s=(p5 or {}).get("summary",{});p6s=(p6 or {}).get("summary",{});p7s=(p7 or {}).get("summary",{});p8s=(p8 or {}).get("summary",{})
    rows=[
        {"Phase":"0 · Preflight","Implementation gist":"Prove environment/model/device compatibility before expensive work.","Status":"PASS" if pf and pf.get('ready') else "FAIL" if pf else "NOT RUN","Key metrics":"Hard fails; warnings; device visibility; RAM/disk headroom","TPM takeaway":"Environment qualification is a gate, not debugging after the fact.","Top 1% interview question":"Why separate preflight from model validation?","Ideal expected answer":"Preflight isolates setup/runtime compatibility so model failures are not confused with environment failures."},
        {"Phase":"2 · Functional","Implementation gist":"Load tokenizer/model → place tensors → deterministic generation → non-empty output.","Status":stat(p2),"Key metrics":"Load time; generation time; device; dtype; generated tokens","TPM takeaway":"A single PASS only proves basic functionality on this exact stack/backend.","Top 1% interview question":"What does functional verification prove—and not prove?","Ideal expected answer":"It proves the inference path works once; it does not prove stability, parity, performance, or production readiness."},
        {"Phase":"3 · Qualification","Implementation gist":"Repeat runs + prompt variations + long-context sanity.","Status":stat(p3,True),"Key metrics":"Pass rate; deterministic repeatability; average repeat time; memory evidence","TPM takeaway":"One successful run is insufficient; reliability requires repeated and varied evidence.","Top 1% interview question":"Why test repeatability before benchmarking?","Ideal expected answer":"Because unstable execution makes performance numbers untrustworthy and can hide runtime or memory issues."},
        {"Phase":"4 · Parity","Implementation gist":"CPU reference vs DISTINCT target backend using same prompt/model/settings; CPU-only hosted runs are explicitly SKIPPED.","Status":stat(p4,True),"Key metrics":"Parity mode; text match; top-1 token match; top-5 overlap; max/mean logit difference","TPM takeaway":"Functional does not automatically mean numerically or behaviorally credible.","Top 1% interview question":"Do you require bitwise-identical CPU and GPU outputs?","Ideal expected answer":"Not always; assess deterministic text/token agreement and bounded numeric drift appropriate to precision/backend."},
        {"Phase":"5 · Benchmark","Implementation gist":"Warm up → measure prefill → end-to-end → throughput → variability → memory.","Status":stat(p5),"Key metrics":f"Prefill {p5s.get('prefill_ms','—')} ms; TTFT proxy {p5s.get('ttft_proxy_ms','—')} ms; TPOT~ {p5s.get('approx_tpot_ms','—')} ms/tok; E2E {p5s.get('e2e_ms','—')} ms; {p5s.get('tokens_per_s','—')} tok/s","TPM takeaway":"Benchmark only fixed workloads; latency, throughput, variability and capacity all matter.","Top 1% interview question":"Why distinguish prefill from decode in LLM inference?","Ideal expected answer":"Prefill processes prompt tokens largely in parallel; decode is sequential token generation with different bottlenecks and KPIs."},
        {"Phase":"6 · Optimization","Implementation gist":"Measure baseline → apply one controlled tuning change → remeasure → parity check.","Status":stat(p6,True),"Key metrics":f"Speedup {p6s.get('speedup','—')}x; latency gain {p6s.get('latency_improvement_pct','—')}%; throughput gain {p6s.get('throughput_gain_pct','—')}%","TPM takeaway":"Optimization is a measured before/after delta, not a list of knobs.","Top 1% interview question":"How do you prove an optimization really helped?","Ideal expected answer":"Hold workload constant, change one factor, measure repeatably, verify performance gain and preserve functional correctness."},
        {"Phase":"7 · Regression","Implementation gist":"Compare candidate against known-good baseline using thresholds and correctness guard.","Status":stat(p7),"Key metrics":f"Latency Δ {p7s.get('latency_delta_pct','—')}%; throughput Δ {p7s.get('throughput_delta_pct','—')}%; threshold {p7s.get('threshold_pct','—')}%","TPM takeaway":"A baseline turns performance knowledge into an automated release guardrail.","Top 1% interview question":"What makes a regression test valid?","Ideal expected answer":"Same workload/environment, trusted baseline, explicit thresholds, repeatable measurements, and correctness checks."},
        {"Phase":"8 · Production","Implementation gist":"Aggregate upstream gates + SLO + regression + resource/reproducibility evidence into go/no-go.","Status":stat(p8),"Key metrics":"Gate completion; latency SLO; min throughput; regression gate; capacity headroom","TPM takeaway":"Production readiness is scoped evidence across quality, performance, reliability and operability—not a benchmark score.","Top 1% interview question":"What would make you block production even if benchmark numbers look good?","Ideal expected answer":"Failed functional/parity/regression gates, insufficient headroom, missing reproducibility/observability, or unmet SLOs."},
        {"Phase":"Reference · Generic vs B70","Implementation gist":"Map each generic verification step to the corresponding Intel XPU/B70 action.","Status":"REFERENCE","Key metrics":"Device backend; memory capacity; runtime identity; dtype support","TPM takeaway":"The model workflow stays mostly the same; backend, memory headroom and runtime evidence change.","Top 1% interview question":"What fundamentally changes when porting a Hugging Face model from CPU/CUDA to Intel Arc Pro?","Ideal expected answer":"Keep model/tokenizer semantics constant, switch to the supported XPU/runtime path, validate dtype/operator support, device placement and accelerator-specific evidence."},
        {"Phase":"Environment","Implementation gist":"Capture OS, Python, framework versions, device visibility and accelerator identity.","Status":"PASS" if pf else "NOT RUN","Key metrics":"Python/torch/transformers versions; XPU/CUDA/MPS visibility; device name","TPM takeaway":"Environment metadata is part of reproducibility, not administrative trivia.","Top 1% interview question":"Why must benchmark reports include software-stack versions?","Ideal expected answer":"Driver/framework/runtime changes can materially alter correctness and performance, so results without version context are not reproducible."},
        {"Phase":"TPM Checklist","Implementation gist":"Keep phase gates in the correct order from compatibility to production readiness.","Status":"REFERENCE","Key metrics":"Gate completion and exit criteria per phase","TPM takeaway":"Do not optimize or benchmark before proving the prior gate; each phase answers a different customer question.","Top 1% interview question":"What is the correct order for qualifying a new model on a new accelerator?","Ideal expected answer":"Preflight → functional → robustness → parity → benchmark → optimize → regression → production qualification."},
    ]
    st.dataframe(rows,width="stretch",hide_index=True)

    st.markdown("### Readiness dashboard")
    readiness_rows=[r for r in rows if r["Phase"].startswith(("0 ·","2 ·","3 ·","4 ·","5 ·","6 ·","7 ·","8 ·"))]
    phase_names=[r["Phase"] for r in readiness_rows]
    score_map={"PASS":100,"WARN":60,"SKIPPED":25,"FAIL":0,"NOT RUN":0}
    scores=[score_map.get(r["Status"],0) for r in readiness_rows]
    score_df=pd.DataFrame({"Phase":phase_names,"Readiness":scores}).set_index("Phase")
    st.bar_chart(score_df)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Benchmark E2E",f"{p5s.get('e2e_ms',0):.1f} ms" if p5s else "—")
    c2.metric("Benchmark throughput",f"{p5s.get('tokens_per_s',0):.2f} tok/s" if p5s else "—")
    c3.metric("Optimization speedup",f"{p6s.get('speedup',0):.2f}x" if p6s else "—")
    c4.metric("Production ready",str(p8s.get('ready','—')) if p8s else "—")

    st.markdown("### Performance / change dashboard")
    perf_rows=[]
    if p5s:
        perf_rows.extend([{"Metric":"Prefill ms","Value":float(p5s.get('prefill_ms',0))},{"Metric":"E2E ms","Value":float(p5s.get('e2e_ms',0))},{"Metric":"Tokens/sec","Value":float(p5s.get('tokens_per_s',0))}])
    if p6s:
        perf_rows.extend([{"Metric":"Optimization latency gain %","Value":float(p6s.get('latency_improvement_pct',0))},{"Metric":"Optimization throughput gain %","Value":float(p6s.get('throughput_gain_pct',0))}])
    if p7s:
        perf_rows.extend([{"Metric":"Regression latency delta %","Value":float(p7s.get('latency_delta_pct',0))},{"Metric":"Regression throughput delta %","Value":float(p7s.get('throughput_delta_pct',0))}])
    if perf_rows:
        st.bar_chart(pd.DataFrame(perf_rows).set_index("Metric"))
    else:
        st.info("Run Phase 5–7 to populate performance/change visuals.")

    st.markdown("### Organic implementation sequence to remember")
    st.code("""Preflight compatibility
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
Only then: scale-out / serving / long-term operations""",language="text")

    assumptions_box(["Summary values are only as trustworthy as the individual phase evidence that produced them.", "SKIPPED parity on CPU-only Streamlit Cloud must not be interpreted as accelerator qualification.", "Dashboard charts combine metrics with different units for learning convenience; use dedicated charts/reports for formal benchmarking."])

st.divider()
st.caption("🧪 Lightweight qualification lab. Results are scoped to the tested model, software stack, hardware, precision and workload; they are not universal vendor performance claims.")

