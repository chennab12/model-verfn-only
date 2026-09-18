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

def render_step_expanders(result, output_label="Final generated output", metrics=None):
    if result is None:
        return
    final_status = "PASS"
    if result.get("steps"):
        final_status = result["steps"][-1]["status"]
    st.success("✅ PASS") if final_status == "PASS" else st.error("❌ FAIL")
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
            st.dataframe(compare_rows, use_container_width=True, hide_index=True)

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
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
        )
        model.eval()
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
            out = model.generate(
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
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
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
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
        )
        model.eval()
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
                out = model.generate(
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
                out = model.generate(
                    **inputs, max_new_tokens=min(max_new_tokens, 16), do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            suite_results.append((name, tok_len, bool(decoded), decoded[:120]))
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
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
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
        from transformers import AutoTokenizer, AutoModelForCausalLM

        add_step(1, "Import PyTorch + Transformers", "PASS",
                 "import torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM",
                 f"torch={torch.__version__}; transformers={transformers.__version__}",
                 "Initialize the libraries needed for reference-vs-target comparison.",
                 "Phase 4 compares two executions of the same model configuration on different backends.",
                 "Framework/runtime comparison setup",
                 "Same logic on B60.", "Same logic on B70.",
                 "Hardware changes the target backend, not the comparison methodology.")

        target_device = resolve_device(prefer_device)
        result["device"] = target_device
        target_dtype = selected_dtype(target_device, dtype_choice)
        result["dtype"] = str(target_dtype).replace("torch.", "")
        add_step(2, "Define reference and target paths", "PASS",
                 "reference_device='cpu'; target_device=selected_backend",
                 f"reference=cpu/fp32; target={target_device}/{result['dtype']}",
                 "Create a CPU reference path and a target-backend path for parity comparison.",
                 "CPU is used as a practical baseline because it is often the most universal functional reference.",
                 "Reference-vs-target methodology",
                 "B60 becomes the target XPU path when selected.", "B70 becomes the target XPU path when selected.",
                 "The reference path stays CPU; only the target hardware changes.")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        add_step(3, "Load shared tokenizer", "PASS",
                 f"AutoTokenizer.from_pretrained('{model_id}')",
                 f"{tokenizer.__class__.__name__} loaded",
                 "Guarantee that both CPU and target runs use identical tokenization and prompt formatting.",
                 "A shared tokenizer prevents false parity differences caused by preprocessing mismatches.",
                 "Shared preprocessing baseline",
                 "Same tokenizer for CPU and B60.", "Same tokenizer for CPU and B70.",
                 "Parity requires preprocessing to stay constant.")

        # Prepare identical text and cpu inputs
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt
        cpu_inputs = tokenizer(formatted, return_tensors="pt")
        input_len = int(cpu_inputs["input_ids"].shape[-1])
        add_step(4, "Prepare identical reference prompt", "PASS",
                 "formatted = tokenizer.apply_chat_template(...)\ncpu_inputs = tokenizer(formatted, return_tensors='pt')",
                 f"formatted_length={len(formatted)} chars; input_tokens={input_len}",
                 "Ensure both runs see exactly the same prompt content and token IDs.",
                 "Parity comparisons are only meaningful when the model input is identical across paths.",
                 "Controlled experimental input",
                 "Same prompt and token IDs for B60 comparison.", "Same prompt and token IDs for B70 comparison.",
                 "Any input mismatch would invalidate the parity check.")

        # CPU reference path
        cpu_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
        )
        cpu_model.eval().to("cpu")
        with torch.no_grad():
            cpu_logits = cpu_model(**cpu_inputs).logits[0, -1, :].float().cpu()
            cpu_gen = cpu_model.generate(
                **cpu_inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        cpu_new = cpu_gen[0][input_len:]
        cpu_text = tokenizer.decode(cpu_new, skip_special_tokens=True).strip()
        cpu_top5 = torch.topk(cpu_logits, k=5)
        cpu_top5_ids = cpu_top5.indices.tolist()
        result["samples"].append(f"CPU output: {cpu_text}")
        add_step(5, "Run CPU reference inference", "PASS",
                 "cpu_model = AutoModelForCausalLM.from_pretrained(...).to('cpu')\nlogits = cpu_model(**cpu_inputs).logits\ncpu_model.generate(...)",
                 f"cpu_output_len={len(cpu_new)}; cpu_top5_next_token_ids={cpu_top5_ids}; cpu_text={cpu_text[:120]}",
                 "Create a baseline output and baseline next-token distribution.",
                 "The CPU run gives a practical reference for both decoded text and the logits used to choose next tokens.",
                 "Reference inference / logits baseline",
                 "B60 will be compared against this CPU reference.", "B70 will be compared against this CPU reference.",
                 "CPU is a common baseline because it is widely available and often easier to trust/debug.")

        # Target path
        target_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=target_dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True
        )
        target_model.eval().to(target_device)
        tgt_inputs = {k: v.to(target_device) for k, v in cpu_inputs.items()}
        with torch.no_grad():
            tgt_logits = target_model(**tgt_inputs).logits[0, -1, :].float().cpu()
            tgt_gen = target_model.generate(
                **tgt_inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        if target_device == "cuda":
            torch.cuda.synchronize()
        elif target_device == "xpu" and hasattr(torch.xpu, "synchronize"):
            torch.xpu.synchronize()
        tgt_new = tgt_gen[0][input_len:]
        tgt_text = tokenizer.decode(tgt_new, skip_special_tokens=True).strip()
        tgt_top5 = torch.topk(tgt_logits, k=5)
        tgt_top5_ids = tgt_top5.indices.tolist()
        result["samples"].append(f"Target output ({target_device}): {tgt_text}")
        add_step(6, "Run target-backend inference", "PASS",
                 "target_model = AutoModelForCausalLM.from_pretrained(...).to(target_device)\ntarget_logits = target_model(**target_inputs).logits\ntarget_model.generate(...)",
                 f"target_output_len={len(tgt_new)}; target_top5_next_token_ids={tgt_top5_ids}; target_text={tgt_text[:120]}",
                 "Run the same model/input on the chosen target backend so its behavior can be compared against CPU.",
                 "This isolates backend-dependent numeric/runtime differences while keeping the model/prompt constant.",
                 "Target inference / backend-specific execution",
                 "If target is B60, this is the B60/XPU behavior being compared to CPU.",
                 "If target is B70, this is the B70/XPU behavior being compared to CPU.",
                 "Same comparison structure; only the target hardware changes.")

        same_text_exact = cpu_text == tgt_text
        same_text_norm = normalize_text(cpu_text) == normalize_text(tgt_text)
        top1_match = int(cpu_top5_ids[0] == tgt_top5_ids[0])
        top5_overlap = len(set(cpu_top5_ids).intersection(set(tgt_top5_ids)))
        max_abs_diff = float((cpu_logits - tgt_logits).abs().max().item())
        mean_abs_diff = float((cpu_logits - tgt_logits).abs().mean().item())

        result["summary"] = {
            "same_text_exact": same_text_exact,
            "same_text_normalized": same_text_norm,
            "top1_match": top1_match,
            "top5_overlap": top5_overlap,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
        }

        add_step(7, "Compare CPU vs target evidence", "PASS",
                 "compare(decoded_text, top1, top5, logits_diff)",
                 f"same_text_exact={same_text_exact}; normalized_match={same_text_norm}; "
                 f"top1_match={top1_match}; top5_overlap={top5_overlap}; "
                 f"max_abs_diff={max_abs_diff:.6f}; mean_abs_diff={mean_abs_diff:.6f}",
                 "Convert both runs into comparable parity metrics.",
                 "Phase 4 examines both human-readable output and lower-level next-token score similarity.",
                 "Correctness / parity metrics",
                 "B60 may show larger numeric drift than CPU yet still remain functionally acceptable.",
                 "B70 may also differ numerically from CPU; exact equality is not always required for useful parity.",
                 "Parity checks look for credible similarity rather than demanding bitwise identity across all backends.")

        verdict_pass = bool(tgt_text) and (same_text_norm or top1_match == 1 or top5_overlap >= 3)
        result["output"] = (
            f"Parity verdict={'PASS' if verdict_pass else 'WARN/FAIL'} | "
            f"normalized_text_match={same_text_norm} | top1_match={top1_match} | "
            f"top5_overlap={top5_overlap} | max_abs_diff={max_abs_diff:.6f}"
        )
        add_step(8, "Apply Phase 4 parity verdict", "PASS" if verdict_pass else "WARN",
                 "PASS if target output is credible vs CPU using text and next-token evidence",
                 result["output"],
                 "Provide a practical pre-benchmark confidence gate that target backend behavior is believable.",
                 "Phase 4 does not require perfect bitwise identity. It asks whether the target backend behaves credibly enough to trust subsequent benchmarking.",
                 "Reference parity / correctness gate",
                 "B60 PASS means B60 behavior is acceptably close to CPU for this test.",
                 "B70 PASS means B70 behavior is acceptably close to CPU for this test.",
                 "A backend may be functional but still suspicious if parity metrics diverge too far from CPU.")
        return result

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["steps"].append({
            "number": len(result["steps"]) + 1,
            "name": "Failure captured",
            "status": "FAIL",
            "command": "exception handler",
            "output": result["error"],
            "purpose": "Capture the exact point and reason the parity flow stopped.",
            "behind_scenes": "The traceback helps classify whether the failure came from CPU reference, target backend, or comparison logic.",
            "concept": "Failure isolation / parity debugging",
            "b60": "Determine whether B60 failed while CPU succeeded, which is strong evidence of backend-specific issues.",
            "b70": "Determine whether B70 failed while CPU succeeded, which is strong evidence of backend-specific issues.",
            "why_diff": "Parity work is specifically designed to expose backend-specific correctness concerns before benchmarking."
        })
        return result

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
    "A lightweight functional checker with a **preflight gate**, a transparent **single-run functional tab**, "
    "and now two new tabs for **Phase 3 robustness qualification** and **Phase 4 CPU-vs-target parity**."
)

with st.sidebar:
    st.header("Input")
    model_id = st.text_input("Hugging Face model", value=DEFAULT_MODEL)
    device = st.selectbox("Device", ["Auto", "CPU", "CUDA", "MPS", "XPU"])
    dtype_choice = st.selectbox("Dtype", ["Auto", "FP32", "FP16", "BF16"])
    max_new_tokens = st.slider("Max new tokens", 8, 128, 32, 8)
    trust_remote_code = st.toggle("Trust remote code", value=False)
    st.caption("For Intel Arc Pro B60/B70 qualification, choose XPU explicitly when you want to qualify that hardware.")

tabs = st.tabs([
    "0 · Preflight",
    "1 · Run Verification",
    "2 · Phase 3 Qualification",
    "3 · Phase 4 Parity",
    "4 · Generic vs B70",
    "5 · Environment",
    "6 · TPM Checklist",
])

with tabs[0]:
    st.subheader("Setup compatibility preflight")
    st.write("Run this first. **HARD FAIL** blocks later phases; **WARN** means the setup may still work but deserves attention.")
    if st.button("🔎 Run preflight", type="primary", use_container_width=True):
        with st.spinner("Checking setup compatibility..."):
            st.session_state["preflight"] = run_preflight(model_id.strip(), device, dtype_choice, trust_remote_code)

    pf = st.session_state.get("preflight")
    if pf:
        st.success("✅ PREFLIGHT READY — no hard compatibility blockers detected.") if pf["ready"] else st.error("❌ PREFLIGHT BLOCKED — fix HARD FAIL items before later phases.")
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
    st.subheader("Functional smoke test — TPM transparent mode")
    pf = st.session_state.get("preflight")
    ready = bool(pf and pf.get("ready"))
    if not pf:
        st.warning("Run **0 · Preflight** first.")
    elif not ready:
        st.error("Functional verification is blocked because preflight has a HARD FAIL.")

    prompt = st.text_area("Prompt", value="Reply with one short sentence explaining what a GPU does.", height=90, key="phase2_prompt")
    st.caption("This tab intentionally exposes the internal AI/ML flow. It is functional verification, not a performance benchmark.")
    if st.button("▶ Run functional verification", use_container_width=True, disabled=not ready):
        with st.spinner("Loading and running model..."):
            st.session_state["phase2_result"] = run_verification(model_id.strip(), prompt.strip(), max_new_tokens, device, dtype_choice, trust_remote_code)
    result = st.session_state.get("phase2_result")
    if result:
        metrics = {
            "Device": result.get("device") or "—",
            "Dtype": result.get("dtype") or "—",
            "Load time": f"{result['load_seconds']:.2f}s" if result.get("load_seconds") is not None else "—",
            "Generate time": f"{result['generation_seconds']:.2f}s" if result.get("generation_seconds") is not None else "—",
        }
        render_step_expanders(result, output_label="Final generated output", metrics=metrics)

with tabs[2]:
    st.subheader("Phase 3 — robustness / repeatability qualification")
    pf = st.session_state.get("preflight")
    ready = bool(pf and pf.get("ready"))
    if not pf:
        st.warning("Run **0 · Preflight** first.")
    elif not ready:
        st.error("Phase 3 is blocked because preflight has a HARD FAIL.")
    q_prompt = st.text_area("Base prompt for qualification", value="Explain in one short sentence what inference means in AI.", height=90, key="phase3_prompt")
    col1, col2 = st.columns(2)
    with col1:
        repeats = st.slider("Repeat count", 2, 5, 3, 1)
    with col2:
        long_repeat_factor = st.slider("Long-context factor", 20, 120, 60, 10)
    st.caption("Phase 3 asks: can the same model run repeatedly and credibly across prompt variations, not just once?")
    if st.button("▶ Run Phase 3 qualification", use_container_width=True, disabled=not ready):
        with st.spinner("Running repeatability and stability checks..."):
            st.session_state["phase3_result"] = run_phase3_qualification(
                model_id.strip(), q_prompt.strip(), max_new_tokens, device, dtype_choice, trust_remote_code,
                repeats=repeats, long_repeat_factor=long_repeat_factor
            )
    result = st.session_state.get("phase3_result")
    if result:
        summary = result.get("summary", {})
        metrics = {
            "Device": result.get("device") or "—",
            "Dtype": result.get("dtype") or "—",
            "Pass rate": f"{summary.get('pass_rate', 0.0):.2%}",
            "Deterministic": str(summary.get("deterministic", "—")),
        }
        render_step_expanders(result, output_label="Phase 3 verdict", metrics=metrics)

with tabs[3]:
    st.subheader("Phase 4 — correctness / parity against CPU reference")
    pf = st.session_state.get("preflight")
    ready = bool(pf and pf.get("ready"))
    if not pf:
        st.warning("Run **0 · Preflight** first.")
    elif not ready:
        st.error("Phase 4 is blocked because preflight has a HARD FAIL.")
    p_prompt = st.text_area("Prompt for parity comparison", value="In one short sentence, explain what tokenization does in an LLM.", height=90, key="phase4_prompt")
    st.caption("Phase 4 asks: does the selected backend behave credibly compared with a CPU reference on the same prompt/model settings?")
    if st.button("▶ Run Phase 4 parity", use_container_width=True, disabled=not ready):
        with st.spinner("Running CPU-vs-target parity check..."):
            st.session_state["phase4_result"] = run_phase4_parity(
                model_id.strip(), p_prompt.strip(), max_new_tokens, device, dtype_choice, trust_remote_code
            )
    result = st.session_state.get("phase4_result")
    if result:
        summary = result.get("summary", {})
        metrics = {
            "Target": result.get("device") or "—",
            "Dtype": result.get("dtype") or "—",
            "Top1 match": str(summary.get("top1_match", "—")),
            "Top5 overlap": str(summary.get("top5_overlap", "—")),
        }
        render_step_expanders(result, output_label="Phase 4 verdict", metrics=metrics)

with tabs[4]:
    st.subheader("Generic vs Intel Arc Pro B70")
    st.dataframe(COMPARISON, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Current runtime")
    snap = runtime_snapshot()
    st.dataframe([{"Item": k, "Value": v} for k, v in snap.items()], use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Minimal TPM progression gate")
    checks = [
        "Phase 0 — Preflight: setup compatibility has no HARD FAIL.",
        "Phase 2 — Single-run verification: tokenizer + model + generation + non-empty output succeed.",
        "Phase 3 — Qualification: repeated runs and prompt variations remain stable enough to trust the environment.",
        "Phase 4 — Parity: target backend behavior is credible against a CPU reference.",
        "Only after Phases 0/2/3/4 should benchmarking become trustworthy.",
        "Then come benchmarking, optimization, regression automation, and production qualification.",
    ]
    for i, item in enumerate(checks, 1):
        st.write(f"**{i}.** {item}")

st.divider()
st.caption(
    "This app is still focused on functional qualification. The new Phase 3 and Phase 4 tabs "
    "bridge the gap between a single smoke test and later benchmarking."
)
