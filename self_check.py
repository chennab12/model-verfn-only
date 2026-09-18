"""Offline dependency/source smoke check. Does not download a model."""
from pathlib import Path
import ast
import importlib
import sys

REQUIRED = {
    "streamlit": (1, 57),
    "transformers": (5, 10),
    "torch": (2, 10),
    "accelerate": (1, 10),
    "safetensors": (0, 6),
    "huggingface_hub": (1, 0),
    "psutil": (6, 0),
    "pandas": (2, 2),
}
EXPECTED_FUNCS = {
    "run_preflight",
    "run_verification",
    "run_phase3_qualification",
    "run_phase4_parity",
    "run_phase5_benchmark",
    "run_phase6_optimization",
    "run_phase7_regression",
    "run_phase8_production_qualification",
}

def vt(v):
    out=[]
    for p in str(v).split('.'):
        n=''.join(c for c in p if c.isdigit())
        if not n: break
        out.append(int(n))
    return tuple(out)

failures=[]
print('Python:', sys.version.split()[0])
for name, minimum in REQUIRED.items():
    try:
        mod=importlib.import_module(name)
        version=getattr(mod,'__version__','0')
        ok=vt(version) >= minimum
        print(f'{name:16} {version:12} ' + ('PASS' if ok else f'FAIL (< {minimum})'))
        if not ok: failures.append(name)
    except Exception as e:
        print(f'{name:16} IMPORT FAIL: {e}')
        failures.append(name)

app=Path(__file__).with_name('app.py')
source=app.read_text(encoding='utf-8')
tree=ast.parse(source)
funcs={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
missing=sorted(EXPECTED_FUNCS-funcs)
print('app.py syntax: PASS')
print('phase functions:', 'PASS' if not missing else f'FAIL missing={missing}')
if 'use_container_width=' in source:
    failures.append('deprecated_streamlit_width')
    print('Streamlit deprecated width API: FAIL')
else:
    print('Streamlit deprecated width API: PASS')

if failures:
    raise SystemExit('SELF CHECK FAILED: ' + ', '.join(failures))
print('SELF CHECK PASSED')
