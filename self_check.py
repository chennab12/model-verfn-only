from pathlib import Path
import ast
source=Path("app.py").read_text(encoding="utf-8")
ast.parse(source)
required=["def quick_benchmark(","def benchmark_loaded_model(","def run_phase5_benchmark(","def run_phase6_optimization(","def run_phase7_regression(","def run_phase8_production_qualification(","Assumptions & caveats","quick_benchmark_v9"]
missing=[x for x in required if x not in source]
if missing: raise SystemExit("Missing: "+", ".join(missing))
print("app.py syntax/structure: PASS")
