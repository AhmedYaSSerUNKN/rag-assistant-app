"""Self-check against the assignment's deliverables checklist. Run before submitting:
    python scripts/verify_submission.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, hint: str = "") -> None:
    checks.append((name, bool(ok), hint))


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


# --- deliverables ---
check("notebooks/rag_pipeline.ipynb exists", exists("notebooks/rag_pipeline.ipynb"))
check("backend/app/main.py exists", exists("backend/app/main.py"))
check("backend/.env.example exists", exists("backend/.env.example"))
check("backend/requirements.txt pinned",
      exists("backend/requirements.txt")
      and all("==" in l for l in (ROOT / "backend/requirements.txt").read_text().split() if l.strip()),
      "every dependency must be pinned with ==")
check("frontend/app.py exists", exists("frontend/app.py"))
check("frontend/.env.example exists", exists("frontend/.env.example"))
check("README.md exists", exists("README.md"))
check(".gitignore exists", exists(".gitignore"))

# --- vector store exported ---
store = ROOT / "backend/data/vector_store"
check("vector store exported to backend/", any(store.glob("*.sqlite3")),
      "run notebook section 2.7")
check("store_config.json present", (store / "store_config.json").exists(),
      "run notebook section 2.7")

# --- common point-losers ---
frontend_src = " ".join(
    (ROOT / "frontend" / f).read_text() for f in ("app.py", "api_client.py") if (ROOT / "frontend" / f).exists()
)
check("no hard-coded localhost:8000 in frontend", "http://localhost:8000" not in frontend_src,
      "read the URL from API_BASE_URL instead")

gitignore = (ROOT / ".gitignore").read_text() if exists(".gitignore") else ""
check(".gitignore excludes .env", "\n.env" in gitignore)
check(".gitignore excludes .venv/", ".venv/" in gitignore)
check(".gitignore excludes raw corpus", "data/raw/*" in gitignore)

# --- evaluation ---
eval_file = ROOT / "config/eval_questions.json"
n_questions = len(json.loads(eval_file.read_text())) if eval_file.exists() else 0
check("at least 10 evaluation questions", n_questions >= 10, f"found {n_questions}")
check("evaluation results generated", exists("reports/evaluation_results.md"),
      "run notebook section 2.6")

# --- tests ---
try:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT / "backend",
                            capture_output=True, text=True, timeout=300)
    check("pytest passes", result.returncode == 0, result.stdout.strip().splitlines()[-1] if result.stdout else "")
except Exception as exc:  # noqa: BLE001
    check("pytest passes", False, str(exc))

# --- git hygiene ---
try:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    check("no .env committed", not any(f.endswith(".env") for f in tracked))
    check("no .venv committed", not any(f.startswith(".venv/") for f in tracked))
except Exception:  # noqa: BLE001
    pass

# --- report ---
width = max(len(n) for n, _, _ in checks)
failed = 0
for name, ok, hint in checks:
    mark = "PASS" if ok else "FAIL"
    line = f"[{mark}] {name.ljust(width)}"
    if not ok:
        failed += 1
        line += f"   -> {hint}" if hint else ""
    print(line)

print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
if failed:
    print("Note: vector-store and evaluation checks only pass after the notebook has been run end to end.")
sys.exit(1 if failed else 0)
