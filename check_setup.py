"""
check_setup.py — run this before anything else.

    python check_setup.py

Every line should show a tick. Fix the crosses before moving on.
"""

import importlib
import os
import platform
import shutil
import subprocess
import sys

PACKAGES = [
    "mlflow", "sklearn", "pandas", "numpy", "scipy", "joblib",
    "fastapi", "uvicorn", "pydantic", "prometheus_client", "requests",
]

REQUIRED_FILES = [
    "data/raw/train.csv",
    "data/raw/test.csv",
]


def check(label, passed, note=""):
    mark = "[ok]" if passed else "[--]"
    print("  " + mark + "  " + label.ljust(30) + note)
    return passed


def main():
    print("=" * 62)
    print("SETUP CHECK — Jakarta Housing Price Prediction MLOps")
    print("=" * 62)
    all_good = True

    # 1. Python
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 10
    all_good = check("Python 3.10+", ok, platform.python_version()) and all_good

    print("  " + "     " + "machine".ljust(30) + platform.machine()
          + " / " + platform.system())

    # 2. Isolated environment
    conda_name = os.environ.get("CONDA_DEFAULT_ENV", "")
    in_venv = sys.prefix != sys.base_prefix
    active = bool(conda_name) or in_venv
    if conda_name:
        note = "conda: " + conda_name
    elif in_venv:
        note = sys.prefix
    else:
        note = "not active — run: make setup && source .venv/bin/activate"
    all_good = check("virtual environment", active, note) and all_good

    # 3. Packages
    print("\n  Python packages:")
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            installed = getattr(module, "__version__", "ok")
            check(name, True, str(installed))
        except Exception:
            all_good = check(name, False, "not installed") and all_good

    # 4. Raw data
    print("\n  Raw data:")
    for path in REQUIRED_FILES:
        exists = os.path.exists(path)
        if exists:
            size_kb = round(os.path.getsize(path) / 1024)
            note = str(size_kb) + " KB"
        else:
            note = "missing"
        all_good = check(path, exists, note) and all_good

    # 5. Docker
    print("\n  Docker:")
    has_docker = shutil.which("docker") is not None
    if has_docker:
        try:
            result = subprocess.run(["docker", "ps"], capture_output=True,
                                    text=True, timeout=20)
            running = result.returncode == 0
            note = "" if running else "open Docker Desktop first"
            check("Docker daemon running", running, note)
        except Exception:
            check("Docker daemon running", False, "no response")
    else:
        check("docker command", False, "Docker Desktop not installed")

    print("\n" + "=" * 62)
    if all_good:
        print("Ready. Next:  make data && make train")
    else:
        print("Something is missing above.")
        print("  Docker failing does not block the Python parts —")
        print("  only `make docker-up` needs it.")
    print("=" * 62)


if __name__ == "__main__":
    main()
