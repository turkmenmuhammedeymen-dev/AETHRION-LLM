"""Read-only Codespaces preflight for resources, data gates and model artifacts."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def _memory_bytes() -> tuple[int | None, int | None]:
    total = available = None
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0]) * 1024
        total = fields.get("MemTotal")
        available = fields.get("MemAvailable")
    except (OSError, ValueError, IndexError):
        pass
    v2 = Path("/sys/fs/cgroup")
    limit = _read_int(v2 / "memory.max")
    current = _read_int(v2 / "memory.current")
    if limit is None:
        limit = _read_int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        current = _read_int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if limit and limit < (1 << 60):
        total = min(total, limit) if total else limit
        if current is not None:
            available = max(0, limit - current)
    return total, available


def _cpu_quota() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="ascii").split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    quota = _read_int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _read_int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota and period:
        return quota / period
    return None


def _gib(value: int | None) -> str:
    return "unknown" if value is None else f"{value / (1024 ** 3):.2f} GiB"


def main() -> int:
    print("=== AETHRION CODESPACES PREFLIGHT (READ ONLY) ===")
    print("Python:", platform.python_version())
    print("Platform:", platform.platform())
    logical = os.cpu_count() or 1
    quota = _cpu_quota()
    effective = min(logical, quota) if quota else logical
    print(f"CPU: {logical} logical; effective container limit: {effective:.2f}")
    total_ram, available_ram = _memory_bytes()
    print("RAM total/limit:", _gib(total_ram))
    print("RAM available:", _gib(available_ram))
    disk = shutil.disk_usage(ROOT)
    print("Disk total:", _gib(disk.total))
    print("Disk used:", _gib(disk.used))
    print("Disk free:", _gib(disk.free))
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"GPU {index}:", torch.cuda.get_device_name(index))
    except Exception as exc:
        print("PyTorch check failed:", type(exc).__name__, str(exc))
    corpus_path = ROOT / "data/processed/corpus_manifest.json"
    if corpus_path.is_file():
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        print("Global dataset status:", corpus.get("dataset_status", "unknown"))
        print("Global training_allowed:", corpus.get("training_allowed") is True)
    else:
        print("Global dataset manifest: MISSING (training blocked)")
    model_path = ROOT / "models/manifest.json"
    if model_path.is_file():
        manifest = json.loads(model_path.read_text(encoding="utf-8"))
        models = manifest.get("models", {})
        counts = {}
        for model_id, entry in models.items():
            status = entry.get("status", "missing")
            counts[status] = counts.get(status, 0) + 1
            checkpoint = ROOT / entry.get("checkpoint", "")
            print(f"Model {model_id}: {status}; checkpoint_present={checkpoint.is_file()}")
        print("Model status counts:", json.dumps(counts, sort_keys=True))
    else:
        print("Model manifest: MISSING")
    if disk.free < 3 * 1024 ** 3:
        print("WARNING: less than 3 GiB free; do not download datasets.")
    if not quota or quota >= 4:
        print("CPU note: effective CPU count is a runtime limit, not a GPU.")
    print("RESULT: diagnostics only; no files changed, no data downloaded, no training started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
