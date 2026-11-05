"""One-command, fail-closed Kaggle pipeline for AETHRION-LLM.

Default behavior: environment checks, unit tests, optional bounded import,
review-template generation, and approval-gate validation. Training only runs
when every explicit dataset/source approval is present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = ROOT / "data/processed/corpus_manifest.json"
RAW_DIR = ROOT / "data/raw/cosmos"
SOURCE_MANIFEST = RAW_DIR / "sources.json"
SOURCE_REVIEW = ROOT / "data/processed/privacy_review/source_review_template.csv"
CORPUS = ROOT / "data/processed/corpus.txt"
REPORT = ROOT / "data/processed/corpus-report.json"
TOKENIZER = ROOT / "artifacts/tokenizer.json"
CONFIG = ROOT / "configs/turkish_pilot.json"
HELDOUT = ROOT / "data/heldout/approved-eval.txt"


def run(args: list[str]) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_report() -> None:
    import torch

    total, used, free = shutil.disk_usage(ROOT)
    print("=== AETHRION ENVIRONMENT ===")
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}:", torch.cuda.get_device_name(index))
    print(f"Disk free: {free / (1024 ** 3):.2f} GiB")
    if free < 3 * 1024 ** 3:
        raise RuntimeError("Less than 3 GiB free disk; refusing pipeline execution.")


def run_tests() -> None:
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def import_if_missing(enabled: bool, max_records: int) -> None:
    if SOURCE_MANIFEST.is_file():
        print("Source import skipped: existing source manifest found.")
        return
    if not enabled:
        print("No imported source found. Re-run with --import-if-missing to stream a bounded review sample.")
        return
    run([
        sys.executable, "-m", "aethrion.import_hf_dataset",
        "--output-dir", str(RAW_DIR.relative_to(ROOT)),
        "--max-records", str(max_records),
    ])


def create_pending_domain_template() -> None:
    if not SOURCE_MANIFEST.is_file():
        return
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    domains: set[str] = set()
    for source in payload.get("sources", []):
        source_path = (RAW_DIR / source.get("path", "")).resolve()
        if RAW_DIR.resolve() not in source_path.parents or not source_path.is_file():
            raise RuntimeError("Source manifest contains a missing or unsafe path.")
        with source_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise RuntimeError(f"Invalid source record at {source_path.name}:{line_number}")
                if "url" not in item:
                    continue
                url = item.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise RuntimeError(f"Invalid source URL at {source_path.name}:{line_number}")
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise RuntimeError(f"Invalid source URL at {source_path.name}:{line_number}")
                domains.add(parsed.hostname.lower().rstrip("."))

    SOURCE_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_REVIEW.exists():
        print("Existing domain review preserved; not overwritten.")
        return
    columns = [
        "domain", "license_name", "license_evidence_url", "rights_holder",
        "ai_training_permission", "attribution_required", "review_status",
    ]
    with SOURCE_REVIEW.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for domain in sorted(domains):
            writer.writerow({
                "domain": domain, "license_name": "UNKNOWN",
                "license_evidence_url": "", "rights_holder": "UNKNOWN",
                "ai_training_permission": "NO",
                "attribution_required": "", "review_status": "PENDING",
            })
    print(f"Created {len(domains)} PENDING domain rows: {SOURCE_REVIEW}")


def source_records_have_urls() -> bool:
    """Detect row-level source URLs; source-manifest dataset-card URLs do not count."""
    if not SOURCE_MANIFEST.is_file():
        return False
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    for source in payload.get("sources", []):
        source_path = (RAW_DIR / source.get("path", "")).resolve()
        if RAW_DIR.resolve() not in source_path.parents or not source_path.is_file():
            raise RuntimeError("Source manifest contains a missing or unsafe path.")
        with source_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise RuntimeError(f"Invalid source record at {source_path.name}:{line_number}")
                if "url" in item:
                    url = item.get("url")
                    parsed = urlparse(url) if isinstance(url, str) else None
                    if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname:
                        raise RuntimeError(f"Invalid source URL at {source_path.name}:{line_number}")
                    return True
    return False


def approval_status() -> tuple[bool, str]:
    if not DATASET_MANIFEST.is_file():
        return False, f"Missing dataset manifest: {DATASET_MANIFEST}"
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or dataset.get("training_allowed") is not True:
        return False, "Dataset-level training approval is not explicitly Boolean true."
    if not SOURCE_MANIFEST.is_file():
        return False, "Source manifest is missing."
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8")).get("sources", [])
    if not sources:
        return False, "Source manifest has no sources."
    for source in sources:
        if source.get("allow_training") is not True:
            return False, f"Source not approved: {source.get('source_id', 'unknown')}"
        if not source.get("license") or not source.get("permitted_uses"):
            return False, f"Source license/use metadata incomplete: {source.get('source_id', 'unknown')}"
    has_urls = source_records_have_urls()
    if has_urls:
        if not SOURCE_REVIEW.is_file():
            return False, "URL-attributed data requires the per-domain review CSV."
        with SOURCE_REVIEW.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "domain", "license_name", "license_evidence_url", "rights_holder",
                "ai_training_permission", "attribution_required", "review_status",
            }
            if not required.issubset(set(reader.fieldnames or [])):
                return False, "Domain review CSV is missing required columns."
            rows = list(reader)
        if not rows:
            return False, "Domain review CSV has no reviewed domains."
        for row in rows:
            if (
                (row.get("review_status") or "").strip().upper() != "APPROVED"
                or (row.get("ai_training_permission") or "").strip().upper() not in {"YES", "TRUE"}
                or not (row.get("license_name") or "").strip()
                or (row.get("license_name") or "").strip().upper() == "UNKNOWN"
                or not (row.get("rights_holder") or "").strip()
                or (row.get("rights_holder") or "").strip().upper() == "UNKNOWN"
                or not (row.get("license_evidence_url") or "").strip().startswith("https://")
                or (row.get("attribution_required") or "").strip().upper() not in {"YES", "NO"}
            ):
                return False, f"Domain review is pending or incomplete: {row.get('domain', 'unknown')}"
    return True, "Dataset and source approval metadata passed automated checks."


def main() -> int:
    parser = argparse.ArgumentParser(description="AETHRION one-command Kaggle pipeline")
    parser.add_argument("--import-if-missing", action="store_true",
                        help="Stream at most --max-records rows for manual rights/privacy review.")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--config", default="configs/turkish_pilot.json")
    args = parser.parse_args()

    if args.max_records < 1 or args.max_records > 5000:
        parser.error("--max-records must be between 1 and 5000.")

    environment_report()
    if not args.skip_tests:
        run_tests()

    import_if_missing(args.import_if_missing, args.max_records)
    create_pending_domain_template()

    approved, message = approval_status()
    print("\n=== TRAINING APPROVAL GATE ===")
    print(message)
    if not approved:
        print("SAFE STOP: tests and preflight completed; corpus preparation/tokenization/training were NOT run.")
        print("Review source rights, privacy, attribution, dataset manifest, source manifest, and every domain.")
        return 0

    if not SOURCE_MANIFEST.is_file():
        raise RuntimeError("Approved source manifest disappeared.")
    review_arg = ["--source-review", str(SOURCE_REVIEW)] if source_records_have_urls() else []

    run([
        sys.executable, "-m", "aethrion.prepare_data",
        "--input", str(RAW_DIR),
        "--manifest", str(SOURCE_MANIFEST),
        "--dataset-manifest", str(DATASET_MANIFEST),
        *review_arg,
        "--output", str(CORPUS),
        "--report", str(REPORT),
    ])

    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config.get("data_path") != str(CORPUS.relative_to(ROOT)):
        raise RuntimeError("Training config data_path must point to the approved prepared corpus.")
    run([
        sys.executable, "-m", "aethrion.tokenizer",
        "--train", str(CORPUS),
        "--output", str(TOKENIZER),
        "--vocab-size", str(config["vocab_size"]),
    ])
    run([sys.executable, "-m", "aethrion.train", "--config", args.config])

    checkpoint = ROOT / config["output_dir"] / "checkpoint_last.pt"
    print("\nTraining finished:", checkpoint)
    if HELDOUT.is_file():
        if sha256(HELDOUT) == sha256(CORPUS):
            raise RuntimeError("Held-out evaluation file is identical to training corpus.")
        run([
            sys.executable, "-m", "aethrion.evaluate",
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(TOKENIZER),
            "--data", str(HELDOUT),
        ])
    else:
        print("Independent evaluation skipped: add reviewed data/heldout/approved-eval.txt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
