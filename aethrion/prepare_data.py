"""Build a reproducible, license-gated and deduplicated text corpus."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
REVIEW_COLUMNS = {
    "domain",
    "license_name",
    "license_evidence_url",
    "rights_holder",
    "ai_training_permission",
    "attribution_required",
    "review_status",
}


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).replace("\x00", " ").split()).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_training_enabled(path: Path) -> dict:
    """Fail closed unless the dataset-level manifest explicitly enables training."""
    if not path.is_file():
        raise ValueError(f"Dataset manifest does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset manifest must be a JSON object.")

    if payload.get("training_allowed") is not True:
        raise ValueError(
            "Training blocked: dataset manifest must explicitly set "
            "training_allowed to true."
        )

    return payload


def load_sources_manifest(path: Path, root: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a non-empty sources list.")

    root = root.resolve()
    approved = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each source entry must be an object.")
        required = ("path", "source_id", "license", "permitted_uses", "allow_training")
        missing = [key for key in required if not entry.get(key)]
        if missing:
            raise ValueError(f"Source entry missing: {', '.join(missing)}")
        if entry["allow_training"] is not True:
            raise ValueError(f"Source is not approved for training: {entry['source_id']}")

        source = (root / entry["path"]).resolve()
        if root not in source.parents:
            raise ValueError(f"Source path escapes input directory: {entry['path']}")
        if source in seen:
            raise ValueError(f"Duplicate source path: {entry['path']}")
        if not source.is_file() or source.suffix.lower() not in {".txt", ".jsonl"}:
            raise ValueError(f"Missing or unsupported source file: {entry['path']}")
        seen.add(source)
        approved.append({**entry, "_resolved_path": source})
    return approved


def load_domain_review(path: Path) -> dict[str, dict]:
    """Load explicit, evidence-backed approval for every exact URL hostname."""
    if not path.is_file():
        raise ValueError(f"Source review file does not exist: {path}")

    reviews = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REVIEW_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Source review is missing required columns: " + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, start=2):
            domain = (row.get("domain") or "").strip().lower().rstrip(".")
            if not domain:
                raise ValueError(f"Source review has an empty domain at line {line_number}.")
            if domain in reviews:
                raise ValueError(f"Duplicate domain in source review: {domain}")

            license_name = (row.get("license_name") or "").strip()
            evidence = (row.get("license_evidence_url") or "").strip()
            rights_holder = (row.get("rights_holder") or "").strip()
            training = (row.get("ai_training_permission") or "").strip().upper()
            attribution = (row.get("attribution_required") or "").strip().upper()
            status = (row.get("review_status") or "").strip().upper()

            parsed_evidence = urlparse(evidence)
            evidence_is_url = parsed_evidence.scheme == "https" and bool(parsed_evidence.netloc)
            approved = (
                status == "APPROVED"
                and training in {"YES", "TRUE"}
                and bool(license_name) and license_name.upper() != "UNKNOWN"
                and evidence_is_url
                and bool(rights_holder) and rights_holder.upper() != "UNKNOWN"
                and attribution in {"YES", "NO"}
            )
            reviews[domain] = {
                "approved": approved,
                "license_name": license_name,
                "license_evidence_url": evidence,
                "rights_holder": rights_holder,
                "attribution_required": attribution,
                "review_status": status or "PENDING",
            }
    return reviews


def iter_records(path: Path):
    with path.open(encoding="utf-8-sig", errors="strict") as handle:
        for number, line in enumerate(handle, 1):
            if path.suffix.lower() == ".txt":
                yield number, line, {}
                continue
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path.name}:{number}: {exc}") from exc
            value = item.get("text") if isinstance(item, dict) else None
            if isinstance(value, str):
                yield number, value, item


def build_corpus(
    input_dir: Path,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    seed: int = 1337,
    min_chars: int = 20,
    max_chars: int = 12000,
    min_records: int = 100,
    source_review_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
) -> dict:
    if min_chars < 1 or max_chars < min_chars or min_records < 1:
        raise ValueError("Invalid corpus size limits.")

    if dataset_manifest_path is None:
        raise ValueError("Training blocked: dataset_manifest_path is required.")
    require_training_enabled(dataset_manifest_path)

    sources = load_sources_manifest(manifest_path, input_dir)
    domain_reviews = (
        load_domain_review(source_review_path) if source_review_path is not None else None
    )
    unique = {}
    stats = []
    totals = Counter()
    domain_counts = Counter()
    blocked_domains = set()

    for source in sources:
        path = source["_resolved_path"]
        counts = Counter()
        for _, raw, metadata in iter_records(path):
            totals["records_read"] += 1
            counts["records_read"] += 1

            # URL-attributed JSONL is mixed-source data: every exact hostname must
            # have its own evidence-backed approval, in addition to the dataset gate.
            if "url" in metadata:
                url = metadata.get("url")
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(
                        f"Record in {path.name} has an empty URL; refusing unattributed data."
                    )
                parsed = urlparse(url)
                domain = (parsed.hostname or "").lower().rstrip(".")
                if parsed.scheme not in {"http", "https"} or not domain:
                    raise ValueError(
                        f"Record in {path.name} has an invalid source URL; refusing it."
                    )
                if domain_reviews is None:
                    raise ValueError(
                        "URL-attributed JSONL requires --source-review with explicit "
                        "approval for every source hostname."
                    )
                domain_counts[domain] += 1
                review = domain_reviews.get(domain)
                if review is None or not review["approved"]:
                    blocked_domains.add(domain)

            value = normalize_text(raw)
            if len(value) < min_chars:
                totals["too_short"] += 1
                counts["too_short"] += 1
                continue
            if len(value) > max_chars:
                totals["too_long"] += 1
                counts["too_long"] += 1
                continue
            if EMAIL_RE.search(value):
                totals["email_like_records_excluded"] += 1
                counts["email_like_records_excluded"] += 1
                continue

            key = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()
            if key in unique:
                totals["duplicates"] += 1
                counts["duplicates"] += 1
                continue
            unique[key] = value
            totals["records_kept"] += 1
            counts["records_kept"] += 1

        stats.append({
            "source_id": source["source_id"],
            "path": source["path"],
            "license": source["license"],
            "permitted_uses": source["permitted_uses"],
            "url": source.get("url"),
            "retrieved_at": source.get("retrieved_at"),
            "sha256": sha256_file(path),
            **dict(counts),
        })

    if blocked_domains:
        sample = ", ".join(sorted(blocked_domains)[:20])
        suffix = " ..." if len(blocked_domains) > 20 else ""
        raise ValueError(
            "Training blocked: these URL hostnames are missing an evidence-backed "
            f"APPROVED review: {sample}{suffix}"
        )

    records = list(unique.values())
    random.Random(seed).shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in records:
            handle.write(value + "\n")

    domain_report = []
    if domain_reviews is not None:
        for domain, count in sorted(domain_counts.items()):
            review = domain_reviews[domain]
            domain_report.append({
                "domain": domain,
                "records": count,
                "license_name": review["license_name"],
                "license_evidence_url": review["license_evidence_url"],
                "rights_holder": review["rights_holder"],
                "attribution_required": review["attribution_required"],
                "review_status": review["review_status"],
            })

    report = {
        "format": "aethrion-corpus-report-v2",
        "seed": seed,
        "normalization": "Unicode NFC + whitespace collapse",
        "deduplication": "SHA-256 of casefolded normalized record",
        "filters": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "email_like_records_excluded": True,
        },
        "totals": dict(totals),
        "domain_review": domain_report,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "sources": stats,
        "warning": (
            "Automated checks do not replace human license, privacy, or quality review. "
            "URL-attributed records require an approved exact-host review."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(records) < min_records:
        raise ValueError(
            f"Only {len(records)} unique records remain; minimum required is {min_records}."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description="Prepare an approved AETHRION text corpus.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Dataset-level JSON manifest; training_allowed must be true.",
    )
    parser.add_argument(
        "--source-review",
        default=None,
        help="CSV with evidence-backed approval for every URL hostname.",
    )
    parser.add_argument("--output", default="data/processed/corpus.txt")
    parser.add_argument("--report", default="data/processed/corpus-report.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--min-records", type=int, default=100)
    args = parser.parse_args()
    report = build_corpus(
        Path(args.input),
        Path(args.manifest),
        Path(args.output),
        Path(args.report),
        args.seed,
        args.min_chars,
        args.max_chars,
        args.min_records,
        Path(args.source_review) if args.source_review else None,
        Path(args.dataset_manifest),
    )
    print(json.dumps(report["totals"], ensure_ascii=False, sort_keys=True))
    print(f"Corpus: {args.output} | SHA-256: {report['output']['sha256']}")
    print(f"Audit report: {args.report}")


if __name__ == "__main__":
    main()
