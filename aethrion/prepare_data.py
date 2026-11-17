"""Build a reproducible, license-gated and deduplicated text corpus."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
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
        if not isinstance(entry["license"], str) or entry["license"].strip().upper() in {
            "UNKNOWN", "PENDING", "PENDING_HUMAN_REVIEW", "UNVERIFIED", "TBD"
        }:
            raise ValueError(f"Source license is not verified: {entry['source_id']}")
        if not isinstance(entry["permitted_uses"], str) or "train" not in entry["permitted_uses"].lower():
            raise ValueError(f"Source permitted_uses must explicitly include training: {entry['source_id']}")

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


def _assign_group_splits(records: list[dict], seed: int) -> dict[str, str]:
    """Assign each provenance group wholly to exactly one deterministic split."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    if len(groups) < 3:
        raise ValueError(
            "Training blocked: at least three independent source groups are required "
            "for train, validation and test. Add source files or record-level group_id metadata."
        )
    names = ("train", "validation", "test")
    targets = {"train": len(records) * 0.80, "validation": len(records) * 0.10, "test": len(records) * 0.10}
    counts = {name: 0 for name in names}
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group_id: len(groups[group_id]), reverse=True)
    assignments: dict[str, str] = {}
    # Seed every partition with a distinct whole group before balancing the remainder.
    for group_id, split_name in zip(ordered[:3], names):
        assignments[group_id] = split_name
        counts[split_name] += len(groups[group_id])
    for group_id in ordered[3:]:
        selected = max(names, key=lambda name: targets[name] - counts[name])
        assignments[group_id] = selected
        counts[selected] += len(groups[group_id])
    return assignments


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
    unique: dict[str, dict[str, str]] = {}
    stats = []
    totals = Counter()
    domain_counts = Counter()
    blocked_domains = set()

    for source in sources:
        path = source["_resolved_path"]
        counts = Counter()
        for line_number, raw, metadata in iter_records(path):
            group_id = f"source:{source['source_id']}"
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
                group_id = f"host:{domain}"
                domain_counts[domain] += 1
                review = domain_reviews.get(domain)
                if review is None or not review["approved"]:
                    blocked_domains.add(domain)
            elif "group_id" in metadata:
                explicit_group = metadata.get("group_id")
                if not isinstance(explicit_group, str) or not explicit_group.strip():
                    raise ValueError(
                        f"Record in {path.name}:{line_number} has an invalid group_id."
                    )
                group_id = f"source:{source['source_id']}:group:{explicit_group.strip()}"

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
            unique[key] = {"text": value, "group_id": group_id}
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
    if len(records) < min_records:
        raise ValueError(
            f"Only {len(records)} unique records remain; minimum required is {min_records}."
        )
    assignments = _assign_group_splits(records, seed)
    random.Random(seed).shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_records(target: Path, selected: list[dict[str, str]]) -> None:
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for item in selected:
                handle.write(item["text"] + "\n")

    split_report = {}
    ordered_split_records = []
    for split_name in ("train", "validation", "test"):
        selected = [item for item in records if assignments[item["group_id"]] == split_name]
        if not selected:
            raise ValueError(f"Training blocked: deterministic group split left {split_name} empty.")
        ordered_split_records.extend(selected)
        split_path = output_path.with_name(
            f"{output_path.stem}-{split_name}{output_path.suffix or '.txt'}"
        )
        _write_records(split_path, selected)
        group_hashes = sorted({
            hashlib.sha256(item["group_id"].encode("utf-8")).hexdigest()
            for item in selected
        })
        split_report[split_name] = {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "bytes": split_path.stat().st_size,
            "records": len(selected),
            "groups": group_hashes,
        }
    # The combined corpus is exactly the concatenation of the three disjoint partitions.
    _write_records(output_path, ordered_split_records)

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
            "records": len(records),
        },
        "split_strategy": "whole-source-or-exact-hostname-group-v1",
        "splits": split_report,
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
    return report



def validate_training_approval(
    dataset_manifest_path: Path,
    source_manifest_path: Path,
    input_dir: Path,
    corpus_path: Path,
    corpus_report_path: Path,
    source_review_path: Path | None = None,
) -> dict[str, dict]:
    """Re-check approvals and prove the corpus and all group-disjoint splits match."""
    require_training_enabled(dataset_manifest_path)
    if not source_manifest_path.is_file():
        raise ValueError(f"Training blocked: source manifest does not exist: {source_manifest_path}")

    sources = load_sources_manifest(source_manifest_path, input_dir)
    if not sources:
        raise ValueError("Training blocked: source manifest contains no approved sources.")

    source_ids = [source["source_id"] for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Training blocked: source manifest contains duplicate source IDs.")

    domain_reviews = load_domain_review(source_review_path) if source_review_path else None
    pending_domains = (
        sorted(domain for domain, review in domain_reviews.items() if not review["approved"])
        if domain_reviews is not None else []
    )
    if pending_domains:
        raise ValueError(
            "Training blocked: source review contains unapproved domains: "
            + ", ".join(pending_domains[:20])
        )

    for source in sources:
        source_path = source["_resolved_path"]
        for line_number, _, metadata in iter_records(source_path):
            if "url" not in metadata:
                continue
            url = metadata.get("url")
            parsed = urlparse(url) if isinstance(url, str) else None
            domain = (parsed.hostname or "").lower().rstrip(".") if parsed else ""
            if parsed is None or parsed.scheme not in {"http", "https"} or not domain:
                raise ValueError(
                    f"Training blocked: invalid source URL in {source_path.name}:{line_number}"
                )
            if domain_reviews is None:
                raise ValueError(
                    "Training blocked: URL-attributed records require --source-review."
                )
            review = domain_reviews.get(domain)
            if review is None or not review["approved"]:
                raise ValueError(
                    f"Training blocked: hostname has no evidence-backed approval: {domain}"
                )

    if not corpus_path.is_file():
        raise ValueError(f"Training blocked: prepared corpus does not exist: {corpus_path}")
    if not corpus_report_path.is_file():
        raise ValueError(f"Training blocked: corpus audit report does not exist: {corpus_report_path}")
    report = json.loads(corpus_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("format") != "aethrion-corpus-report-v2":
        raise ValueError("Training blocked: unsupported or invalid corpus audit report.")

    output = report.get("output")
    if not isinstance(output, dict):
        raise ValueError("Training blocked: corpus audit report has no output metadata.")
    reported_path = Path(str(output.get("path", ""))).resolve()
    if reported_path != corpus_path.resolve():
        raise ValueError("Training blocked: corpus path does not match its audit report.")
    if output.get("sha256") != sha256_file(corpus_path):
        raise ValueError("Training blocked: prepared corpus SHA-256 does not match its audit report.")
    if output.get("bytes") != corpus_path.stat().st_size:
        raise ValueError("Training blocked: prepared corpus size does not match its audit report.")

    reported_sources = report.get("sources")
    if not isinstance(reported_sources, list):
        raise ValueError("Training blocked: corpus audit report has no source list.")
    expected = {
        source["source_id"]: (source["path"], sha256_file(source["_resolved_path"]))
        for source in sources
    }
    observed = {}
    for item in reported_sources:
        if not isinstance(item, dict) or not item.get("source_id") or item["source_id"] in observed:
            raise ValueError("Training blocked: corpus audit report has invalid or duplicate sources.")
        observed[item["source_id"]] = (item.get("path"), item.get("sha256"))
    if observed != expected:
        raise ValueError("Training blocked: corpus audit sources differ from the current approved source manifest.")

    # Reconstruct every normalized record and its provenance group from the
    # currently approved raw sources. This prevents a hand-edited audit report
    # from relabeling validation/test records as training data.
    filters = report.get("filters")
    seed = report.get("seed")
    if (
        not isinstance(filters, dict)
        or type(filters.get("min_chars")) is not int
        or type(filters.get("max_chars")) is not int
        or filters["min_chars"] < 1
        or filters["max_chars"] < filters["min_chars"]
        or filters.get("email_like_records_excluded") is not True
        or type(seed) is not int
    ):
        raise ValueError("Training blocked: corpus report filters or split seed are invalid.")
    reconstructed: dict[str, dict[str, str]] = {}
    for source in sources:
        source_path = source["_resolved_path"]
        for line_number, raw, metadata in iter_records(source_path):
            group_id = f"source:{source['source_id']}"
            if "url" in metadata:
                url = metadata.get("url")
                parsed = urlparse(url) if isinstance(url, str) else None
                domain = (parsed.hostname or "").lower().rstrip(".") if parsed else ""
                if parsed is None or parsed.scheme not in {"http", "https"} or not domain:
                    raise ValueError(
                        f"Training blocked: invalid source URL in {source_path.name}:{line_number}"
                    )
                review = domain_reviews.get(domain) if domain_reviews is not None else None
                if review is None or not review["approved"]:
                    raise ValueError(f"Training blocked: hostname is not approved: {domain}")
                group_id = f"host:{domain}"
            elif "group_id" in metadata:
                explicit_group = metadata.get("group_id")
                if not isinstance(explicit_group, str) or not explicit_group.strip():
                    raise ValueError(
                        f"Training blocked: invalid group_id in {source_path.name}:{line_number}"
                    )
                group_id = f"source:{source['source_id']}:group:{explicit_group.strip()}"
            value = normalize_text(raw)
            if (
                len(value) < filters["min_chars"]
                or len(value) > filters["max_chars"]
                or EMAIL_RE.search(value)
            ):
                continue
            record_hash = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()
            reconstructed.setdefault(record_hash, {"text": value, "group_id": group_id})

    rebuilt_records = list(reconstructed.values())
    if len(rebuilt_records) != report.get("totals", {}).get("records_kept"):
        raise ValueError("Training blocked: raw source records do not reproduce the audited corpus count.")
    rebuilt_assignments = _assign_group_splits(rebuilt_records, seed)
    random.Random(seed).shuffle(rebuilt_records)
    expected_partitions = {
        split_name: [
            row for row in rebuilt_records
            if rebuilt_assignments[row["group_id"]] == split_name
        ]
        for split_name in ("train", "validation", "test")
    }

    split_report = report.get("splits")
    required_splits = {"train", "validation", "test"}
    if not isinstance(split_report, dict) or set(split_report) != required_splits:
        raise ValueError("Training blocked: corpus report requires train, validation and test split metadata.")
    derived = {}
    seen_groups: dict[str, str] = {}
    split_record_total = 0
    combined_digest = hashlib.sha256()
    for split_name in ("train", "validation", "test"):
        item = split_report.get(split_name)
        if not isinstance(item, dict):
            raise ValueError(f"Training blocked: invalid {split_name} split metadata.")
        expected_path = corpus_path.with_name(
            f"{corpus_path.stem}-{split_name}{corpus_path.suffix or '.txt'}"
        ).resolve()
        split_path = Path(str(item.get("path", ""))).resolve()
        if split_path != expected_path or not split_path.is_file():
            raise ValueError(f"Training blocked: {split_name} split path is missing or unexpected.")
        if item.get("sha256") != sha256_file(split_path):
            raise ValueError(f"Training blocked: {split_name} split SHA-256 mismatch.")
        if item.get("bytes") != split_path.stat().st_size:
            raise ValueError(f"Training blocked: {split_name} split size mismatch.")
        with split_path.open("rb") as split_handle:
            for chunk in iter(lambda: split_handle.read(1024 * 1024), b""):
                combined_digest.update(chunk)
        expected_rows = expected_partitions[split_name]
        expected_bytes = "".join(row["text"] + "\n" for row in expected_rows).encode("utf-8")
        if split_path.read_bytes() != expected_bytes:
            raise ValueError(
                f"Training blocked: {split_name} split content does not reproduce from reviewed source records."
            )
        expected_group_hashes = sorted({
            hashlib.sha256(row["group_id"].encode("utf-8")).hexdigest()
            for row in expected_rows
        })
        record_count = item.get("records")
        groups = item.get("groups")
        if type(record_count) is not int or record_count != len(expected_rows) or record_count < 1:
            raise ValueError(f"Training blocked: {split_name} split has an invalid record count.")
        if groups != expected_group_hashes:
            raise ValueError(f"Training blocked: {split_name} provenance group list does not match raw sources.")
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"Training blocked: {split_name} split has no provenance groups.")
        if len(set(groups)) != len(groups) or any(
            not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in groups
        ):
            raise ValueError(f"Training blocked: {split_name} split group hashes are invalid.")
        for group_hash in groups:
            previous = seen_groups.get(group_hash)
            if previous is not None:
                raise ValueError(
                    f"Training blocked: provenance group appears in both {previous} and {split_name}."
                )
            seen_groups[group_hash] = split_name
        split_record_total += record_count
        derived[split_name] = {
            "path": split_path,
            "sha256": item["sha256"],
            "records": record_count,
            "groups": groups,
        }
    if split_record_total != report.get("totals", {}).get("records_kept"):
        raise ValueError("Training blocked: split record counts do not cover the approved corpus.")
    if combined_digest.hexdigest() != sha256_file(corpus_path):
        raise ValueError("Training blocked: combined corpus is not the exact concatenation of approved splits.")
    return derived


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
