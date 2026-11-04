"""Build a reproducible, license-gated and deduplicated text corpus."""
from __future__ import annotations
import argparse, hashlib, json, random, re, unicodedata
from collections import Counter
from pathlib import Path

EMAIL_RE = re.compile(r"(?i)\\b[\\w.+-]+@[\\w.-]+\\.[a-z]{2,}\\b")

def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).replace("\\x00", " ").split()).strip()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def load_sources_manifest(path: Path, root: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries: raise ValueError("Manifest must contain a non-empty sources list.")
    root = root.resolve(); approved = []; seen = set()
    for entry in entries:
        if not isinstance(entry, dict): raise ValueError("Each source entry must be an object.")
        required = ("path", "source_id", "license", "permitted_uses", "allow_training")
        missing = [key for key in required if not entry.get(key)]
        if missing: raise ValueError(f"Source entry missing: {', '.join(missing)}")
        if entry["allow_training"] is not True: raise ValueError(f"Source is not approved for training: {entry['source_id']}")
        source = (root / entry["path"]).resolve()
        if root not in source.parents: raise ValueError(f"Source path escapes input directory: {entry['path']}")
        if source in seen: raise ValueError(f"Duplicate source path: {entry['path']}")
        if not source.is_file() or source.suffix.lower() not in {".txt", ".jsonl"}:
            raise ValueError(f"Missing or unsupported source file: {entry['path']}")
        seen.add(source); approved.append({**entry, "_resolved_path": source})
    return approved

def iter_records(path: Path):
    with path.open(encoding="utf-8-sig", errors="strict") as handle:
        for number, line in enumerate(handle, 1):
            if path.suffix.lower() == ".txt": yield number, line; continue
            if not line.strip(): continue
            try: item = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSONL at {path.name}:{number}: {exc}") from exc
            value = item.get("text") if isinstance(item, dict) else None
            if isinstance(value, str): yield number, value

def build_corpus(input_dir: Path, manifest_path: Path, output_path: Path, report_path: Path,
                 seed: int = 1337, min_chars: int = 20, max_chars: int = 12000) -> dict:
    if min_chars < 1 or max_chars < min_chars: raise ValueError("Require 1 <= min_chars <= max_chars.")
    sources = load_sources_manifest(manifest_path, input_dir); unique = {}; stats = []; totals = Counter()
    for source in sources:
        path = source["_resolved_path"]; counts = Counter()
        for _, raw in iter_records(path):
            totals["records_read"] += 1; counts["records_read"] += 1; value = normalize_text(raw)
            if len(value) < min_chars: totals["too_short"] += 1; counts["too_short"] += 1; continue
            if len(value) > max_chars: totals["too_long"] += 1; counts["too_long"] += 1; continue
            if EMAIL_RE.search(value): totals["email_like_records_excluded"] += 1; counts["email_like_records_excluded"] += 1; continue
            key = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()
            if key in unique: totals["duplicates"] += 1; counts["duplicates"] += 1; continue
            unique[key] = value; totals["records_kept"] += 1; counts["records_kept"] += 1
        stats.append({"source_id":source["source_id"],"path":source["path"],"license":source["license"],
          "permitted_uses":source["permitted_uses"],"url":source.get("url"),"retrieved_at":source.get("retrieved_at"),
          "sha256":sha256_file(path),**dict(counts)})
    records = list(unique.values()); random.Random(seed).shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\\n") as handle:
        for value in records: handle.write(value + "\\n")
    report = {"format":"aethrion-corpus-report-v1","seed":seed,
      "normalization":"Unicode NFC + whitespace collapse",
      "deduplication":"SHA-256 of casefolded normalized record",
      "filters":{"min_chars":min_chars,"max_chars":max_chars,"email_like_records_excluded":True},
      "totals":dict(totals),"output":{"path":str(output_path),"sha256":sha256_file(output_path),"bytes":output_path.stat().st_size},
      "sources":stats,"warning":"Automated filters do not replace human license, privacy, or quality review."}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
    if len(records) < 100: raise ValueError(f"Only {len(records)} unique records remain; add more approved source text.")
    return report

def main():
    parser=argparse.ArgumentParser(description="Prepare an approved AETHRION text corpus.")
    parser.add_argument("--input",required=True); parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",default="data/processed/corpus.txt")
    parser.add_argument("--report",default="data/processed/corpus-report.json")
    parser.add_argument("--seed",type=int,default=1337); parser.add_argument("--min-chars",type=int,default=20)
    parser.add_argument("--max-chars",type=int,default=12000); args=parser.parse_args()
    report=build_corpus(Path(args.input),Path(args.manifest),Path(args.output),Path(args.report),args.seed,args.min_chars,args.max_chars)
    print(json.dumps(report["totals"],ensure_ascii=False,sort_keys=True))
    print(f"Corpus: {args.output} | SHA-256: {report['output']['sha256']}"); print(f"Audit report: {args.report}")

if __name__ == "__main__": main()
