"""Collect UTF-8 text and JSONL records containing a 'text' field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".jsonl"}:
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                cleaned = " ".join(line.split())
                if cleaned:
                    yield cleaned
        else:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    value = item.get("text") if isinstance(item, dict) else None
                    if isinstance(value, str) and value.strip():
                        yield " ".join(value.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if not source.exists():
        raise SystemExit(f"Input directory does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as out:
        for line in iter_text_files(source):
            out.write(line + "\n")
            count += 1
    if count == 0:
        raise SystemExit("No usable text found. Add licensed UTF-8 .txt or .jsonl files.")
    print(f"Wrote {count:,} text lines to {target}")


if __name__ == "__main__":
    main()
