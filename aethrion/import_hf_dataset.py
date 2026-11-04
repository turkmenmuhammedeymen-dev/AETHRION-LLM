"""Stream a bounded, attributed sample from the public Cosmos Turkish corpus."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

DATASET_ID = "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0"
DATASET_URL = "https://huggingface.co/datasets/ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0"
LICENSE = "CC-BY-4.0"

def main() -> None:
    parser = argparse.ArgumentParser(description="Import a bounded sample of the Cosmos Turkish corpus.")
    parser.add_argument("--output-dir", default="data/raw/cosmos")
    parser.add_argument("--max-records", type=int, default=5_000,
                        help="Maximum records to stream; default is 5,000, not the full 22.8 GB corpus.")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install the optional data dependency: pip install -e ".[data]"') from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "cosmos_sample.jsonl"
    dataset = load_dataset(DATASET_ID, split=args.split, streaming=True)
    written = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in islice(dataset, args.max_records):
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        output_file.unlink(missing_ok=True)
        raise SystemExit("No usable text records were returned; check the dataset schema/network.")
    manifest = {
        "sources": [{
            "source_id": "cosmos-turkish-corpus-v1.0",
            "path": output_file.name,
            "license": LICENSE,
            "permitted_uses": "Attribution required; verify upstream dataset card and all component rights before training or redistribution.",
            "allow_training": False,
            "url": DATASET_URL,
            "dataset_id": DATASET_ID,
            "attribution": "Cosmos Research Group / Yıldız Technical University; cite the dataset card",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "records_imported": written,
            "split": args.split,
            "note": "Bounded streaming sample. Upstream corpus is approximately 22.8 GB; this command does not fetch the full corpus."
        }]
    }
    (output_dir / "sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Imported {written:,} records to {output_file}")
    print(f"Manifest: {output_dir / 'sources.json'}")
    print("Training is NOT approved by default. Review all rights, attribution, privacy, and content.")
    print('Only after approval, change "allow_training" to true in sources.json.')

if __name__ == "__main__":
    main()
