"""Prepare auditable text-family datasets from owner-supplied JSONL records.

Supported IDs: chat, code, research, document and safety. Non-CHAT text
tokenizers are trained only on their training partition; CHAT must reuse the
exact BASE tokenizer. This tool never approves data and always emits
training_allowed=false and owner_approval=false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from aethrion.tokenizer import ByteBPETokenizer

SUPPORTED = {"chat", "code", "research", "document", "safety"}
SPLITS = ("train", "validation", "test")


class PreparationError(ValueError):
    """Invalid raw records or unsafe preparation settings."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise PreparationError("Input JSONL is missing or empty.")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"Invalid JSON on line {line_no}.") from exc
        if not isinstance(item, dict):
            raise PreparationError(f"Line {line_no} must be a JSON object.")
        rows.append(item)
    if len(rows) < 3:
        raise PreparationError("At least three records are required for train/validation/test.")
    ids = [str(row.get("id", "")).strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise PreparationError("Every record needs a unique, non-empty id.")
    return rows


def _group_id(row: dict[str, Any], *, fallback: str | None = None) -> str:
    value = row.get("group_id")
    if (not isinstance(value, str) or not value.strip()) and fallback:
        value = row.get(fallback)
    if not isinstance(value, str) or not value.strip():
        required = "group_id" if fallback is None else f"group_id or {fallback}"
        raise PreparationError(f"Every record requires a non-empty {required} for leakage-safe splitting.")
    return value.strip()


def _group_hash(group_id: str) -> str:
    """Store only a one-way pseudonymous provenance identifier in prepared NPZ data."""
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()


def _split(sample_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def _pad(values: list[int], length: int) -> list[int]:
    if not values:
        raise PreparationError("Text fields cannot be empty.")
    values = values[:length]
    return values + [0] * (length - len(values))


def _required_text(row: dict[str, Any], key: str, row_id: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"Record {row_id!r} requires non-empty string field {key!r}.")
    return value.strip()


def _build_arrays(model_id: str, rows: list[dict[str, Any]], tokenizer: ByteBPETokenizer,
                  max_length: int, reason_classes: int, seed: int,
                  layout_vocab_size: int, tag_classes: int) -> dict[str, np.ndarray]:
    ids = [str(row["id"]) for row in rows]
    group_ids = [_group_id(row) for row in rows]
    splits = [_split(group_id, seed) for group_id in group_ids]
    group_hashes = [_group_hash(group_id) for group_id in group_ids]
    arrays: dict[str, np.ndarray] = {
        "sample_id": np.asarray(ids, dtype=np.str_),
        "group_id": np.asarray(group_hashes, dtype=np.str_),
        "split": np.asarray(splits, dtype=np.str_),
    }
    if model_id == "chat":
        inputs, labels = [], []
        for row_id, row in zip(ids, rows):
            prompt = _required_text(row, "prompt", row_id)
            response = _required_text(row, "response", row_id)
            prompt_ids = tokenizer.encode(prompt + "\n")
            response_ids = tokenizer.encode(response)
            if not response_ids:
                raise PreparationError(f"Record {row_id!r} has no response tokens.")
            prompt_ids = prompt_ids[:max_length - 1]
            response_ids = response_ids[:max_length - len(prompt_ids)]
            if not response_ids:
                raise PreparationError(
                    f"Record {row_id!r} prompt leaves no room for a supervised response token."
                )
            sequence = prompt_ids + response_ids
            target = [-100] * len(prompt_ids) + response_ids
            inputs.append(_pad(sequence, max_length))
            labels.append(target + [-100] * (max_length - len(target)))
        arrays["input_ids"] = np.asarray(inputs, dtype=np.int64)
        arrays["labels"] = np.asarray(labels, dtype=np.int64)
    elif model_id == "code":
        sequences = [
            _pad(tokenizer.encode(_required_text(row, "text", row_id)), max_length)
            for row_id, row in zip(ids, rows)
        ]
        arrays["token_ids"] = np.asarray(sequences, dtype=np.int64)
    elif model_id == "research":
        queries, passages = [], []
        for row_id, row in zip(ids, rows):
            queries.append(_pad(tokenizer.encode(_required_text(row, "query", row_id)), max_length))
            passages.append(_pad(tokenizer.encode(_required_text(row, "passage", row_id)), max_length))
        arrays["query_ids"] = np.asarray(queries, dtype=np.int64)
        arrays["passage_ids"] = np.asarray(passages, dtype=np.int64)
    elif model_id == "safety":
        tokens, decisions, reasons = [], [], []
        decision_map = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
        for row_id, row in zip(ids, rows):
            tokens.append(_pad(tokenizer.encode(_required_text(row, "text", row_id)), max_length))
            decision = row.get("decision")
            if decision not in decision_map:
                raise PreparationError(f"Record {row_id!r} decision must be ALLOW, REVIEW or BLOCK.")
            raw_reasons = row.get("reasons")
            if not isinstance(raw_reasons, list) or any(type(x) is not int for x in raw_reasons):
                raise PreparationError(f"Record {row_id!r} requires integer reason IDs in a list.")
            if any(x < 0 or x >= reason_classes for x in raw_reasons):
                raise PreparationError(f"Record {row_id!r} has a reason ID outside 0..{reason_classes - 1}.")
            multi_hot = [0] * reason_classes
            for index in raw_reasons:
                multi_hot[index] = 1
            decisions.append(decision_map[decision])
            reasons.append(multi_hot)
        arrays["token_ids"] = np.asarray(tokens, dtype=np.int64)
        arrays["decision_labels"] = np.asarray(decisions, dtype=np.int64)
        arrays["reason_labels"] = np.asarray(reasons, dtype=np.int64)
    elif model_id == "document":
        tokens, boxes, layouts, tags = [], [], [], []
        for row_id, row in zip(ids, rows):
            token_ids = tokenizer.encode(_required_text(row, "text", row_id))[:max_length]
            raw_boxes, raw_layouts, raw_tags = row.get("boxes"), row.get("layout_ids"), row.get("tag_labels")
            if not isinstance(raw_boxes, list) or not isinstance(raw_layouts, list) or not isinstance(raw_tags, list):
                raise PreparationError(f"Record {row_id!r} requires boxes, layout_ids and tag_labels lists.")
            count = len(token_ids)
            if len(raw_boxes) != count or len(raw_layouts) != count or len(raw_tags) != count:
                raise PreparationError(f"Record {row_id!r} layout arrays must match its token count ({count}).")
            if any(not isinstance(box, list) or len(box) != 4 for box in raw_boxes):
                raise PreparationError(f"Record {row_id!r} boxes must be [token,4] normalized coordinates.")
            if any(not isinstance(x, (int, float)) for box in raw_boxes for x in box):
                raise PreparationError(f"Record {row_id!r} boxes must contain numeric coordinates.")
            if any(float(x) < 0 or float(x) > 1 for box in raw_boxes for x in box):
                raise PreparationError(f"Record {row_id!r} boxes must be normalized to [0,1].")
            if any(type(x) is not int for x in raw_layouts + raw_tags):
                raise PreparationError(f"Record {row_id!r} layout and tag IDs must be integers.")
            if any(x < 0 or x >= layout_vocab_size for x in raw_layouts):
                raise PreparationError(f"Record {row_id!r} layout ID exceeds layout_vocab_size.")
            if any(x < 0 or x >= tag_classes for x in raw_tags):
                raise PreparationError(f"Record {row_id!r} tag ID exceeds tag_classes.")
            tokens.append(_pad(token_ids, max_length))
            boxes.append(raw_boxes + [[0.0, 0.0, 0.0, 0.0]] * (max_length - count))
            layouts.append(_pad(raw_layouts, max_length))
            tags.append(_pad(raw_tags, max_length))
        arrays["token_ids"] = np.asarray(tokens, dtype=np.int64)
        arrays["boxes"] = np.asarray(boxes, dtype=np.float32)
        arrays["layout_ids"] = np.asarray(layouts, dtype=np.int64)
        arrays["tag_labels"] = np.asarray(tags, dtype=np.int64)
    return arrays


def prepare_text_dataset(
    input_path: str | Path,
    *,
    model_id: str,
    output_root: str | Path,
    license_name: str,
    permitted_use: str,
    evidence_uri: str,
    source_uri: str,
    seed: int = 1337,
    vocab_size: int = 2048,
    max_length: int = 512,
    reason_classes: int = 16,
    layout_vocab_size: int = 32,
    tag_classes: int = 32,
    tokenizer_source_path: str | Path | None = None,
) -> dict[str, Any]:
    if model_id not in SUPPORTED:
        raise PreparationError(f"Unsupported text preparation model: {model_id}")
    if not all(isinstance(v, str) and v.strip() for v in (license_name, permitted_use, evidence_uri, source_uri)):
        raise PreparationError("License, permitted_use and evidence_uri must be documented; approvals stay false.")
    if max_length < 2 or vocab_size < 256 or reason_classes < 1 or layout_vocab_size < 1 or tag_classes < 1:
        raise PreparationError("Invalid tokenizer or sequence dimensions.")
    if model_id == "chat" and tokenizer_source_path is None:
        raise PreparationError("CHAT instruction preparation requires the exact existing BASE tokenizer via tokenizer_source_path.")
    if model_id != "chat" and tokenizer_source_path is not None:
        raise PreparationError("tokenizer_source_path is supported only for CHAT instruction data.")
    tokenizer_source = Path(tokenizer_source_path).resolve() if tokenizer_source_path is not None else None
    if tokenizer_source is not None and (not tokenizer_source.is_file() or tokenizer_source.stat().st_size == 0):
        raise PreparationError("The supplied BASE tokenizer file is missing or empty.")
    source = Path(input_path).resolve()
    rows = _read_jsonl(source)
    group_ids = [_group_id(row) for row in rows]
    split_names = [_split(group_id, seed) for group_id in group_ids]
    corpus_fields = []
    for row, split_name in zip(rows, split_names):
        if split_name != "train":
            continue
        for key in ("prompt", "response", "text", "query", "passage"):
            if isinstance(row.get(key), str):
                corpus_fields.append(row[key])
    if not corpus_fields:
        raise PreparationError("No training-split text fields found to train the tokenizer.")
    if model_id == "chat":
        try:
            tokenizer = ByteBPETokenizer.load(tokenizer_source)
        except (OSError, UnicodeError, ValueError, KeyError) as exc:
            raise PreparationError("The supplied BASE tokenizer is invalid.") from exc
    else:
        tokenizer = ByteBPETokenizer()
        tokenizer.train("\n".join(corpus_fields), vocab_size=vocab_size)
    arrays = _build_arrays(model_id, rows, tokenizer, max_length, reason_classes, seed,
                           layout_vocab_size, tag_classes)
    split_names = arrays["split"].tolist()
    if any(name not in split_names for name in SPLITS):
        raise PreparationError("Deterministic split produced an empty split; add records or change the seed.")
    if model_id == "safety":
        labels = arrays["decision_labels"]
        for split in SPLITS:
            counts = np.bincount(labels[arrays["split"] == split], minlength=3)
            if np.any(counts < 25):
                raise PreparationError(f"SAFE {split} requires at least 25 samples per ALLOW/REVIEW/BLOCK class.")
    manifests_root = Path(output_root).expanduser().resolve() / "manifests"
    destination = manifests_root / model_id
    manifest_path = manifests_root / f"{model_id}.json"
    # Prepared datasets are provenance-bearing artifacts. Never silently replace
    # an earlier source copy, NPZ, tokenizer or manifest on a rerun.
    if manifests_root.is_symlink() or destination.exists() or destination.is_symlink():
        raise PreparationError(
            f"Refusing to overwrite existing preparation output: {destination}"
        )
    if manifest_path.exists() or manifest_path.is_symlink():
        raise PreparationError(
            f"Refusing to overwrite existing preparation manifest: {manifest_path}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    source_copy = destination / "source.jsonl"
    shutil.copyfile(source, source_copy)
    dataset_path = destination / "dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    tokenizer_path = destination / "tokenizer.json"
    if tokenizer_source is not None:
        shutil.copyfile(tokenizer_source, tokenizer_path)
    else:
        tokenizer.save(tokenizer_path)
    source_record = {
        "source_id": f"{model_id}-input-jsonl",
        "source_uri": source_uri,
        "source_path": f"{model_id}/source.jsonl",
        "sha256": _sha256(source_copy),
        "license": license_name,
        "permitted_use": permitted_use,
        "evidence_uri": evidence_uri,
        "allow_training": False,
        "privacy_reviewed": False,
    }
    manifest = {
        "format": "aethrion-specialist-dataset-v1",
        "dataset_id": f"{model_id}-prepared-{_sha256(dataset_path)[:12]}",
        "model_id": model_id,
        "modality": (
            "text" if model_id in {"chat", "research", "safety"}
            else ("code" if model_id == "code" else "document")
        ),
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "speaker_consent_reviewed": False,
        "dataset_path": f"{model_id}/dataset.npz",
        "dataset_sha256": _sha256(dataset_path),
        "tokenizer_path": f"{model_id}/tokenizer.json",
        "tokenizer_sha256": _sha256(tokenizer_path),
        "sources": [source_record],
        "preparation": {
            "tool": "aethrion.specialist_prepare",
            "seed": seed,
            "max_length": max_length,
            "vocab_size": tokenizer.vocab_size,
            "layout_vocab_size": layout_vocab_size,
            "tag_classes": tag_classes,
            "record_count": len(rows),
            "split_counts": {name: split_names.count(name) for name in SPLITS},
            "automatic_approval": False,
        },
    }
    if model_id == "safety":
        manifest["safety_min_per_class"] = 25
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "model_id": model_id,
        "records": len(rows),
        "split_counts": manifest["preparation"]["split_counts"],
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "tokenizer": str(tokenizer_path),
        "training_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare auditable AETHRION text-specialist NPZ datasets.")
    parser.add_argument("--input", required=True, help="Owner-supplied JSONL source file.")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--output-root", default="data/specialists")
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--permitted-use", required=True)
    parser.add_argument("--evidence-uri", required=True)
    parser.add_argument("--source-uri", required=True, help="Canonical URL or dataset identifier for the reviewed source.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--reason-classes", type=int, default=16)
    parser.add_argument("--layout-vocab-size", type=int, default=32)
    parser.add_argument("--tag-classes", type=int, default=32)
    parser.add_argument(
        "--tokenizer", default=None,
        help="Existing BASE tokenizer path required for --model chat; never retrain it during instruction preparation.",
    )
    args = parser.parse_args()
    result = prepare_text_dataset(
        args.input, model_id=args.model, output_root=args.output_root,
        license_name=args.license_name, permitted_use=args.permitted_use,
        evidence_uri=args.evidence_uri, source_uri=args.source_uri, seed=args.seed, vocab_size=args.vocab_size,
        max_length=args.max_length, reason_classes=args.reason_classes,
        layout_vocab_size=args.layout_vocab_size, tag_classes=args.tag_classes,
        tokenizer_source_path=args.tokenizer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
