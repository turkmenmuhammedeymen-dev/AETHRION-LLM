"""Prepare first-party IMAGE, VISION and point-cloud 3D datasets.

All generated manifests remain unapproved. The tool records original assets
and hashes, but does not establish their license, privacy or attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np

from aethrion.specialist_prepare import _split
from aethrion.tokenizer import ByteBPETokenizer


class VisualPreparationError(ValueError):
    """Invalid image or point-cloud records."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VisualPreparationError("Visual JSONL is missing or empty.")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VisualPreparationError(f"Invalid JSON on line {line_no}.") from exc
        if not isinstance(row, dict):
            raise VisualPreparationError(f"Line {line_no} must be a JSON object.")
        rows.append(row)
    ids = [str(row.get("id", "")).strip() for row in rows]
    if len(rows) < 3 or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise VisualPreparationError("At least three records with unique, non-empty IDs are required.")
    return rows


def _resolve_asset(root: Path, value: Any, row_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise VisualPreparationError(f"Record {row_id!r} requires an asset path.")
    path = Path(value)
    if path.is_absolute():
        raise VisualPreparationError("Asset paths must be relative to the JSONL source folder.")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise VisualPreparationError(f"Asset path for {row_id!r} escapes the source folder.") from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise VisualPreparationError(f"Asset for {row_id!r} is missing or empty.")
    return resolved


def _safe_id(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:48] or "sample"
    return f"{stem}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def prepare_visual_dataset(
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
    max_text_length: int = 128,
    image_size: int | None = None,
    point_count: int = 512,
) -> dict[str, Any]:
    if model_id not in {"vision", "image", "3d"}:
        raise VisualPreparationError("model_id must be vision, image or 3d.")
    if not all(isinstance(value, str) and value.strip()
               for value in (license_name, permitted_use, evidence_uri, source_uri)):
        raise VisualPreparationError("License, permitted use, evidence and canonical source URI are required.")
    if vocab_size < 256 or max_text_length < 2:
        raise VisualPreparationError("Invalid tokenizer vocabulary or text length.")
    if model_id == "3d" and point_count < 16:
        raise VisualPreparationError("3D point_count must be at least 16.")
    if model_id in {"vision", "image"}:
        expected_size = 224 if model_id == "vision" else 64
        image_size = expected_size if image_size is None else image_size
        if image_size != expected_size:
            raise VisualPreparationError(f"{model_id} image_size must match its model config ({expected_size}).")
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise VisualPreparationError("Install the optional dependency with: pip install -e '.[visual]'") from exc
    source = Path(input_path).resolve()
    rows = _read_rows(source)
    root = source.parent.resolve()
    texts = []
    assets = []
    for row in rows:
        row_id = str(row["id"])
        field = "point_path" if model_id == "3d" else "image_path"
        asset = _resolve_asset(root, row.get(field), row_id)
        if model_id == "3d":
            try:
                points = np.load(asset, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise VisualPreparationError(f"Point cloud for {row_id!r} is not a safe NPY array.") from exc
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 16:
                raise VisualPreparationError(f"Point cloud for {row_id!r} must have shape [points>=16,3].")
            points = np.asarray(points, dtype=np.float32)
            if not np.isfinite(points).all() or np.max(np.abs(points)) > 1.0:
                raise VisualPreparationError(f"Point cloud for {row_id!r} must be finite and normalized to [-1,1].")
        else:
            if asset.stat().st_size > 20 * 1024 * 1024:
                raise VisualPreparationError(f"Image for {row_id!r} exceeds the 20 MiB source limit.")
        caption = row.get("text" if model_id == "3d" else "caption")
        if not isinstance(caption, str) or not caption.strip():
            raise VisualPreparationError(f"Record {row_id!r} requires a non-empty text condition.")
        texts.append(caption.strip())
        assets.append((row, asset))

    tokenizer = ByteBPETokenizer()
    tokenizer.train("\n".join(texts), vocab_size=vocab_size)
    token_rows = [tokenizer.encode(text)[:max_text_length] for text in texts]
    if any(not item for item in token_rows):
        raise VisualPreparationError("Text conditions must produce at least one token.")
    token_ids = np.zeros((len(rows), max_text_length), dtype=np.int64)
    for index, tokens in enumerate(token_rows):
        token_ids[index, :len(tokens)] = tokens

    destination = Path(output_root).resolve() / "manifests" / model_id
    asset_dir = destination / ("point_clouds" if model_id == "3d" else "images")
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_copy = destination / "source.jsonl"
    shutil.copyfile(source, source_copy)
    manifest_sources = [{
        "source_id": f"{model_id}-metadata-jsonl",
        "source_uri": source_uri,
        "source_path": f"{model_id}/source.jsonl",
        "sha256": _sha256(source_copy),
        "license": license_name,
        "permitted_use": permitted_use,
        "evidence_uri": evidence_uri,
        "allow_training": False,
        "privacy_reviewed": False,
    }]

    if model_id == "3d":
        points_array = np.zeros((len(rows), point_count, 3), dtype=np.float32)
        for index, (row, asset) in enumerate(assets):
            points = np.asarray(np.load(asset, allow_pickle=False), dtype=np.float32)
            rng = np.random.default_rng(seed + index)
            if len(points) >= point_count:
                selected = rng.choice(len(points), size=point_count, replace=False)
            else:
                selected = rng.choice(len(points), size=point_count, replace=True)
            points_array[index] = points[selected]
        arrays = {"points": points_array, "token_ids": token_ids}
    else:
        images = np.zeros((len(rows), 3, image_size, image_size), dtype=np.uint8)
        for index, (row, asset) in enumerate(assets):
            try:
                with Image.open(asset) as image:
                    image.load()
                    fitted = ImageOps.fit(image.convert("RGB"), (image_size, image_size), method=Image.Resampling.BICUBIC)
                    images[index] = np.asarray(fitted, dtype=np.uint8).transpose(2, 0, 1)
            except Exception as exc:
                raise VisualPreparationError(f"Image for {row['id']!r} could not be decoded safely.") from exc
        arrays = {"images": images, "token_ids": token_ids}

    arrays["sample_id"] = np.asarray([str(row["id"]) for row in rows], dtype=np.str_)
    arrays["split"] = np.asarray(
        [_split(str(row.get("group_id", row["id"])), seed) for row in rows], dtype=np.str_
    )
    split_names = arrays["split"].tolist()
    if any(name not in split_names for name in ("train", "validation", "test")):
        raise VisualPreparationError("Split has an empty partition; add samples or choose a different seed.")

    for row, asset in assets:
        suffix = ".npy" if model_id == "3d" else ".source"
        name = _safe_id(str(row["id"])) + suffix
        copied = asset_dir / name
        shutil.copyfile(asset, copied)
        manifest_sources.append({
            "source_id": f"asset-{_safe_id(str(row['id']))}",
            "source_uri": str(row.get("source_uri") or source_uri),
            "source_path": f"{model_id}/{'point_clouds' if model_id == '3d' else 'images'}/{name}",
            "sha256": _sha256(copied),
            "license": license_name,
            "permitted_use": permitted_use,
            "evidence_uri": evidence_uri,
            "allow_training": False,
            "privacy_reviewed": False,
        })

    dataset = destination / "dataset.npz"
    np.savez_compressed(dataset, **arrays)
    tokenizer_path = destination / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    manifest_path = destination.parent / f"{model_id}.json"
    manifest = {
        "format": "aethrion-specialist-dataset-v1",
        "dataset_id": f"{model_id}-prepared-{_sha256(dataset)[:12]}",
        "model_id": model_id,
        "modality": "3d" if model_id == "3d" else model_id,
        "training_allowed": False,
        "owner_approval": False,
        "privacy_reviewed": False,
        "speaker_consent_reviewed": False,
        "dataset_path": f"{model_id}/dataset.npz",
        "dataset_sha256": _sha256(dataset),
        "tokenizer_path": f"{model_id}/tokenizer.json",
        "tokenizer_sha256": _sha256(tokenizer_path),
        "sources": manifest_sources,
        "preparation": {
            "tool": "aethrion.specialist_prepare_visual",
            "seed": seed,
            "vocab_size": tokenizer.vocab_size,
            "max_text_length": max_text_length,
            "image_size": image_size,
            "point_count": point_count if model_id == "3d" else None,
            "record_count": len(rows),
            "split_counts": {name: split_names.count(name) for name in ("train", "validation", "test")},
            "automatic_approval": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "model_id": model_id,
        "records": len(rows),
        "split_counts": manifest["preparation"]["split_counts"],
        "dataset": str(dataset),
        "manifest": str(manifest_path),
        "tokenizer": str(tokenizer_path),
        "training_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AETHRION vision, image and point-cloud datasets.")
    parser.add_argument("--input", required=True, help="JSONL with id, asset path and caption/text.")
    parser.add_argument("--model", required=True, choices=("vision", "image", "3d"))
    parser.add_argument("--output-root", default="data/specialists")
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--permitted-use", required=True)
    parser.add_argument("--evidence-uri", required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--point-count", type=int, default=512)
    args = parser.parse_args()
    result = prepare_visual_dataset(
        args.input, model_id=args.model, output_root=args.output_root,
        license_name=args.license_name, permitted_use=args.permitted_use,
        evidence_uri=args.evidence_uri, source_uri=args.source_uri, seed=args.seed,
        vocab_size=args.vocab_size, max_text_length=args.max_text_length,
        image_size=args.image_size, point_count=args.point_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
